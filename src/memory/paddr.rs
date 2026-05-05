// Physical memory implementation

use crate::common::{PAddr, Word};
use crate::generated::config::*;
use std::collections::HashSet;
use std::time::Instant;
// use std::sync::{Arc, Mutex}; // Removed Mutex lock

// Memory regions
const MROM_BASE: PAddr = 0x20000000;
const MROM_SIZE: usize = 0x1000; // 4KB

const SRAM_BASE: PAddr = 0x0f000000;
const SRAM_SIZE: usize = 0x2000; // 8KB
pub const PMEM_PAGE_SIZE: usize = 4096;

pub struct MemorySnapshot {
    pub pmem: Vec<u8>,
    pub mrom: Vec<u8>,
    pub sram: Vec<u8>,
    pub mbase: PAddr,
    pub msize: usize,
}

struct LazyBacking {
    page_size: usize,
    pages: Vec<Vec<u8>>,
    resident: Vec<bool>,
    prefetched: Vec<bool>,
}

pub struct PhysicalMemory {
    pub pmem: Vec<u8>,
    pub mrom: Vec<u8>,
    pub sram: Vec<u8>,
    pub mbase: PAddr,
    pub msize: usize,
    lazy: Option<LazyBacking>,
}

impl PhysicalMemory {
    pub fn new(mbase: PAddr, msize: usize) -> Self {
        let mut pmem = vec![0u8; msize];

        // Random initialization if configured
        if MEM_RANDOM {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32;

            // Simple random fill
            for i in 0..pmem.len() {
                pmem[i] = ((seed.wrapping_mul(1103515245).wrapping_add(i as u32)) >> 8) as u8;
            }
        }

        Self {
            pmem,
            mrom: vec![0u8; MROM_SIZE],
            sram: vec![0u8; SRAM_SIZE],
            mbase,
            msize,
            lazy: None,
        }
    }

    pub fn guest_to_host(&self, paddr: PAddr) -> Option<*mut u8> {
        if self.in_mrom(paddr) {
            let offset = (paddr - MROM_BASE) as usize;
            Some(self.mrom.as_ptr().wrapping_add(offset) as *mut u8)
        } else if self.in_sram(paddr) {
            let offset = (paddr - SRAM_BASE) as usize;
            Some(self.sram.as_ptr().wrapping_add(offset) as *mut u8)
        } else if self.in_pmem(paddr) {
            let offset = (paddr - self.mbase) as usize;
            Some(self.pmem.as_ptr().wrapping_add(offset) as *mut u8)
        } else {
            None
        }
    }

    #[inline]
    fn in_pmem(&self, addr: PAddr) -> bool {
        addr >= self.mbase && addr < self.mbase + self.msize as PAddr
    }

    #[inline]
    fn in_mrom(&self, addr: PAddr) -> bool {
        addr >= MROM_BASE && addr < MROM_BASE + MROM_SIZE as PAddr
    }

    #[inline]
    fn in_sram(&self, addr: PAddr) -> bool {
        addr >= SRAM_BASE && addr < SRAM_BASE + SRAM_SIZE as PAddr
    }

    pub fn read(&mut self, addr: PAddr, len: usize) -> Word {
        self.ensure_lazy_resident(addr, len, false);
        let ret = if let Some(ptr) = self.guest_to_host(addr) {
            unsafe {
                match len {
                    1 => *ptr as Word,
                    2 => {
                        let p = ptr as *const u16;
                        (*p) as Word
                    }
                    4 => {
                        let p = ptr as *const u32;
                        *p as Word
                    }
                    8 => {
                        let p = ptr as *const u64;
                        *p as Word
                    }
                    _ => {
                        log::error!("Invalid read length: {}", len);
                        0
                    }
                }
            }
        } else {
            // Check if it's MMIO
            if crate::generated::config::DEVICE {
                return crate::memory::mmio::mmio_read(addr, len);
            }

            log::error!("Address 0x{:08x} is out of bound", addr);
            0
        };

        crate::utils::mtrace::trace_read(addr, len, ret);
        crate::utils::sandbox::record_paddr(addr, len, false);
        ret
    }

    pub fn write(&mut self, addr: PAddr, len: usize, data: Word) {
        crate::utils::mtrace::trace_write(addr, len, data);
        crate::utils::sandbox::record_paddr(addr, len, true);
        self.ensure_lazy_resident(addr, len, true);

        if let Some(ptr) = self.guest_to_host(addr) {
            unsafe {
                match len {
                    1 => *ptr = data as u8,
                    2 => {
                        let p = ptr as *mut u16;
                        *p = data as u16;
                    }
                    4 => {
                        let p = ptr as *mut u32;
                        *p = data as u32;
                    }
                    8 => {
                        let p = ptr as *mut u64;
                        *p = data as u64;
                    }
                    _ => {
                        log::error!("Invalid write length: {}", len);
                    }
                }
            }
        } else {
            // Check if it's MMIO
            if crate::generated::config::DEVICE {
                crate::memory::mmio::mmio_write(addr, len, data);
                return;
            }
            log::error!("Address 0x{:08x} is out of bound", addr);
        }
    }

    pub fn snapshot(&mut self) -> MemorySnapshot {
        self.materialize_all_lazy_pages();
        MemorySnapshot {
            pmem: self.pmem.clone(),
            mrom: self.mrom.clone(),
            sram: self.sram.clone(),
            mbase: self.mbase,
            msize: self.msize,
        }
    }

    pub fn restore_eager(&mut self, snapshot: MemorySnapshot) -> Result<(), String> {
        if snapshot.mbase != self.mbase || snapshot.msize != self.msize {
            return Err(format!(
                "memory shape mismatch: snapshot [0x{:x}, {}] current [0x{:x}, {}]",
                snapshot.mbase, snapshot.msize, self.mbase, self.msize
            ));
        }
        self.pmem = snapshot.pmem;
        self.mrom = snapshot.mrom;
        self.sram = snapshot.sram;
        self.lazy = None;
        Ok(())
    }

    pub fn restore_lazy(
        &mut self,
        snapshot: MemorySnapshot,
        prefetch_pages: &[usize],
    ) -> Result<(), String> {
        if snapshot.mbase != self.mbase || snapshot.msize != self.msize {
            return Err(format!(
                "memory shape mismatch: snapshot [0x{:x}, {}] current [0x{:x}, {}]",
                snapshot.mbase, snapshot.msize, self.mbase, self.msize
            ));
        }

        let pages = split_pages(&snapshot.pmem, PMEM_PAGE_SIZE);
        let page_count = pages.len();
        self.pmem.fill(0);
        self.mrom = snapshot.mrom;
        self.sram = snapshot.sram;
        self.lazy = Some(LazyBacking {
            page_size: PMEM_PAGE_SIZE,
            pages,
            resident: vec![false; page_count],
            prefetched: vec![false; page_count],
        });

        let unique: HashSet<usize> = prefetch_pages.iter().copied().collect();
        for idx in unique {
            self.load_lazy_page(idx, false, true);
        }
        Ok(())
    }

    fn ensure_lazy_resident(&mut self, addr: PAddr, len: usize, is_write: bool) {
        if !self.in_pmem(addr) {
            return;
        }
        let start = (addr - self.mbase) as usize;
        let end = start.saturating_add(len.saturating_sub(1));
        let page_size = self
            .lazy
            .as_ref()
            .map(|lazy| lazy.page_size)
            .unwrap_or(PMEM_PAGE_SIZE);
        let first = start / page_size;
        let last = end / page_size;
        for idx in first..=last {
            self.load_lazy_page(idx, is_write, false);
        }
    }

    fn load_lazy_page(&mut self, idx: usize, is_write: bool, prefetch: bool) {
        let Some(lazy) = self.lazy.as_mut() else {
            return;
        };
        if idx >= lazy.pages.len() || lazy.resident[idx] {
            return;
        }

        let start_time = Instant::now();
        let page_start = idx * lazy.page_size;
        let page_end = (page_start + lazy.pages[idx].len()).min(self.pmem.len());
        if page_start < self.pmem.len() && page_start < page_end {
            self.pmem[page_start..page_end]
                .copy_from_slice(&lazy.pages[idx][..page_end - page_start]);
        }
        lazy.resident[idx] = true;
        if prefetch {
            lazy.prefetched[idx] = true;
        }
        let stall_us = start_time.elapsed().as_micros();
        crate::utils::sandbox::record_lazy_page(
            idx,
            self.mbase + page_start as PAddr,
            if prefetch {
                "prefetch"
            } else if is_write {
                "write"
            } else {
                "read"
            },
            stall_us,
            is_write,
            lazy.prefetched[idx],
        );
    }

    fn materialize_all_lazy_pages(&mut self) {
        let page_count = self.lazy.as_ref().map(|lazy| lazy.pages.len()).unwrap_or(0);
        for idx in 0..page_count {
            self.load_lazy_page(idx, false, true);
        }
    }
}

fn split_pages(data: &[u8], page_size: usize) -> Vec<Vec<u8>> {
    data.chunks(page_size).map(|chunk| chunk.to_vec()).collect()
}

#[allow(static_mut_refs)]
pub static mut PMEM: Option<PhysicalMemory> = None;

pub fn init() {
    unsafe {
        PMEM = Some(PhysicalMemory::new(MBASE as PAddr, MSIZE as usize));
    }
}

pub fn paddr_read(addr: PAddr, len: usize) -> Word {
    unsafe {
        match &mut PMEM {
            Some(pmem) => pmem.read(addr, len),
            None => {
                panic!("Physical memory not initialized");
            }
        }
    }
}

pub fn paddr_write(addr: PAddr, len: usize, data: Word) {
    unsafe {
        match &mut PMEM {
            Some(pmem) => pmem.write(addr, len, data),
            None => {
                panic!("Physical memory not initialized");
            }
        }
    }
}

// Load image into memory
pub fn load_image(data: &[u8], addr: PAddr) -> Result<(), String> {
    unsafe {
        if let Some(pmem) = &mut PMEM {
            if let Some(ptr) = pmem.guest_to_host(addr) {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                Ok(())
            } else {
                Err(format!(
                    "Cannot load image at invalid address 0x{:08x}",
                    addr
                ))
            }
        } else {
            Err("Physical memory not initialized".to_string())
        }
    }
}

pub fn snapshot_memory() -> Result<MemorySnapshot, String> {
    unsafe {
        PMEM.as_mut()
            .map(|pmem| pmem.snapshot())
            .ok_or_else(|| "Physical memory not initialized".to_string())
    }
}

pub fn restore_memory_eager(snapshot: MemorySnapshot) -> Result<(), String> {
    unsafe {
        PMEM.as_mut()
            .ok_or_else(|| "Physical memory not initialized".to_string())?
            .restore_eager(snapshot)
    }
}

pub fn restore_memory_lazy(
    snapshot: MemorySnapshot,
    prefetch_pages: &[usize],
) -> Result<(), String> {
    unsafe {
        PMEM.as_mut()
            .ok_or_else(|| "Physical memory not initialized".to_string())?
            .restore_lazy(snapshot, prefetch_pages)
    }
}
