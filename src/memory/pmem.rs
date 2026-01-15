// Physical memory implementation

use crate::common::{Word, PAddr};
use crate::generated::config::*;
use std::sync::{Arc, Mutex};

// Memory regions
const MROM_BASE: u32 = 0x20000000;
const MROM_SIZE: usize = 0x1000;  // 4KB

const SRAM_BASE: u32 = 0x0f000000;
const SRAM_SIZE: usize = 0x2000;  // 8KB

pub struct PhysicalMemory {
    pub pmem: Vec<u8>,
    pub mrom: Vec<u8>,
    pub sram: Vec<u8>,
    pub mbase: u32,
    pub msize: usize,
}

impl PhysicalMemory {
    pub fn new(mbase: u32, msize: usize) -> Self {
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
        addr >= self.mbase && addr < self.mbase + self.msize as u32
    }

    #[inline]
    fn in_mrom(&self, addr: PAddr) -> bool {
        addr >= MROM_BASE && addr < MROM_BASE + MROM_SIZE as u32
    }

    #[inline]
    fn in_sram(&self, addr: PAddr) -> bool {
        addr >= SRAM_BASE && addr < SRAM_BASE + SRAM_SIZE as u32
    }

    pub fn read(&self, addr: PAddr, len: usize) -> Word {
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
                        *p
                    }
                    _ => {
                        log::error!("Invalid read length: {}", len);
                        0
                    }
                }
            }
        } else {
            // Check if it's MMIO
            // For now, simple logic, device module needs to be properly hooked up
            // if DEVICE { ... }
            log::error!("Address 0x{:08x} is out of bound", addr);
            0
        };
        
        crate::utils::mtrace::trace_read(addr, len, ret);
        ret
    }

    pub fn write(&mut self, addr: PAddr, len: usize, data: Word) {
        crate::utils::mtrace::trace_write(addr, len, data);
        
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
                        *p = data;
                    }
                    _ => {
                        log::error!("Invalid write length: {}", len);
                    }
                }
            }
        } else {
            // Check if it's MMIO
            log::error!("Address 0x{:08x} is out of bound", addr);
        }
    }
}

lazy_static::lazy_static! {
    pub static ref PMEM: Arc<Mutex<PhysicalMemory>> = {
        Arc::new(Mutex::new(PhysicalMemory::new(MBASE, MSIZE as usize)))
    };
}

pub fn paddr_read(addr: PAddr, len: usize) -> Word {
    PMEM.lock().unwrap().read(addr, len)
}

pub fn paddr_write(addr: PAddr, len: usize, data: Word) {
    PMEM.lock().unwrap().write(addr, len, data);
}

// Load image into memory
pub fn load_image(data: &[u8], addr: PAddr) -> Result<(), String> {
    let pmem = PMEM.lock().unwrap();
    if let Some(ptr) = pmem.guest_to_host(addr) {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    } else {
        Err(format!("Cannot load image at invalid address 0x{:08x}", addr))
    }
}
