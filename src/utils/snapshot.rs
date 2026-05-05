use crate::common::{PrivMode, Word};
use crate::memory::paddr::{MemorySnapshot, PMEM_PAGE_SIZE};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"REMUSNP1";
const VERSION: u32 = 1;

pub enum PrefetchPolicy {
    None,
    Sequential,
    Profile(PathBuf),
}

pub struct LoadOptions {
    pub lazy: bool,
    pub prefetch: PrefetchPolicy,
    pub pages: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            lazy: false,
            prefetch: PrefetchPolicy::None,
            pages: 256,
        }
    }
}

struct CpuImage {
    pc: Word,
    gpr: [Word; 32],
    fpr: [u64; 32],
    csr: Vec<Word>,
    mode: PrivMode,
    is_exception: bool,
    exception_entry: Word,
}

struct DeviceImage {
    clint: crate::device::clint::ClintSnapshot,
    plic: crate::device::plic::PlicSnapshot,
    serial: crate::device::serial::SerialSnapshot,
    intr: u32,
    timer_elapsed_us: u64,
}

pub fn save_vm(path: &Path) -> Result<(), String> {
    let cpu = snapshot_cpu();
    let memory = crate::memory::paddr::snapshot_memory()?;
    let devices = snapshot_devices();

    let mut file =
        File::create(path).map_err(|err| format!("create {}: {}", path.display(), err))?;
    file.write_all(MAGIC).map_err(|err| err.to_string())?;
    file.write_u32::<LittleEndian>(VERSION)
        .map_err(|err| err.to_string())?;
    file.write_u32::<LittleEndian>(crate::common::xlen())
        .map_err(|err| err.to_string())?;
    file.write_u64::<LittleEndian>(memory.mbase)
        .map_err(|err| err.to_string())?;
    file.write_u64::<LittleEndian>(memory.msize as u64)
        .map_err(|err| err.to_string())?;

    write_cpu(&mut file, &cpu)?;
    write_devices(&mut file, &devices)?;
    write_blob(&mut file, &memory.mrom)?;
    write_blob(&mut file, &memory.sram)?;
    write_pmem_pages(&mut file, &memory.pmem)?;
    crate::Log!(
        "REMU-Sandbox snapshot saved: {} (pmem={} bytes)",
        path.display(),
        memory.pmem.len()
    );
    Ok(())
}

pub fn load_vm(path: &Path, options: &LoadOptions) -> Result<(), String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {}", path.display(), err))?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic).map_err(|err| err.to_string())?;
    if &magic != MAGIC {
        return Err(format!("{} is not a REMUSNP1 snapshot", path.display()));
    }

    let version = file
        .read_u32::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    if version != VERSION {
        return Err(format!("unsupported snapshot version {}", version));
    }

    let xlen = file
        .read_u32::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    if xlen != crate::common::xlen() {
        return Err(format!(
            "snapshot xlen {} does not match current xlen {}",
            xlen,
            crate::common::xlen()
        ));
    }

    let mbase = file
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let msize = file
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())? as usize;
    let cpu = read_cpu(&mut file)?;
    let devices = read_devices(&mut file)?;
    let mrom = read_blob(&mut file)?;
    let sram = read_blob(&mut file)?;
    let pmem = read_pmem_pages(&mut file, msize)?;
    let memory = MemorySnapshot {
        pmem,
        mrom,
        sram,
        mbase,
        msize,
    };

    if options.lazy {
        let prefetch_pages = prefetch_pages(&options.prefetch, options.pages, memory.pmem.len());
        crate::memory::paddr::restore_memory_lazy(memory, &prefetch_pages)?;
        crate::Log!(
            "REMU-Sandbox lazy snapshot loaded: {} prefetch_pages={}",
            path.display(),
            prefetch_pages.len()
        );
    } else {
        crate::memory::paddr::restore_memory_eager(memory)?;
        crate::Log!("REMU-Sandbox eager snapshot loaded: {}", path.display());
    }

    restore_devices(devices);
    restore_cpu(cpu);
    crate::isa::riscv32::system::mmu::flush_tlb();
    Ok(())
}

fn snapshot_cpu() -> CpuImage {
    let cpu = crate::cpu::state::CPU.lock().unwrap();
    CpuImage {
        pc: cpu.pc,
        gpr: cpu.gpr,
        fpr: cpu.fpr,
        csr: cpu.csr.to_vec(),
        mode: cpu.mode,
        is_exception: cpu.is_exception,
        exception_entry: cpu.exception_entry,
    }
}

fn restore_cpu(image: CpuImage) {
    let mut cpu = crate::cpu::state::CPU.lock().unwrap();
    cpu.pc = image.pc;
    cpu.gpr = image.gpr;
    cpu.fpr = image.fpr;
    for (idx, value) in image.csr.into_iter().enumerate().take(cpu.csr.len()) {
        cpu.csr[idx] = value;
    }
    cpu.mode = image.mode;
    cpu.is_exception = image.is_exception;
    cpu.exception_entry = image.exception_entry;
}

fn snapshot_devices() -> DeviceImage {
    DeviceImage {
        clint: crate::device::clint::snapshot_state(),
        plic: crate::device::plic::snapshot_state(),
        serial: crate::device::serial::snapshot_state(),
        intr: crate::device::intr::snapshot_state(),
        timer_elapsed_us: crate::device::timer::snapshot_elapsed_us(),
    }
}

fn restore_devices(image: DeviceImage) {
    crate::device::timer::restore_elapsed_us(image.timer_elapsed_us);
    crate::device::intr::restore_state(image.intr);
    crate::device::clint::restore_state(image.clint);
    crate::device::plic::restore_state(image.plic);
    crate::device::serial::restore_state(image.serial);
}

fn write_cpu<W: Write>(w: &mut W, cpu: &CpuImage) -> Result<(), String> {
    w.write_u64::<LittleEndian>(cpu.pc)
        .map_err(|err| err.to_string())?;
    w.write_u8(mode_to_u8(cpu.mode))
        .map_err(|err| err.to_string())?;
    w.write_u8(cpu.is_exception as u8)
        .map_err(|err| err.to_string())?;
    w.write_u64::<LittleEndian>(cpu.exception_entry)
        .map_err(|err| err.to_string())?;
    for value in cpu.gpr {
        w.write_u64::<LittleEndian>(value)
            .map_err(|err| err.to_string())?;
    }
    for value in cpu.fpr {
        w.write_u64::<LittleEndian>(value)
            .map_err(|err| err.to_string())?;
    }
    w.write_u32::<LittleEndian>(cpu.csr.len() as u32)
        .map_err(|err| err.to_string())?;
    for value in &cpu.csr {
        w.write_u64::<LittleEndian>(*value)
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn read_cpu<R: Read>(r: &mut R) -> Result<CpuImage, String> {
    let pc = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let mode = u8_to_mode(r.read_u8().map_err(|err| err.to_string())?)?;
    let is_exception = r.read_u8().map_err(|err| err.to_string())? != 0;
    let exception_entry = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let mut gpr = [0_u64; 32];
    for value in &mut gpr {
        *value = r
            .read_u64::<LittleEndian>()
            .map_err(|err| err.to_string())?;
    }
    let mut fpr = [0_u64; 32];
    for value in &mut fpr {
        *value = r
            .read_u64::<LittleEndian>()
            .map_err(|err| err.to_string())?;
    }
    let csr_len = r
        .read_u32::<LittleEndian>()
        .map_err(|err| err.to_string())? as usize;
    let mut csr = Vec::with_capacity(csr_len);
    for _ in 0..csr_len {
        csr.push(
            r.read_u64::<LittleEndian>()
                .map_err(|err| err.to_string())?,
        );
    }
    Ok(CpuImage {
        pc,
        gpr,
        fpr,
        csr,
        mode,
        is_exception,
        exception_entry,
    })
}

fn write_devices<W: Write>(w: &mut W, dev: &DeviceImage) -> Result<(), String> {
    w.write_u64::<LittleEndian>(dev.clint.mtimecmp)
        .map_err(|err| err.to_string())?;
    w.write_u32::<LittleEndian>(dev.clint.msip)
        .map_err(|err| err.to_string())?;

    for value in dev.plic.priority {
        w.write_u32::<LittleEndian>(value)
            .map_err(|err| err.to_string())?;
    }
    w.write_u64::<LittleEndian>(dev.plic.pending)
        .map_err(|err| err.to_string())?;
    w.write_u64::<LittleEndian>(dev.plic.line_level)
        .map_err(|err| err.to_string())?;
    w.write_u64::<LittleEndian>(dev.plic.claimed)
        .map_err(|err| err.to_string())?;
    for value in dev.plic.enable {
        w.write_u64::<LittleEndian>(value)
            .map_err(|err| err.to_string())?;
    }
    for value in dev.plic.threshold {
        w.write_u32::<LittleEndian>(value)
            .map_err(|err| err.to_string())?;
    }

    write_blob(w, &dev.serial.rx_fifo)?;
    for value in [
        dev.serial.ier,
        dev.serial.fcr,
        dev.serial.lcr,
        dev.serial.mcr,
        dev.serial.dll,
        dev.serial.dlm,
        dev.serial.scr,
        dev.serial.thr_empty as u8,
        dev.serial.thre_pending as u8,
    ] {
        w.write_u8(value).map_err(|err| err.to_string())?;
    }

    w.write_u32::<LittleEndian>(dev.intr)
        .map_err(|err| err.to_string())?;
    w.write_u64::<LittleEndian>(dev.timer_elapsed_us)
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn read_devices<R: Read>(r: &mut R) -> Result<DeviceImage, String> {
    let clint = crate::device::clint::ClintSnapshot {
        mtimecmp: r
            .read_u64::<LittleEndian>()
            .map_err(|err| err.to_string())?,
        msip: r
            .read_u32::<LittleEndian>()
            .map_err(|err| err.to_string())?,
    };

    let mut priority = [0_u32; crate::device::plic::SNAPSHOT_MAX_SOURCE];
    for value in &mut priority {
        *value = r
            .read_u32::<LittleEndian>()
            .map_err(|err| err.to_string())?;
    }
    let pending = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let line_level = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let claimed = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())?;
    let mut enable = [0_u64; crate::device::plic::SNAPSHOT_CONTEXTS];
    for value in &mut enable {
        *value = r
            .read_u64::<LittleEndian>()
            .map_err(|err| err.to_string())?;
    }
    let mut threshold = [0_u32; crate::device::plic::SNAPSHOT_CONTEXTS];
    for value in &mut threshold {
        *value = r
            .read_u32::<LittleEndian>()
            .map_err(|err| err.to_string())?;
    }
    let plic = crate::device::plic::PlicSnapshot {
        priority,
        pending,
        line_level,
        claimed,
        enable,
        threshold,
    };

    let rx_fifo = read_blob(r)?;
    let serial = crate::device::serial::SerialSnapshot {
        rx_fifo,
        ier: r.read_u8().map_err(|err| err.to_string())?,
        fcr: r.read_u8().map_err(|err| err.to_string())?,
        lcr: r.read_u8().map_err(|err| err.to_string())?,
        mcr: r.read_u8().map_err(|err| err.to_string())?,
        dll: r.read_u8().map_err(|err| err.to_string())?,
        dlm: r.read_u8().map_err(|err| err.to_string())?,
        scr: r.read_u8().map_err(|err| err.to_string())?,
        thr_empty: r.read_u8().map_err(|err| err.to_string())? != 0,
        thre_pending: r.read_u8().map_err(|err| err.to_string())? != 0,
    };

    Ok(DeviceImage {
        clint,
        plic,
        serial,
        intr: r
            .read_u32::<LittleEndian>()
            .map_err(|err| err.to_string())?,
        timer_elapsed_us: r
            .read_u64::<LittleEndian>()
            .map_err(|err| err.to_string())?,
    })
}

fn write_blob<W: Write>(w: &mut W, data: &[u8]) -> Result<(), String> {
    w.write_u64::<LittleEndian>(data.len() as u64)
        .map_err(|err| err.to_string())?;
    w.write_all(data).map_err(|err| err.to_string())
}

fn read_blob<R: Read>(r: &mut R) -> Result<Vec<u8>, String> {
    let len = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())? as usize;
    let mut data = vec![0_u8; len];
    r.read_exact(&mut data).map_err(|err| err.to_string())?;
    Ok(data)
}

fn write_pmem_pages<W: Write>(w: &mut W, pmem: &[u8]) -> Result<(), String> {
    let page_count = pmem.chunks(PMEM_PAGE_SIZE).count();
    w.write_u64::<LittleEndian>(PMEM_PAGE_SIZE as u64)
        .map_err(|err| err.to_string())?;
    w.write_u64::<LittleEndian>(page_count as u64)
        .map_err(|err| err.to_string())?;
    for page in pmem.chunks(PMEM_PAGE_SIZE) {
        w.write_u32::<LittleEndian>(page.len() as u32)
            .map_err(|err| err.to_string())?;
        w.write_all(page).map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn read_pmem_pages<R: Read>(r: &mut R, expected_size: usize) -> Result<Vec<u8>, String> {
    let page_size = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())? as usize;
    if page_size != PMEM_PAGE_SIZE {
        return Err(format!("unsupported snapshot page size {}", page_size));
    }
    let page_count = r
        .read_u64::<LittleEndian>()
        .map_err(|err| err.to_string())? as usize;
    let mut pmem = Vec::with_capacity(page_count * page_size);
    for _ in 0..page_count {
        let len = r
            .read_u32::<LittleEndian>()
            .map_err(|err| err.to_string())? as usize;
        let mut page = vec![0_u8; len];
        r.read_exact(&mut page).map_err(|err| err.to_string())?;
        pmem.extend_from_slice(&page);
    }
    pmem.resize(expected_size, 0);
    Ok(pmem)
}

fn prefetch_pages(policy: &PrefetchPolicy, limit: usize, pmem_len: usize) -> Vec<usize> {
    let page_count = (pmem_len + PMEM_PAGE_SIZE - 1) / PMEM_PAGE_SIZE;
    match policy {
        PrefetchPolicy::None => Vec::new(),
        PrefetchPolicy::Sequential => (0..page_count.min(limit)).collect(),
        PrefetchPolicy::Profile(path) => read_profile_pages(path, limit, page_count),
    }
}

fn read_profile_pages(path: &Path, limit: usize, page_count: usize) -> Vec<usize> {
    let Ok(content) = std::fs::read_to_string(path) else {
        crate::Log!("failed to read hot page profile {}", path.display());
        return Vec::new();
    };
    let mut pages = Vec::new();
    for line in content.lines().skip(1) {
        let first = line.split(',').next().unwrap_or("").trim();
        if let Ok(idx) = first.parse::<usize>() {
            if idx < page_count {
                pages.push(idx);
            }
        }
        if pages.len() >= limit {
            break;
        }
    }
    pages.sort_unstable();
    pages.dedup();
    pages
}

fn mode_to_u8(mode: PrivMode) -> u8 {
    match mode {
        PrivMode::User => 0,
        PrivMode::Supervisor => 1,
        PrivMode::Machine => 3,
    }
}

fn u8_to_mode(mode: u8) -> Result<PrivMode, String> {
    match mode {
        0 => Ok(PrivMode::User),
        1 => Ok(PrivMode::Supervisor),
        3 => Ok(PrivMode::Machine),
        other => Err(format!("invalid privilege mode {}", other)),
    }
}
