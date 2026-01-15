// Memory module

pub mod pmem;
pub mod mmio;

pub use pmem::*;

pub fn init_mem() {
    crate::Log!("Initializing memory...");
    // Print physical memory info
    let pmem = pmem::PMEM.lock().unwrap();
    crate::Log!("physical memory area [0x{:08x}, 0x{:08x}]",
        pmem.mbase, pmem.mbase.wrapping_add(pmem.msize as u32) - 1);
}
