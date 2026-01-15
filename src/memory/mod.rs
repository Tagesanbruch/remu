use crate::generated::config::*;

pub mod mmio;
pub mod pmem;

pub use pmem::{paddr_read, paddr_write, load_image};

pub fn init_mem() {
    // Initialize MMIO
    mmio::init_mmio();
    
    // Verify configs
    crate::Log!("physical memory area [0x{:08x}, 0x{:08x}]", 
             MBASE, MBASE + MSIZE);
}
