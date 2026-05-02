use crate::generated::config::*;

pub mod mmio;
pub mod paddr;
pub mod vaddr;

pub use paddr::{load_image, paddr_read, paddr_write};
pub use vaddr::{vaddr_ifetch, vaddr_read, vaddr_write};

pub fn init_mem() {
    // Initialize MMIO
    mmio::init_mmio();

    // Initialize Physical Memory
    paddr::init();

    // Verify configs
    crate::Log!(
        "physical memory area [0x{:08x}, 0x{:08x}]",
        MBASE,
        MBASE + MSIZE
    );
}
