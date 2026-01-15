// PLIC - Platform-Level Interrupt Controller

use crate::common::Word;

pub fn init() {
    log::info!("PLIC initialized");
}

pub fn plic_read(_addr: u32, _len: usize) -> Word {
    // TODO: Implement PLIC registers
    0
}

pub fn plic_write(_addr: u32, _len: usize, _data: Word) {
    // TODO: Implement PLIC registers
}
