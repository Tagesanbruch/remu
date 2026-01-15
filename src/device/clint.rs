// CLINT - Core Local Interruptor

use crate::common::Word;

pub fn init() {
    log::info!("CLINT initialized");
}

pub fn clint_read(_addr: u32, _len: usize) -> Word {
    // TODO: Implement CLINT registers
    0
}

pub fn clint_write(_addr: u32, _len: usize, _data: Word) {
    // TODO: Implement CLINT registers
}
