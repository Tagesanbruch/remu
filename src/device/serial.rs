// Serial device (UART)

use crate::common::Word;
use std::io::{self, Write};
use std::sync::Mutex;

pub const SERIAL_MMIO: u32 = 0xa00003f8;

lazy_static::lazy_static! {
    static ref SERIAL: Mutex<SerialDevice> = Mutex::new(SerialDevice::new());
}

pub struct SerialDevice {
    // Simple UART model
}

impl SerialDevice {
    pub fn new() -> Self {
        Self {}
    }

    pub fn read(&self, _offset: u32) -> Word {
        // Read not implemented for now
        0
    }

    pub fn write(&mut self, _offset: u32, data: Word) {
        // Print character to stdout
        let ch = (data & 0xff) as u8;
        print!("{}", ch as char);
        io::stdout().flush().ok();
    }
}

pub fn init() {
    log::info!("Serial device initialized at 0x{:08x}", SERIAL_MMIO);
}

pub fn serial_read(offset: u32) -> Word {
    SERIAL.lock().unwrap().read(offset)
}

pub fn serial_write(offset: u32, data: Word) {
   SERIAL.lock().unwrap().write(offset, data)
}
