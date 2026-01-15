// Serial Device (UART)

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};

pub fn init_serial() {
    if !HAS_SERIAL { return; }
    
    register_mmio("serial", SERIAL_MMIO, 8, Box::new(serial_callback));
}

pub fn serial_update() {
    // Flush stdout if needed
    // std::io::stdout().flush().unwrap();
}

fn serial_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    if is_write {
        let offset = addr - SERIAL_MMIO;
        if offset == 0 {
            print!("{}", (data as u8) as char);
        }
        0
    } else {
        0
    }
}
