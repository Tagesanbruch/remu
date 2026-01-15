// Timer device (RTC)

use crate::common::Word;
use std::time::{SystemTime, UNIX_EPOCH};

pub const RTC_MMIO: u32 = 0xa0000048;

pub fn init() {
    log::info!("Timer device initialized at 0x{:08x}", RTC_MMIO);
}

pub fn timer_read() -> Word {
    // Return current time in microseconds
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap();
    let micros = duration.as_micros() as u64;
    (micros & 0xffffffff) as Word  // Return low 32 bits for RV32
}

pub fn update() {
    // Timer updates happen on read
}
