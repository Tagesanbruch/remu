// Timer Device (RTC) and Helper

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};
use std::time::Instant;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref BOOT_TIME: Mutex<Option<Instant>> = Mutex::new(None);
}

pub fn init_timer() {
    let mut boot_time = BOOT_TIME.lock().unwrap();
    if boot_time.is_none() {
        *boot_time = Some(Instant::now());
    }
    
    if !HAS_TIMER { return; }
    
    register_mmio("rtc", RTC_MMIO, 8, Box::new(rtc_callback));
}

fn rtc_callback(addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    if is_write {
        // RTC is read-only
        0
    } else {
        let offset = addr - RTC_MMIO;
        if offset == 0 || offset == 4 {
            get_time_u32(if offset == 0 { 0 } else { 1 })
        } else {
            0
        }
    }
}

pub fn get_time_u64() -> u64 {
    let boot_time = BOOT_TIME.lock().unwrap();
    if let Some(boot) = *boot_time {
        boot.elapsed().as_micros() as u64
    } else {
        0
    }
}

pub fn get_time_u32(idx: usize) -> u32 {
    let us = get_time_u64();
    if idx == 0 {
        us as u32
    } else {
        (us >> 32) as u32
    }
}
