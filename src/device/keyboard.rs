// Keyboard Device (i8042)

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};
use std::sync::Mutex;
use std::collections::VecDeque;

lazy_static::lazy_static! {
    static ref KEY_QUEUE: Mutex<VecDeque<u32>> = Mutex::new(VecDeque::new());
}

pub fn init_keyboard() {
    if !HAS_KEYBOARD { return; }
    
    register_mmio("i8042", I8042_DATA_MMIO, 4, Box::new(i8042_callback));
}

fn i8042_callback(_addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    if is_write {
        0
    } else {
        let mut queue = KEY_QUEUE.lock().unwrap();
        if let Some(am_scancode) = queue.pop_front() {
            am_scancode
        } else {
            0
        }
    }
}

pub fn send_key(scancode: u32, is_keydown: bool) {
    if !HAS_KEYBOARD { return; }
    let am_scancode = if is_keydown { scancode } else { scancode | 0x8000 };
    let mut queue = KEY_QUEUE.lock().unwrap();
    queue.push_back(am_scancode);
}
