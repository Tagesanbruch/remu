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

// AM Key Codes (matching amdev.h)
pub const AM_KEY_NONE: u32 = 0;
pub const AM_KEY_ESCAPE: u32 = 1;
pub const AM_KEY_F1: u32 = 2;
pub const AM_KEY_F2: u32 = 3;
pub const AM_KEY_F3: u32 = 4;
pub const AM_KEY_F4: u32 = 5;
pub const AM_KEY_F5: u32 = 6;
pub const AM_KEY_F6: u32 = 7;
pub const AM_KEY_F7: u32 = 8;
pub const AM_KEY_F8: u32 = 9;
pub const AM_KEY_F9: u32 = 10;
pub const AM_KEY_F10: u32 = 11;
pub const AM_KEY_F11: u32 = 12;
pub const AM_KEY_F12: u32 = 13;
pub const AM_KEY_GRAVE: u32 = 14;
pub const AM_KEY_1: u32 = 15;
pub const AM_KEY_2: u32 = 16;
pub const AM_KEY_3: u32 = 17;
pub const AM_KEY_4: u32 = 18;
pub const AM_KEY_5: u32 = 19;
pub const AM_KEY_6: u32 = 20;
pub const AM_KEY_7: u32 = 21;
pub const AM_KEY_8: u32 = 22;
pub const AM_KEY_9: u32 = 23;
pub const AM_KEY_0: u32 = 24;
pub const AM_KEY_MINUS: u32 = 25;
pub const AM_KEY_EQUALS: u32 = 26;
pub const AM_KEY_BACKSPACE: u32 = 27;
pub const AM_KEY_TAB: u32 = 28;
pub const AM_KEY_Q: u32 = 29;
pub const AM_KEY_W: u32 = 30;
pub const AM_KEY_E: u32 = 31;
pub const AM_KEY_R: u32 = 32;
pub const AM_KEY_T: u32 = 33;
pub const AM_KEY_Y: u32 = 34;
pub const AM_KEY_U: u32 = 35;
pub const AM_KEY_I: u32 = 36;
pub const AM_KEY_O: u32 = 37;
pub const AM_KEY_P: u32 = 38;
pub const AM_KEY_LEFTBRACKET: u32 = 39;
pub const AM_KEY_RIGHTBRACKET: u32 = 40;
pub const AM_KEY_BACKSLASH: u32 = 41;
pub const AM_KEY_CAPSLOCK: u32 = 42;
pub const AM_KEY_A: u32 = 43;
pub const AM_KEY_S: u32 = 44;
pub const AM_KEY_D: u32 = 45;
pub const AM_KEY_F: u32 = 46;
pub const AM_KEY_G: u32 = 47;
pub const AM_KEY_H: u32 = 48;
pub const AM_KEY_J: u32 = 49;
pub const AM_KEY_K: u32 = 50;
pub const AM_KEY_L: u32 = 51;
pub const AM_KEY_SEMICOLON: u32 = 52;
pub const AM_KEY_APOSTROPHE: u32 = 53;
pub const AM_KEY_RETURN: u32 = 54;
pub const AM_KEY_LSHIFT: u32 = 55;
pub const AM_KEY_Z: u32 = 56;
pub const AM_KEY_X: u32 = 57;
pub const AM_KEY_C: u32 = 58;
pub const AM_KEY_V: u32 = 59;
pub const AM_KEY_B: u32 = 60;
pub const AM_KEY_N: u32 = 61;
pub const AM_KEY_M: u32 = 62;
pub const AM_KEY_COMMA: u32 = 63;
pub const AM_KEY_PERIOD: u32 = 64;
pub const AM_KEY_SLASH: u32 = 65;
pub const AM_KEY_RSHIFT: u32 = 66;
pub const AM_KEY_LCTRL: u32 = 67;
pub const AM_KEY_APPLICATION: u32 = 68;
pub const AM_KEY_LALT: u32 = 69;
pub const AM_KEY_SPACE: u32 = 70;
pub const AM_KEY_RALT: u32 = 71;
pub const AM_KEY_RCTRL: u32 = 72;
pub const AM_KEY_UP: u32 = 73;
pub const AM_KEY_DOWN: u32 = 74;
pub const AM_KEY_LEFT: u32 = 75;
pub const AM_KEY_RIGHT: u32 = 76;
pub const AM_KEY_INSERT: u32 = 77;
pub const AM_KEY_DELETE: u32 = 78;
pub const AM_KEY_HOME: u32 = 79;
pub const AM_KEY_END: u32 = 80;
pub const AM_KEY_PAGEUP: u32 = 81;
pub const AM_KEY_PAGEDOWN: u32 = 82;

pub fn send_key(scancode: u32, is_keydown: bool) {
    if !HAS_KEYBOARD { return; }
    
    // Convert SDL scancode to AM Key Code
    // Note: REMU assumes the `scancode` here IS the AM key code for now,
    // because mapping ~100 keys in Rust manually is tedious. 
    // Ideally SDL scancode -> AM Key map should be here.
    // For now, let's trust caller passes AM_KEY if possible, or we implement mapping.
    
    // Correction: We must implement mapping because SDL scancodes != AM keys.
    // However, to keep it simple and consistent with NEMU's behavior,
    // we will rely on sdl.rs to call us, and we'll implement the map in sdl.rs or here.
    // Let's implement a direct map here if we receive raw SDL scancodes (u32).
    
    // Actually, to make it clean, let's accept AM_* keys directly.
    // The caller (sdl.rs) should map SDL -> AM.
    
    let am_scancode = if is_keydown { scancode | 0x8000 } else { scancode };
    let mut queue = KEY_QUEUE.lock().unwrap();
    queue.push_back(am_scancode);
}
