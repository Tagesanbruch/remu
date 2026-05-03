// Core Local Interruptor (CLINT)

use crate::common::{PAddr, Word};
use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};

// Register offsets
const CLINT_BASE: PAddr = 0x02000000;
const CLINT_MSIP: PAddr = 0x0000;
const CLINT_MTIMECMP: PAddr = 0x4000;
const CLINT_MTIME: PAddr = 0xbff8;

struct ClintState {
    mtimecmp: u64,
    msip: u32,
}

lazy_static! {
    static ref CLINT: Arc<Mutex<ClintState>> = Arc::new(Mutex::new(ClintState {
        mtimecmp: 0,
        msip: 0,
    }));
}

pub fn init_clint() {
    if !HAS_CLINT {
        return;
    }

    // 0x02000000 - 0x0200ffff (64KB)
    register_mmio("clint", CLINT_BASE, 0x10000, Box::new(clint_callback));
}

fn clint_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - CLINT_BASE;
    let mut state = CLINT.lock().unwrap();

    if is_write {
        match offset {
            CLINT_MSIP => {
                state.msip = data as u32;
            }
            _ if offset == CLINT_MTIMECMP => {
                // Lower 32 bits
                state.mtimecmp = (state.mtimecmp & 0xFFFFFFFF00000000) | (data as u64);
            }
            _ if offset == CLINT_MTIMECMP + 4 => {
                // Upper 32 bits
                state.mtimecmp = (state.mtimecmp & 0x00000000FFFFFFFF) | ((data as u64) << 32);
            }
            _ => {}
        }
        0
    } else {
        match offset {
            CLINT_MSIP => state.msip as Word,
            CLINT_MTIME => crate::device::timer::get_time_u32(0) as Word,
            0xbffc => {
                // CLINT_MTIME + 4
                crate::device::timer::get_time_u32(1) as Word
            }
            _ if offset == CLINT_MTIMECMP => (state.mtimecmp & 0xFFFFFFFF) as Word,
            _ if offset == CLINT_MTIMECMP + 4 => (state.mtimecmp >> 32) as Word,
            _ => 0,
        }
    }
}

// Public API for timer update to call periodically?
pub fn clint_check_intr() {
    let state = CLINT.lock().unwrap();
    let _ = state.msip; // Dummy read
                        // check_timer_intr(&state); // Internal check only modifies state
}

pub fn get_mip_status() -> Word {
    let state = CLINT.lock().unwrap();
    let now = crate::device::timer::get_time_u64();
    let mtip = if now >= state.mtimecmp { 1 << 7 } else { 0 };
    let msip = if (state.msip & 1) != 0 { 1 << 3 } else { 0 };
    mtip | msip
}
