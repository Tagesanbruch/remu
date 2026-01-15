// Core Local Interruptor (CLINT)

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

// Register offsets
const CLINT_MSIP: u32 = 0x0000;
const CLINT_MTIMECMP: u32 = 0x4000;
const CLINT_MTIME: u32 = 0xbff8;

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
    if !HAS_CLINT { return; }
    
    // 0x02000000 - 0x0200ffff (64KB)
    register_mmio("clint", 0x02000000, 0x10000, Box::new(clint_callback));
}

fn clint_callback(addr: PAddr, len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - 0x02000000;
    let mut state = CLINT.lock().unwrap();
    
    if is_write {
        match offset {
             CLINT_MSIP => { 
                 state.msip = data;
             }
             _ if offset == CLINT_MTIMECMP => { // Lower 32 bits
                 state.mtimecmp = (state.mtimecmp & 0xFFFFFFFF00000000) | (data as u64);
             }
             _ if offset == CLINT_MTIMECMP + 4 => { // Upper 32 bits
                 state.mtimecmp = (state.mtimecmp & 0x00000000FFFFFFFF) | ((data as u64) << 32);
             }
             _ => {}
        }
        0
    } else {
        match offset {
            CLINT_MSIP => state.msip,
            CLINT_MTIME => {
                crate::device::timer::get_time_u32(0)
            }
            0xbffc => { // CLINT_MTIME + 4
                crate::device::timer::get_time_u32(1)
            }
            _ if offset == CLINT_MTIMECMP => (state.mtimecmp & 0xFFFFFFFF) as u32,
            _ if offset == CLINT_MTIMECMP + 4 => (state.mtimecmp >> 32) as u32,
            _ => 0
        }
    }
}

// Public API for timer update to call periodically?
pub fn clint_check_intr() {
    let mut state = CLINT.lock().unwrap();
    state.msip = state.msip; // Dummy read
    // check_timer_intr(&state); // Internal check only modifies state
}

pub fn get_mip_status() -> Word {
    let state = CLINT.lock().unwrap();
    let now = crate::device::timer::get_time_u64();
    let mtip = if now >= state.mtimecmp { 1 << 7 } else { 0 };
    let msip = if (state.msip & 1) != 0 { 1 << 3 } else { 0 };
    mtip | msip
}
