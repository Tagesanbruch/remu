// Core Local Interruptor (CLINT)

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};

// Register offsets
const CLINT_MSIP: u32 = 0x0000;
const CLINT_MTIME: u32 = 0xbff8;

pub fn init_clint() {
    if !HAS_CLINT { return; }
    
    // 0x02000000 - 0x0200ffff (64KB)
    register_mmio("clint", 0x02000000, 0x10000, Box::new(clint_callback));
}

fn clint_callback(addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    let offset = addr - 0x02000000;
    
    if is_write {
        match offset {
             CLINT_MSIP => { 
                 // Write MSIP
             }
             _ => {}
        }
        0
    } else {
        match offset {
            CLINT_MTIME => {
                crate::device::timer::get_time_u32(0)
            }
            // CLINT_MTIME + 4 (0xbffc)
            0xbffc => {
                crate::device::timer::get_time_u32(1)
            }
            _ => 0
        }
    }
}
