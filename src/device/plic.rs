// PLIC Device

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};

pub fn init_plic() {
    if !HAS_PLIC { return; }
    
    // 0x0c000000 - 0x0c200000+ (4MB range usually)
    register_mmio("plic", 0x0c000000, 0x400000, Box::new(plic_callback));
}

fn plic_callback(addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    let _offset = addr - 0x0c000000;
    
    if is_write {
        // Handle priority, pending, enable writes
        0
    } else {
        // Read
        0
    }
}
