// PLIC Device

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};

pub fn init_plic() {
    if !HAS_PLIC { return; }
    
    // 0x0c000000 - 0x0c200000+ (4MB range usually)
    register_mmio("plic", 0x0c000000, 0x400000, Box::new(plic_callback));
}

fn plic_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - 0x0c000000;
    
    if is_write {
        // Complete Register: 0x201004 (Context 1)
        if offset == 0x201004 {
             // In NEMU: if (completion == 10) { ... } -> handled by serial update typically
             // For now, just ack.
             // crate::Log!("PLIC: Complete IRQ {}", data);
             crate::utils::dtrace::trace_dtrace(addr, _len, data, is_write, "plic");
        } else {
             // crate::Log!("PLIC: Write offset 0x{:x} data 0x{:x}", offset, data);
             crate::utils::dtrace::trace_dtrace(addr, _len, data, is_write, "plic");
        }
        0
    } else {
        // Read
        // Context 1 (S-mode) Claim Register: 0x201004
        if offset == 0x201004 {
            let state = crate::device::intr::get_intr_state();
            if (state & (1 << 9)) != 0 {
                 // crate::Log!("PLIC: Claim IRQ 10 (UART)");
                 crate::utils::dtrace::trace_dtrace(addr, _len, 10, is_write, "plic");
                 return 10;
            }
        }
        
        // crate::Log!("PLIC: Read offset 0x{:x}", offset);
        crate::utils::dtrace::trace_dtrace(addr, _len, 0, is_write, "plic");
        0
    }
}
