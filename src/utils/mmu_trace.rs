// MMU Trace

use crate::generated::config::*;
use crate::common::{PAddr, VAddr};

#[derive(Clone)]
pub struct MmuTraceEntry {
    pub vaddr: VAddr,
    pub paddr: PAddr,
    pub type_: i32,
    pub success: bool,
}

impl ToString for MmuTraceEntry {
    fn to_string(&self) -> String {
        if self.success {
            // NEMU: MMU Success: vaddr=0xc034953c -> paddr=0x8074953c type=0
            format!("MMU Success: vaddr=0x{:08x} -> paddr=0x{:08x} type={}", 
                self.vaddr, self.paddr, self.type_)
        } else {
            format!("MMU Fail: vaddr=0x{:08x} type={}", self.vaddr, self.type_)
        }
    }
}

lazy_static::lazy_static! {
    static ref MMU_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<MmuTraceEntry>> = {
        let size = if crate::generated::config::TRACE_MMU { 
            crate::generated::config::TRACE_MMU_RINGBUF as usize 
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_mmu(vaddr: VAddr, paddr: PAddr, type_: i32, success: bool) {
    if !TRACE_MMU { return; }
    
    let entry = MmuTraceEntry {
        vaddr,
        paddr,
        type_,
        success,
    };
    
    MMU_BUF.lock().unwrap().push(entry);
}

pub fn show_mmu_trace() {
    if !TRACE_MMU { return; }
    
    crate::Log!("--- RingBuffer Content ---");
    let buf = MMU_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("--------------------------");
}
