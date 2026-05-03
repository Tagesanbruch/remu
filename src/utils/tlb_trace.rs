// TLB Trace

use crate::common::{PAddr, VAddr};
use crate::generated::config::*;

#[derive(Clone)]
pub struct TlbTraceEntry {
    pub vaddr: VAddr,
    pub paddr: PAddr,
    pub kind: i32,
    pub hit: bool,
    pub flush: bool,
}

impl ToString for TlbTraceEntry {
    fn to_string(&self) -> String {
        if self.flush {
            "TLB Flush".to_string()
        } else if self.hit {
            format!(
                "TLB Hit: vaddr=0x{:016x} -> paddr=0x{:016x} type={}",
                self.vaddr, self.paddr, self.kind
            )
        } else {
            format!("TLB Miss: vaddr=0x{:016x} type={}", self.vaddr, self.kind)
        }
    }
}

lazy_static::lazy_static! {
    static ref TLB_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<TlbTraceEntry>> = {
        let size = if crate::generated::config::TRACE_TLB {
            crate::generated::config::TRACE_TLB_RINGBUF as usize
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_tlb(vaddr: VAddr, paddr: PAddr, kind: i32, hit: bool) {
    if !TRACE_TLB {
        return;
    }

    TLB_BUF.lock().unwrap().push(TlbTraceEntry {
        vaddr,
        paddr,
        kind,
        hit,
        flush: false,
    });
}

pub fn trace_tlb_flush() {
    if !TRACE_TLB {
        return;
    }

    TLB_BUF.lock().unwrap().push(TlbTraceEntry {
        vaddr: 0,
        paddr: 0,
        kind: -1,
        hit: false,
        flush: true,
    });
}

pub fn show_tlb_trace() {
    if !TRACE_TLB {
        return;
    }

    let (hits, misses, walks) = crate::isa::riscv32::system::mmu::tlb_stats();
    crate::Log!("--- TLB Trace Content ---");
    crate::Log!(
        "TLB Stats: hits={} misses={} page_walks={}",
        hits,
        misses,
        walks
    );
    let buf = TLB_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("-------------------------");
}
