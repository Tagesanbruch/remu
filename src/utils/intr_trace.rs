// Interrupt Trace

use crate::generated::config::*;
use crate::common::Word;

#[derive(Clone)]
pub struct IntrTraceEntry {
    pub cause: Word,
    pub epc: Word,
    pub is_intr: bool,
}

impl ToString for IntrTraceEntry {
    fn to_string(&self) -> String {
        // NEMU Format: Intr: Cause=3 EPC=0x8001cc94
        format!("Intr: Cause={} EPC=0x{:08x}", 
            self.cause, self.epc)
    }
}

lazy_static::lazy_static! {
    static ref INTR_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<IntrTraceEntry>> = {
        let size = if crate::generated::config::TRACE_INTR { 
            crate::generated::config::TRACE_INTR_RINGBUF as usize 
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_intr(cause: Word, epc: Word, is_intr: bool) {
    if !TRACE_INTR { return; }
    
    let entry = IntrTraceEntry {
        cause,
        epc,
        is_intr,
    };
    
    INTR_BUF.lock().unwrap().push(entry);
}

pub fn show_intr_trace() {
    if !TRACE_INTR { return; }
    
    crate::Log!("--- RingBuffer Content ---");
    let buf = INTR_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("--------------------------");
}
