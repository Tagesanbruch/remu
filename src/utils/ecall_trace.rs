// ECALL Trace

use crate::generated::config::*;
use crate::common::Word;

#[derive(Clone)]
pub struct EcallTraceEntry {
    pub pc: Word,
    pub cause: Word,
    pub mode: u8,
}

impl ToString for EcallTraceEntry {
    fn to_string(&self) -> String {
        let mode_str = match self.mode {
            3 => "Machine",
            1 => "Supervisor",
            0 => "User",
            _ => "Unknown"
        };
        format!("ECALL: Mode={} Cause={} @ PC=0x{:08x}", 
            mode_str, self.cause, self.pc)
    }
}

lazy_static::lazy_static! {
    static ref ECALL_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<EcallTraceEntry>> = {
        let size = if crate::generated::config::TRACE_ECALL { 
            crate::generated::config::TRACE_ECALL_RINGBUF as usize 
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_ecall(pc: Word, cause: Word, mode: u8) {
    if !TRACE_ECALL { return; }
    
    let entry = EcallTraceEntry {
        pc,
        cause,
        mode,
    };
    
    ECALL_BUF.lock().unwrap().push(entry);
}

pub fn show_ecall_trace() {
    if !TRACE_ECALL { return; }
    
    crate::Log!("--- RingBuffer Content ---");
    let buf = ECALL_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("--------------------------");
}
