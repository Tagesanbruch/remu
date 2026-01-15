// Device Trace (DTRACE)

use crate::generated::config::*;
use crate::common::{PAddr, Word};

#[derive(Clone)]
pub struct DTraceEntry {
    pub addr: PAddr,
    pub len: usize,
    pub data: Word,
    pub is_write: bool,
    pub device_name: String,
}

impl ToString for DTraceEntry {
    fn to_string(&self) -> String {
        let type_str = if self.is_write { "write" } else { "read" };
        format!("{} {} at 0x{:08x} len={} data=0x{:08x}", 
                self.device_name, type_str, self.addr, self.len, self.data)
    }
}

lazy_static::lazy_static! {
    static ref DTRACE_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<DTraceEntry>> = {
        let size = if DTRACE { DTRACE_RINGBUF as usize } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_dtrace(addr: PAddr, len: usize, data: Word, is_write: bool, device_name: &str) {
    if !DTRACE { return; }
    
    let entry = DTraceEntry {
        addr,
        len,
        data,
        is_write,
        device_name: device_name.to_string(),
    };
    
    DTRACE_BUF.lock().unwrap().push(entry);
}

pub fn show_dtrace() {
    if !DTRACE { return; }
    
    crate::Log!("--- DTRACE Content ---");
    let buf = DTRACE_BUF.lock().unwrap();
    // Only show if not empty
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("----------------------");
}
