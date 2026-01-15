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
        // NEMU: WriteMMIO: clint, addr: 0x2000000, data: 0 
        if self.is_write {
             format!("WriteMMIO: {}, addr: 0x{:x}, data: 0x{:x}", 
                self.device_name, self.addr, self.data)
        } else {
             format!("ReadMMIO: {}, addr: 0x{:x}", 
                self.device_name, self.addr)
        }
    }
}

lazy_static::lazy_static! {
    static ref DTRACE_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<DTraceEntry>> = {
        let size = if crate::generated::config::DTRACE { 
            crate::generated::config::DTRACE_RINGBUF as usize 
        } else { 1 };
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
