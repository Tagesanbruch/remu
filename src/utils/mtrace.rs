use crate::generated::config::*;
use crate::common::{PAddr, Word};

#[derive(Clone, Copy)]
pub struct MTraceEntry {
    pub addr: PAddr,
    pub len: usize,
    pub data: Word,
    pub is_write: bool, // true = write, false = read
}

impl ToString for MTraceEntry {
    fn to_string(&self) -> String {
        let type_str = if self.is_write { "write" } else { "read" };
        format!("{} at 0x{:08x} len={} data=0x{:08x}", 
                type_str, self.addr, self.len, self.data)
    }
}

// Global MTRACE ring buffer
lazy_static::lazy_static! {
    static ref MTRACE_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<MTraceEntry>> = {
        let size = if crate::generated::config::MTRACE { 
            crate::generated::config::MTRACE_RINGBUF as usize 
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_read(addr: PAddr, len: usize, data: Word) {
    if !MTRACE { return; }
    
    // Check condition? (Simple true for now)
    
    let entry = MTraceEntry {
        addr,
        len,
        data,
        is_write: false,
    };
    
    MTRACE_BUF.lock().unwrap().push(entry);
}

pub fn trace_write(addr: PAddr, len: usize, data: Word) {
    if !MTRACE { return; }
    
    let entry = MTraceEntry {
        addr,
        len,
        data,
        is_write: true,
    };
    
    MTRACE_BUF.lock().unwrap().push(entry);
}

pub fn show_mtrace() {
    if !MTRACE { return; }
    
    crate::Log!("--- MTRACE Content ---");
    let buf = MTRACE_BUF.lock().unwrap();
    for entry in buf.iter() {
        crate::Log!("{}", entry.to_string());
    }
    crate::Log!("----------------------");
}
