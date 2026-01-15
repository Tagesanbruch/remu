// Instruction Trace (ITRACE) implementation

use crate::common::Word;
use crate::utils::ringbuffer::RingBuffer;
use crate::isa::riscv32::disasm;
use std::sync::Mutex;
use lazy_static::lazy_static;

#[derive(Clone)]
pub struct ItraceEntry {
    pub pc: Word,
    pub inst: Word,
}

lazy_static! {
    static ref ITRACE_BUF: Mutex<RingBuffer<ItraceEntry>> = {
        let size = if crate::generated::config::ITRACE {
             crate::generated::config::ITRACE_RINGBUF as usize
        } else { 1 };
        Mutex::new(RingBuffer::new(size))
    };
}

pub fn log_inst(pc: Word, inst: Word) {
    if !crate::generated::config::ITRACE { return; }
    
    let entry = ItraceEntry { pc, inst };
    ITRACE_BUF.lock().unwrap().push(entry);
}

pub fn show_itrace() {
    if !crate::generated::config::ITRACE { return; }
    
    let buf = ITRACE_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("Ringbuffer no element.");
        return;
    }
    
    crate::Log!("--- RingBuffer Content ---");
    
    for entry in buf.iter() {
        let disasm_str = disasm::disasm(entry.inst, entry.pc);
        // Format: PC: bytes instruction disasm
        let bytes = format!("{:02x} {:02x} {:02x} {:02x}",
            entry.inst & 0xff,
            (entry.inst >> 8) & 0xff,
            (entry.inst >> 16) & 0xff,
            (entry.inst >> 24) & 0xff);
        
        crate::Log!("0x{:08x}: {} {}", entry.pc, bytes, disasm_str);
    }
    
    crate::Log!("--------------------------");
}
