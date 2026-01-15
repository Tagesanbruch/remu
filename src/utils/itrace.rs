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
        #[cfg(feature = "trace")]
        {
            Mutex::new(RingBuffer::new(32)) // Default size, should come from config
        }
        #[cfg(not(feature = "trace"))]
        {
            Mutex::new(RingBuffer::new(0))
        }
    };
}

pub fn log_inst(pc: Word, inst: Word) {
    #[cfg(feature = "trace")]
    {
        let entry = ItraceEntry { pc, inst };
        ITRACE_BUF.lock().unwrap().push(entry);
    }
    #[cfg(not(feature = "trace"))]
    {
        let _ = (pc, inst); // Suppress unused warning
    }
}

pub fn show_itrace() {
    #[cfg(feature = "trace")]
    {
        use crate::utils::log::{ANSI_FG_BLUE, ANSI_NONE};
        
        let buf = ITRACE_BUF.lock().unwrap();
        if buf.is_empty() {
            println!("{}[itrace.rs:38 show_itrace] Ringbuffer no element.{}", 
                ANSI_FG_BLUE, ANSI_NONE);
            return;
        }
        
        println!("{}[itrace.rs:42 show_itrace] --- RingBuffer Content ---{}", 
            ANSI_FG_BLUE, ANSI_NONE);
        
        for entry in buf.iter() {
            let disasm_str = disasm::disasm(entry.inst, entry.pc);
            // Format: PC: bytes instruction disasm
            let bytes = format!("{:02x} {:02x} {:02x} {:02x}",
                entry.inst & 0xff,
                (entry.inst >> 8) & 0xff,
                (entry.inst >> 16) & 0xff,
                (entry.inst >> 24) & 0xff);
            
            println!("{}[itrace.rs:44 show_itrace] {:#010x}: {} {}{}", 
                ANSI_FG_BLUE, entry.pc, bytes, disasm_str, ANSI_NONE);
        }
        
        println!("{}[itrace.rs:50 show_itrace] --------------------------{}", 
            ANSI_FG_BLUE, ANSI_NONE);
    }
}
