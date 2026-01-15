use std::fs;
use goblin::elf::Elf;
use crate::common::VAddr;
use crate::generated::config::*;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub addr: VAddr,
    pub size: u32,
}

#[derive(Clone)]
pub struct FTraceEntry {
    pub pc: VAddr,
    pub target: VAddr,
    pub is_call: bool, // true = call, false = ret
    pub call_depth: usize,
    pub target_name: String,
}

impl ToString for FTraceEntry {
    fn to_string(&self) -> String {
        let indent = "  ".repeat(self.call_depth);
        if self.is_call {
            format!("{}call [{}] @ 0x{:08x}", indent, self.target_name, self.target)
        } else {
            format!("{}ret [{}]", indent, self.target_name)
        }
    }
}

pub struct FTrace {
    pub symbols: Vec<Symbol>,
    pub call_depth: usize,
    pub buf: crate::utils::ringbuffer::RingBuffer<FTraceEntry>,
}

impl FTrace {
    pub fn new() -> Self {
        let size = if FTRACE { 1024 } else { 1 }; // Hardcoded default or from config
        Self {
            symbols: Vec::new(),
            call_depth: 0,
            buf: crate::utils::ringbuffer::RingBuffer::new(size),
        }
    }

    pub fn load_elf(&mut self, elf_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = fs::read(elf_file)?;
        let elf = Elf::parse(&buffer)?;
        
        for sym in elf.syms.iter() {
            let name = if let Some(name) = elf.strtab.get_at(sym.st_name) {
                name
            } else {
                continue;
            };
            
            // Filter function symbols (STT_FUNC = 2)
            if sym.st_type() == 2 {
                self.symbols.push(Symbol {
                    name: name.to_string(),
                    addr: sym.st_value as u32,
                    size: sym.st_size as u32,
                });
            }
        }
        
        self.symbols.sort_by_key(|s| s.addr);
        println!("[FTRACE] Loaded {} function symbols from {}", self.symbols.len(), elf_file);
        Ok(())
    }

    pub fn find_symbol(&self, addr: VAddr) -> Option<&Symbol> {
        // Binary search or linear scan
        for sym in &self.symbols {
            if addr >= sym.addr && addr < sym.addr + sym.size {
                return Some(sym);
            }
        }
        None
    }
    
    pub fn trace_call(&mut self, pc: VAddr, target: VAddr) {
        if !FTRACE { return; }
        
        let target_name = if let Some(sym) = self.find_symbol(target) {
             sym.name.clone()
        } else {
             "???".to_string()
        };
        
        let entry = FTraceEntry {
            pc,
            target,
            is_call: true,
            call_depth: self.call_depth,
            target_name,
        };
        
        self.buf.push(entry);
        self.call_depth += 1;
    }
    
    pub fn trace_ret(&mut self, pc: VAddr) {
        if !FTRACE { return; }
        
        if self.call_depth > 0 {
            self.call_depth -= 1;
        }
        
        let ret_name = if let Some(sym) = self.find_symbol(pc) {
             sym.name.clone()
        } else {
             "???".to_string()
        };
        
        let entry = FTraceEntry {
            pc,
            target: 0,
            is_call: false,
            call_depth: self.call_depth,
            target_name: ret_name,
        };
        
        self.buf.push(entry);
    }
    
    pub fn show(&self) {
        if !crate::generated::config::FTRACE { return; }
        // NEMU uses Log for header, loop for content
        crate::Log!("--- FTRACE Content ---");
        for entry in self.buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
        crate::Log!("----------------------");
    }
}

lazy_static::lazy_static! {
    pub static ref FTRACE_INST: std::sync::Mutex<FTrace> = {
        std::sync::Mutex::new(FTrace::new())
    };
}

pub fn init_ftrace(elf_file: &str) {
    if !FTRACE { return; }
    
    if let Err(e) = FTRACE_INST.lock().unwrap().load_elf(elf_file) {
        eprintln!("Failed to load ELF file for FTRACE: {}", e);
    }
}

pub fn trace_call(pc: VAddr, target: VAddr) {
    FTRACE_INST.lock().unwrap().trace_call(pc, target);
}

pub fn trace_ret(pc: VAddr) {
    FTRACE_INST.lock().unwrap().trace_ret(pc);
}

// Added show_ftrace function to be called from common panic
pub fn show_ftrace() {
    FTRACE_INST.lock().unwrap().show();
}
