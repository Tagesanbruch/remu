// CPU state structure

use crate::common::{Word, PrivMode};
use crate::config::RuntimeConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct CpuState {
    pub pc: u32,
    pub gpr: [Word; 32],
    pub csr: HashMap<u16, Word>,
    pub mode: PrivMode,
    pub is_exception: bool,
    pub exception_entry: u32,
}

impl CpuState {
    pub fn new() -> Self {
        Self {
            pc: 0,
            gpr: [0; 32],
            csr: HashMap::new(),
            mode: PrivMode::Machine,
            is_exception: false,
            exception_entry: 0,
        }
    }

    pub fn init(&mut self) {
        // Reset PC to reset vector
        let cfg = RuntimeConfig::default();
        self.pc = crate::config::reset_vector(&cfg);
        
        // Zero all GPRs
        self.gpr = [0; 32];
        
        // Initialize key CSRs
        self.init_csr();
        
        // Start in Machine mode
        self.mode = PrivMode::Machine;
        
        log::info!("CPU initialized: PC = 0x{:08x}", self.pc);
    }

    pub fn init_csr(&mut self) {
        // mstatus
        self.csr.insert(0x300, 0x1800);
        
        // misa: MXL=1 (32-bit), Extensions: I(8), M(12), A(0), S(18)
        let misa = (1 << 30) | (1 << 0) | (1 << 8) | (1 << 12) | (1 << 18);
        self.csr.insert(0x301, misa);
    }

    pub fn get_gpr(&self, idx: usize) -> Word {
        if idx == 0 {
            0  // x0 is always 0
        } else {
            self.gpr[idx]
        }
    }

    pub fn set_gpr(&mut self, idx: usize, val: Word) {
        if idx != 0 {
            self.gpr[idx] = val;
        }
    }

    pub fn get_csr(&self, addr: u16) -> Word {
        *self.csr.get(&addr).unwrap_or(&0)
    }

    pub fn set_csr(&mut self, addr: u16, val: Word) {
        self.csr.insert(addr, val);
    }
}

// Global CPU instance
lazy_static::lazy_static! {
    pub static ref CPU: Arc<Mutex<CpuState>> = Arc::new(Mutex::new(CpuState::new()));
}
