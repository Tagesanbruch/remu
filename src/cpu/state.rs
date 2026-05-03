// CPU state structure

use crate::common::{mask_xlen, PrivMode, Word};
use crate::config::RuntimeConfig;
use std::sync::{Arc, Mutex};

pub struct CpuState {
    pub pc: Word,
    pub gpr: [Word; 32],
    pub fpr: [u64; 32],
    pub csr: [Word; 4096],
    pub mode: PrivMode,
    pub is_exception: bool,
    pub exception_entry: Word,
}

impl CpuState {
    pub fn new() -> Self {
        Self {
            pc: 0,
            gpr: [0; 32],
            fpr: [0; 32],
            csr: [0; 4096],
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
        self.fpr = [0; 32];

        // Initialize key CSRs
        self.init_csr();

        // Start in Machine mode
        self.mode = PrivMode::Machine;

        log::info!("CPU initialized: PC = 0x{:08x}", self.pc);
    }

    pub fn init_csr(&mut self) {
        // mstatus
        self.csr[0x300] = 0x1800; // MPP=11 (Machine)

        // misa: MXL plus I/M/A/S/C/F/D.  F/D currently supports raw register loads/stores.
        let mxl = if crate::generated::config::RV64 {
            2_u64 << 62
        } else {
            1_u64 << 30
        };
        let misa = mxl
            | (1 << 0)  // A
            | (1 << 2)  // C
            | (1 << 3)  // D
            | (1 << 5)  // F
            | (1 << 8)  // I
            | (1 << 12) // M
            | (1 << 18); // S
        self.csr[0x301] = misa;
    }

    pub fn get_gpr(&self, idx: usize) -> Word {
        if idx == 0 {
            0 // x0 is always 0
        } else {
            self.gpr[idx]
        }
    }

    pub fn set_gpr(&mut self, idx: usize, val: Word) {
        if idx != 0 {
            self.gpr[idx] = mask_xlen(val);
        }
    }

    pub fn get_csr(&self, addr: u16) -> Word {
        self.csr[addr as usize]
    }

    pub fn set_csr(&mut self, addr: u16, val: Word) {
        self.csr[addr as usize] = val;
    }
}

// Global CPU instance
lazy_static::lazy_static! {
    pub static ref CPU: Arc<Mutex<CpuState>> = Arc::new(Mutex::new(CpuState::new()));
}
