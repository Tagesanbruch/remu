use crate::common::Word;
use crate::cpu::state::CPU;
use crate::Log;

// CSR Indexes (matching local-include/reg.h and NEMU)
pub const CSR_MTVEC: u16 = 0x305;
pub const CSR_MEPC: u16 = 0x341;
pub const CSR_MCAUSE: u16 = 0x342;
pub const CSR_MIE: u16 = 0x304;
pub const CSR_MIP: u16 = 0x344;
pub const CSR_MTVAL: u16 = 0x343;
pub const CSR_MSCRATCH: u16 = 0x340;
pub const CSR_MSTATUS: u16 = 0x300;
pub const CSR_SSTATUS: u16 = 0x100;
pub const CSR_SEPC: u16 = 0x141;
pub const CSR_SCAUSE: u16 = 0x142;
pub const CSR_SIE: u16 = 0x104;
pub const CSR_SIP: u16 = 0x144;
pub const CSR_STVAL: u16 = 0x143;
pub const CSR_SSCRATCH: u16 = 0x140;
pub const CSR_SATP: u16 = 0x180;
pub const CSR_STVEC: u16 = 0x105;
pub const CSR_MEDELEG: u16 = 0x302;
pub const CSR_MIDELEG: u16 = 0x303;
pub const CSR_TIME: u16 = 0xc01;
pub const CSR_TIMEH: u16 = 0xc81;

pub fn isa_csr_read(cpu: &crate::cpu::state::CpuState, addr: u16) -> Word {
    match addr {
        CSR_SSTATUS => {
            // SSTATUS maps to MSTATUS restricted view
            let mstatus = cpu.csr[CSR_MSTATUS as usize];
            mstatus & 0x800DE162 // Mask S-mode visible bits
        }
        CSR_SIE => cpu.csr[CSR_MIE as usize] & cpu.csr[CSR_MIDELEG as usize],
        CSR_SIP => cpu.csr[CSR_MIP as usize] & cpu.csr[CSR_MIDELEG as usize],
        CSR_TIME => {
            let t = crate::device::timer::get_time_u32(0);
            // crate::Log!("CSR: Read TIME -> {}", t);
            t
        },
        CSR_TIMEH => {
            let t = crate::device::timer::get_time_u32(1);
            // crate::Log!("CSR: Read TIMEH -> {}", t);
            t
        },
        _ => {
            if (addr as usize) < cpu.csr.len() {
                cpu.csr[addr as usize]
            } else {
                0
            }
        }
    }
}

pub fn isa_csr_write(cpu: &mut crate::cpu::state::CpuState, addr: u16, data: Word) {
    match addr {
       CSR_SSTATUS => {
           // Write to MSTATUS alias
           let mask = 0x800DE162; // S-mode writable bits
           let old = cpu.csr[CSR_MSTATUS as usize];
           cpu.csr[CSR_MSTATUS as usize] = (old & !mask) | (data & mask);
       }
       CSR_SIE => {
           let mask = cpu.csr[CSR_MIDELEG as usize];
           let old = cpu.csr[CSR_MIE as usize];
           cpu.csr[CSR_MIE as usize] = (old & !mask) | (data & mask);
       }
       CSR_SIP => {
           let mask = cpu.csr[CSR_MIDELEG as usize] & 0x00000002; // Only SSIP is writable in SIP?
           let old = cpu.csr[CSR_MIP as usize];
           cpu.csr[CSR_MIP as usize] = (old & !mask) | (data & mask);
       }
        _ => {
            if (addr as usize) < cpu.csr.len() {
                cpu.csr[addr as usize] = data;
            }
        }
    }
}

pub fn csr_read(addr: u16) -> Word {
    let cpu = CPU.lock().unwrap();
    isa_csr_read(&cpu, addr)
}

pub fn csr_write(addr: u16, data: Word) {
    let mut cpu = CPU.lock().unwrap();
    isa_csr_write(&mut cpu, addr, data);
}
