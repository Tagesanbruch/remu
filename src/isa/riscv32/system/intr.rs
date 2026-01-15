use crate::common::{Word, PrivMode};
use crate::cpu::state::CPU;
use super::csr::*;

// Interrupt/Exception checking
pub fn isa_query_intr() -> Word {
    let cpu = CPU.lock().unwrap();
    
    let mstatus = cpu.csr[CSR_MSTATUS as usize];
    let mie = (mstatus >> 3) & 1;
    let mie_reg = cpu.csr[CSR_MIE as usize];
    // Include CLINT interrupts dynamically
    let clint_mip = crate::device::clint::get_mip_status();
    let ext_mip = crate::device::intr::get_intr_state();
    let mip_reg = cpu.csr[CSR_MIP as usize] | clint_mip | ext_mip;
    
    let mode = cpu.mode as u32; // 3=M, 1=S, 0=U
    
    // Check M-mode interrupts
    // Enabled if (mode < M) OR (mode == M && diff_MIE=1)
    let m_enable = (mode < 3) || ((mode == 3) && (mie != 0));
    
    if m_enable && ((mip_reg & mie_reg) != 0) {
        // Prioritize: MEIP(11) > MTIP(7) > MSIP(3)
        // Note: bit flags check
        if (mip_reg & (1 << 11)) != 0 && (mie_reg & (1 << 11)) != 0 { return 0x80000000 | 11; }
        if (mip_reg & (1 << 7)) != 0 && (mie_reg & (1 << 7)) != 0 { return 0x80000000 | 7; }
        if (mip_reg & (1 << 3)) != 0 && (mie_reg & (1 << 3)) != 0 { return 0x80000000 | 3; }
    }
    
    // Check S-mode interrupts
    let sie_bit = (mstatus >> 1) & 1;
    let s_enable = (mode < 1) || ((mode == 1) && (sie_bit != 0));
    
    if s_enable {
        let mideleg = cpu.csr[CSR_MIDELEG as usize];
        let pending = mip_reg & mie_reg & mideleg;
        
        if (pending & (1 << 9)) != 0 { return 0x80000000 | 9; } // SEIP
        if (pending & (1 << 5)) != 0 { return 0x80000000 | 5; } // STIP
        if (pending & (1 << 1)) != 0 { return 0x80000000 | 1; } // SSIP
    }
    
    0 // INTR_EMPTY
}

pub fn isa_raise_intr(no: Word, epc: Word) -> Word {
    let mut cpu = CPU.lock().unwrap();
    
    let is_intr = (no & 0x80000000) != 0;
    let cause_code = no & 0x7FFFFFFF;
    
    // Delegation check
    let deleg_reg = if is_intr {
        cpu.csr[CSR_MIDELEG as usize]
    } else {
        cpu.csr[CSR_MEDELEG as usize]
    };
    
    let mut delegate_to_s = false;
    if cpu.mode != PrivMode::Machine {
        if ((deleg_reg >> cause_code) & 1) != 0 {
            delegate_to_s = true;
        }
    }
    
    if delegate_to_s {
        // Trap to S-mode
        // crate::Log!("INTR: Delegated to S-mode Cause 0x{:x} at 0x{:08x}", cause_code, epc);
        crate::utils::intr_trace::trace_intr(cause_code, epc, is_intr);
        cpu.csr[CSR_SCAUSE as usize] = no;
        cpu.csr[CSR_SEPC as usize] = epc;
        cpu.csr[CSR_STVAL as usize] = 0; // TODO: tval
        
        // Update SSTATUS
        // SPIE = SIE, SIE = 0, SPP = Mode
        let _sstatus = cpu.csr[CSR_SSTATUS as usize]; // Actually MSTATUS masked
        // Need to operate on MSTATUS essentially
        let mut mstatus = cpu.csr[CSR_MSTATUS as usize];
        
        let sie = (mstatus >> 1) & 1;
        mstatus &= !0x122; // Clear SPIE(5), SIE(1), SPP(8)
        mstatus |= sie << 5; // SPIE = old SIE
        mstatus |= (cpu.mode as Word) << 8; // SPP = old mode
        
        cpu.csr[CSR_MSTATUS as usize] = mstatus;
        
        cpu.mode = PrivMode::Supervisor;
        cpu.is_exception = true;
        
        // Return STVEC
        cpu.csr[CSR_STVEC as usize]
    } else {
        // Trap to M-mode
        // crate::Log!("INTR: {} -> M-mode Cause 0x{:x} at 0x{:08x}", if is_intr { "Intr" } else { "Excp" }, cause_code, epc);
        crate::utils::intr_trace::trace_intr(cause_code, epc, is_intr);
        cpu.csr[CSR_MCAUSE as usize] = no;
        cpu.csr[CSR_MEPC as usize] = epc;
        cpu.csr[CSR_MTVAL as usize] = 0; // TODO: tval
        
        // Update MSTATUS
        // MPIE = MIE, MIE = 0, MPP = Mode
        let mut mstatus = cpu.csr[CSR_MSTATUS as usize];
        let mie = (mstatus >> 3) & 1;
        mstatus &= !0x1888; // Clear MPIE(7), MIE(3), MPP(11,12)
        mstatus |= mie << 7; // MPIE = old MIE
        mstatus |= (cpu.mode as Word) << 11; // MPP = old mode
        
        cpu.csr[CSR_MSTATUS as usize] = mstatus;
        
        cpu.mode = PrivMode::Machine;
        cpu.is_exception = true;
        
        // Return MTVEC
        cpu.csr[CSR_MTVEC as usize]
    }
}
