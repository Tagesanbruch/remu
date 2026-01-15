use crate::common::{Word, PAddr, VAddr, PrivMode};
use crate::memory::paddr::paddr_read;
use super::csr::{CSR_SATP, CSR_MSTATUS};
use super::intr::isa_raise_intr;
use crate::Log;

pub const MMU_DIRECT: i32 = 0;
pub const MMU_TRANSLATE: i32 = 1;
pub const MMU_FAIL: i32 = 2;

pub const MEM_TYPE_IFETCH: i32 = 0;
pub const MEM_TYPE_READ: i32 = 1;
pub const MEM_TYPE_WRITE: i32 = 2;

pub fn isa_mmu_check(cpu: &crate::cpu::state::CpuState, _vaddr: VAddr, _len: usize, _type: i32) -> i32 {
    let satp = cpu.csr[CSR_SATP as usize];
    let mode = cpu.mode;
    let _mstatus = cpu.csr[CSR_MSTATUS as usize];
    
    // Check M-Status MPRV? (Not typically used in simple OSs, but good for completeness)
    // For now: paging enabled if SATP_MODE=1 (bit 31) AND Priv < M
    if (satp & 0x80000000) != 0 && (mode != PrivMode::Machine) {
        // crate::Log!("MMU: Check vaddr=0x{:08x} -> TRANSLATE (Mode={:?}, SATP=0x{:08x})", vaddr, mode, satp);
        return MMU_TRANSLATE;
    }
    
    // crate::Log!("MMU: Check vaddr=0x{:08x} -> DIRECT", vaddr); // Verbose
    MMU_DIRECT
}

pub fn isa_mmu_translate(cpu: &crate::cpu::state::CpuState, vaddr: VAddr, _len: usize, type_: i32) -> Result<PAddr, Word> {
    let satp = cpu.csr[CSR_SATP as usize];
    let ppn_base = satp & 0x3FFFFF;
    
    let vpn1 = (vaddr >> 22) & 0x3FF;
    let vpn0 = (vaddr >> 12) & 0x3FF;
    
    let pte_addr_l1 = (ppn_base << 12) + (vpn1 * 4);
    
    let pte_l1 = paddr_read(pte_addr_l1, 4);
    
    // Check valid
    if (pte_l1 & 0x1) == 0 {
        crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
        let cause = report_pf(vaddr, type_);
        return Err(cause);
    }
    
    // Leaf check? (R/W/X bits)
    // If bit 1,2,3 all 0 -> Pointer to next level
    let mut pte = pte_l1;
    let mut pg_size = 0; // 0 = 4KB, 1 = 4MB
    
    if ((pte >> 1) & 7) == 0 {
        // Next Level
        let ppn_l0 = (pte >> 10) & 0x3FFFFF;
        let pte_addr_l0 = (ppn_l0 << 12) + (vpn0 * 4);
        pte = paddr_read(pte_addr_l0, 4);
        
        if (pte & 0x1) == 0 {
             crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
             let cause = report_pf(vaddr, type_);
             return Err(cause);
        }
    } else {
        // Superpage (4MB)
        pg_size = 1; 
    }
    
    // Check Permissions
    let r = (pte >> 1) & 1;
    let w = (pte >> 2) & 1;
    let x = (pte >> 3) & 1;
    
    if type_ == MEM_TYPE_IFETCH && x == 0 { 
        crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
        return Err(report_pf(vaddr, type_)); 
    }
    if type_ == MEM_TYPE_READ && r == 0 { 
        crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
        return Err(report_pf(vaddr, type_)); 
    }
    if type_ == MEM_TYPE_WRITE && w == 0 { 
        crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
        return Err(report_pf(vaddr, type_)); 
    }
    
    // Accessed/Dirty update (Should write back)
    // Skipped for now simplification
    
    let ppn = (pte >> 10) & 0x3FFFFF;
    let paddr = if pg_size == 1 {
        // 4MB
        (ppn << 12) | (vaddr & 0x3FFFFF)
    } else {
        // 4KB
        (ppn << 12) | (vaddr & 0xFFF)
    };
    
    crate::utils::mmu_trace::trace_mmu(vaddr, paddr, type_, true);
    Ok(paddr)
}

fn report_pf(_vaddr: VAddr, type_: i32) -> Word {
    let code = match type_ {
        MEM_TYPE_IFETCH => 12, // Inst PF
        MEM_TYPE_READ => 13,   // Load PF
        MEM_TYPE_WRITE => 15,  // Store PF
        _ => 13
    };
    log::error!("Page Fault: type={}, code={} at vaddr=0x{:08x}", type_, code, _vaddr);
    code
}
