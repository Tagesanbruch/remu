use crate::common::{Word, PAddr, VAddr, PrivMode};
use crate::cpu::state::CPU;
use crate::memory::paddr_read;
use super::csr::{CSR_SATP, CSR_MSTATUS};
use super::intr::isa_raise_intr;

pub const MMU_DIRECT: i32 = 0;
pub const MMU_TRANSLATE: i32 = 1;
pub const MMU_FAIL: i32 = 2;

pub const MEM_TYPE_IFETCH: i32 = 0;
pub const MEM_TYPE_READ: i32 = 1;
pub const MEM_TYPE_WRITE: i32 = 2;

pub fn isa_mmu_check(cpu: &crate::cpu::state::CpuState, vaddr: VAddr, _len: usize, _type: i32) -> i32 {
    let satp = cpu.csr[CSR_SATP as usize];
    let mode = cpu.mode;
    let _mstatus = cpu.csr[CSR_MSTATUS as usize];
    
    // Check M-Status MPRV? (Not typically used in simple OSs, but good for completeness)
    // For now: paging enabled if SATP_MODE=1 (bit 31) AND Priv < M
    if (satp & 0x80000000) != 0 && (mode != PrivMode::Machine) {
        return MMU_TRANSLATE;
    }
    
    MMU_DIRECT
}

pub fn isa_mmu_translate(cpu: &crate::cpu::state::CpuState, vaddr: VAddr, _len: usize, type_: i32) -> PAddr {
    let satp = cpu.csr[CSR_SATP as usize];
    let ppn_base = satp & 0x3FFFFF;
    
    let vpn1 = (vaddr >> 22) & 0x3FF;
    let vpn0 = (vaddr >> 12) & 0x3FF;
    
    let pte_addr_l1 = (ppn_base << 12) + (vpn1 * 4);
    
    // drop(cpu); // CPU lock is now held by caller or passed in reference
    // We assume caller holds lock if reference is from MutexGuard
    // Or caller passed reference from somewhere else.
    // BUT paddr_read uses PMEM lock. This is safe (CPU -> PMEM).
    
    let pte_l1 = paddr_read(pte_addr_l1, 4);
    
    // Check valid
    if (pte_l1 & 0x1) == 0 {
        // Page Fault
        // isa_raise_intr is tricky because it locks CPU. 
        // We dropped lock above.
        // But isa_mmu_translate typically called from deep inside execution...
        // Let's just return 0 for now and assume caller handles it or we call raise_intr here
        raise_pf(vaddr, type_);
        return 0; // Invalid physical address?
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
             raise_pf(vaddr, type_);
             return 0;
        }
    } else {
        // Superpage (4MB)
        pg_size = 1; 
    }
    
    // Check Permissions
    let r = (pte >> 1) & 1;
    let w = (pte >> 2) & 1;
    let x = (pte >> 3) & 1;
    
    if type_ == MEM_TYPE_IFETCH && x == 0 { raise_pf(vaddr, type_); return 0; }
    if type_ == MEM_TYPE_READ && r == 0 { raise_pf(vaddr, type_); return 0; } // MxR?
    if type_ == MEM_TYPE_WRITE && w == 0 { raise_pf(vaddr, type_); return 0; }
    
    // Accessed/Dirty update (Should write back)
    // Skipped for now simplification
    
    let ppn = (pte >> 10) & 0x3FFFFF;
    if pg_size == 1 {
        // 4MB
        (ppn << 12) | (vaddr & 0x3FFFFF)
    } else {
        // 4KB
        (ppn << 12) | (vaddr & 0xFFF)
    }
}

pub fn isa_vaddr_read(cpu: &crate::cpu::state::CpuState, vaddr: VAddr, len: usize, type_: i32) -> Word {
    if isa_mmu_check(cpu, vaddr, len, type_) == MMU_DIRECT {
        paddr_read(vaddr, len)
    } else {
        let paddr = isa_mmu_translate(cpu, vaddr, len, type_);
        paddr_read(paddr, len)
    }
}

pub fn isa_vaddr_write(cpu: &crate::cpu::state::CpuState, vaddr: VAddr, len: usize, data: Word, type_: i32) {
    if isa_mmu_check(cpu, vaddr, len, type_) == MMU_DIRECT {
        crate::memory::paddr_write(vaddr, len, data);
    } else {
        let paddr = isa_mmu_translate(cpu, vaddr, len, type_);
        crate::memory::paddr_write(paddr, len, data);
    }
}

fn raise_pf(_vaddr: VAddr, type_: i32) {
    let code = match type_ {
        MEM_TYPE_IFETCH => 12, // Inst PF
        MEM_TYPE_READ => 13,   // Load PF
        MEM_TYPE_WRITE => 15,  // Store PF
        _ => 13
    };
    log::error!("Page Fault: type={}, code={}", type_, code);
    log::warn!("Minimal implementation does not support proper PF raising in mmu.rs yet to avoid deadlock.");
    // Ideally we return an Error from translate, and inst.rs handles it by calling raise_intr.
}
