// Virtual address access implementation

use crate::common::{VAddr, Word};
use crate::isa::riscv32::system::mmu::{isa_mmu_check, isa_mmu_translate, MMU_DIRECT};
use crate::memory::paddr::paddr_read;

// Access types from mmu.rs
pub const MEM_TYPE_IFETCH: i32 = 0;
pub const MEM_TYPE_READ: i32 = 1;
pub const MEM_TYPE_WRITE: i32 = 2;

pub fn vaddr_read(
    cpu: &crate::cpu::state::CpuState,
    vaddr: VAddr,
    len: usize,
) -> Result<Word, Word> {
    if isa_mmu_check(cpu, vaddr, len, MEM_TYPE_READ) == MMU_DIRECT {
        Ok(paddr_read(vaddr, len))
    } else {
        match isa_mmu_translate(cpu, vaddr, len, MEM_TYPE_READ) {
            Ok(paddr) => Ok(paddr_read(paddr, len)),
            Err(cause) => Err(cause),
        }
    }
}

pub fn vaddr_write(
    cpu: &crate::cpu::state::CpuState,
    vaddr: VAddr,
    len: usize,
    data: Word,
) -> Result<(), Word> {
    if isa_mmu_check(cpu, vaddr, len, MEM_TYPE_WRITE) == MMU_DIRECT {
        crate::memory::paddr::paddr_write(vaddr, len, data);
        Ok(())
    } else {
        match isa_mmu_translate(cpu, vaddr, len, MEM_TYPE_WRITE) {
            Ok(paddr) => {
                crate::memory::paddr::paddr_write(paddr, len, data);
                Ok(())
            }
            Err(cause) => Err(cause),
        }
    }
}

pub fn vaddr_ifetch(
    cpu: &crate::cpu::state::CpuState,
    vaddr: VAddr,
    len: usize,
) -> Result<Word, Word> {
    if isa_mmu_check(cpu, vaddr, len, MEM_TYPE_IFETCH) == MMU_DIRECT {
        Ok(paddr_read(vaddr, len))
    } else {
        // mmu_translate returns Result<PAddr, Word> (cause)
        match isa_mmu_translate(cpu, vaddr, len, MEM_TYPE_IFETCH) {
            Ok(paddr) => Ok(paddr_read(paddr, len)),
            Err(cause) => Err(cause),
        }
    }
}
