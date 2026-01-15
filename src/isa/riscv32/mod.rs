// RISC-V32 ISA implementation

pub mod decode;
pub mod inst;
pub mod disasm;
pub mod system;

use crate::common::Word;
// use crate::cpu::state::CPU;  // Unused
use self::system::mmu::{isa_vaddr_read, MEM_TYPE_IFETCH};

pub fn isa_exec_once(pc: Word) {
    // Fetch instruction
    let inst = {
        let cpu = crate::cpu::state::CPU.lock().unwrap();
        isa_vaddr_read(&cpu, pc, 4, MEM_TYPE_IFETCH)
    };
    
    // Log instruction trace
    #[cfg(feature = "trace")]
    {
        crate::utils::itrace::log_inst(pc, inst);
    }
    
    // Decode and execute
    inst::decode_exec(inst, pc);
}
