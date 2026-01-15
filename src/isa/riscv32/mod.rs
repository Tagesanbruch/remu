// RISC-V32 ISA implementation

pub mod decode;
pub mod inst;
pub mod disasm;

use crate::common::Word;
// use crate::cpu::state::CPU;  // Unused
use crate::memory::paddr_read;

pub fn isa_exec_once(pc: Word) {
    // Fetch instruction
    let inst = paddr_read(pc, 4);
    
    // Log instruction trace
    #[cfg(feature = "trace")]
    {
        crate::utils::itrace::log_inst(pc, inst);
    }
    
    // Decode and execute
    inst::decode_exec(inst, pc);
}
