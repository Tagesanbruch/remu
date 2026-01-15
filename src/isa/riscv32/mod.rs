// RISC-V32 ISA implementation

pub mod decode;
pub mod inst;
pub mod disasm;
pub mod system;

use crate::common::Word;
// use crate::cpu::state::CPU;  // Unused
// use self::system::mmu::{isa_vaddr_read, MEM_TYPE_IFETCH};

pub fn isa_exec_once(pc: Word) {
    // Fetch instruction
    let inst_result = {
        let cpu = crate::cpu::state::CPU.lock().unwrap();
        crate::memory::vaddr::vaddr_ifetch(&cpu, pc, 4)
    };
    
    match inst_result {
        Ok(inst) => {
            // Log instruction trace
            // #[cfg(feature = "trace")]
            {
                crate::utils::itrace::log_inst(pc, inst);
            }
            // Decode and execute
            inst::decode_exec(inst, pc);
        }
        Err(cause) => {
             // Raise Instruction Page Fault (12)
             crate::utils::intr_trace::trace_intr(cause, pc, false); // Trace exception
             let new_pc = self::system::intr::isa_raise_intr(cause, pc);
             let mut cpu = crate::cpu::state::CPU.lock().unwrap();
             cpu.pc = new_pc; // Update PC to trap vector
        }
    }
}
