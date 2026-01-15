// RISC-V instruction execution

use super::decode::DecodedInst;
use crate::common::{Word, SWord, RemuState};
// use crate::cpu::state::CPU;
// inst.rs doesn't seem to use them other than for those calls.
// Let's keep them if unsure, or remove. The compiler warned about unused imports before.
use crate::memory::vaddr::{vaddr_read, vaddr_write};
use crate::utils::set_state;

macro_rules! R {
    ($cpu:expr, $idx:expr) => {
        $cpu.get_gpr($idx)
    };
}

macro_rules! W {
    ($cpu:expr, $idx:expr, $val:expr) => {
        $cpu.set_gpr($idx, $val)
    };
}

pub fn decode_exec(cpu: &mut crate::cpu::state::CpuState, inst: Word, pc: Word) {
    let mut dec = DecodedInst::new(inst);
    
    // Default next PC
    let mut dnpc = pc.wrapping_add(4);
    
    // Get opcode and dispatch
    match dec.opcode {
        // LUI
        0b0110111 => {
            dec.decode_u();
            W!(cpu, dec.rd, dec.imm);
        }
        // AUIPC
        0b0010111 => {
            dec.decode_u();
            W!(cpu, dec.rd, pc.wrapping_add(dec.imm));
        }
        // JAL
        0b1101111 => {
            dec.decode_j();
            W!(cpu, dec.rd, pc.wrapping_add(4));
            dnpc = pc.wrapping_add(dec.imm);
            
            // FTRACE: call
            crate::utils::ftrace::trace_call(pc, dnpc);
        }
        // JALR
        0b1100111 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            W!(cpu, dec.rd, pc.wrapping_add(4));
            dnpc = (src1.wrapping_add(dec.imm)) & !1;
            
            // FTRACE
            if dec.rd == 0 && dec.rs1 == 1 && dec.imm == 0 {
                 // ret
                 crate::utils::ftrace::trace_ret(pc);
            } else {
                 // call
                 crate::utils::ftrace::trace_call(pc, dnpc);
            }
        }
        // Branch instructions
        0b1100011 => {
            dec.decode_b();
            let src1 = R!(cpu, dec.rs1);
            let src2 = R!(cpu, dec.rs2);
            let taken = match dec.funct3 {
                0b000 => src1 == src2,  // BEQ
                0b001 => src1 != src2,  // BNE
                0b100 => (src1 as SWord) < (src2 as SWord),  // BLT
                0b101 => (src1 as SWord) >= (src2 as SWord),  // BGE
                0b110 => src1 < src2,  // BLTU
                0b111 => src1 >= src2,  // BGEU
                _ => {
                    log::error!("Invalid branch funct3: 0b{:03b}", dec.funct3);
                    false
                }
            };
            if taken {
                dnpc = pc.wrapping_add(dec.imm);
            }
        }
        // Load instructions
        0b0000011 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            let addr = src1.wrapping_add(dec.imm);
            let val = match dec.funct3 {
                0b000 => {  // LB
                    let v = vaddr_read(&*cpu, addr, 1);
                    ((v as i8) as i32) as u32
                }
                0b001 => {  // LH
                    let v = vaddr_read(&*cpu, addr, 2);
                    ((v as i16) as i32) as u32
                }
                0b010 => vaddr_read(&*cpu, addr, 4),  // LW
                0b100 => vaddr_read(&*cpu, addr, 1),  // LBU
                0b101 => vaddr_read(&*cpu, addr, 2),  // LHU
                _ => {
                    log::error!("Invalid load funct3: 0b{:03b}", dec.funct3);
                    0
                }
            };
            W!(cpu, dec.rd, val);
        }
        // Store instructions
        0b0100011 => {
            dec.decode_s();
            let src1 = R!(cpu, dec.rs1);
            let src2 = R!(cpu, dec.rs2);
            let addr = src1.wrapping_add(dec.imm);
            match dec.funct3 {
                0b000 => vaddr_write(&*cpu, addr, 1, src2),  // SB
                0b001 => vaddr_write(&*cpu, addr, 2, src2),  // SH
                0b010 => vaddr_write(&*cpu, addr, 4, src2),  // SW
                _ => log::error!("Invalid store funct3: 0b{:03b}", dec.funct3),
            }
        }
        // I-type ALU instructions
        0b0010011 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            let val = match dec.funct3 {
                0b000 => src1.wrapping_add(dec.imm),  // ADDI
                0b010 => ((src1 as SWord) < (dec.imm as SWord)) as u32,  // SLTI
                0b011 => (src1 < dec.imm) as u32,  // SLTIU
                0b100 => src1 ^ dec.imm,  // XORI
                0b110 => src1 | dec.imm,  // ORI
                0b111 => src1 & dec.imm,  // ANDI
                0b001 => {  // SLLI
                    let shamt = dec.imm & 0x1f;
                    src1 << shamt
                }
                0b101 => {  // SRLI / SRAI
                    let shamt = dec.imm & 0x1f;
                    if (dec.imm >> 10) & 1 == 1 {
                        // SRAI
                        ((src1 as SWord) >> shamt) as u32
                    } else {
                        // SRLI
                        src1 >> shamt
                    }
                }
                _ => {
                    log::error!("Invalid I-type ALU funct3: 0b{:03b}", dec.funct3);
                    0
                }
            };
            W!(cpu, dec.rd, val);
        }
        // R-type ALU instructions
        0b0110011 => {
            dec.decode_r();
            let src1 = R!(cpu, dec.rs1);
            let src2 = R!(cpu, dec.rs2);
            let val = match (dec.funct7, dec.funct3) {
                (0b0000000, 0b000) => src1.wrapping_add(src2),  // ADD
                (0b0100000, 0b000) => src1.wrapping_sub(src2),  // SUB
                (0b0000000, 0b001) => src1 << (src2 & 0x1f),  // SLL
                (0b0000000, 0b010) => ((src1 as SWord) < (src2 as SWord)) as u32,  // SLT
                (0b0000000, 0b011) => (src1 < src2) as u32,  // SLTU
                (0b0000000, 0b100) => src1 ^ src2,  // XOR
                (0b0000000, 0b101) => src1 >> (src2 & 0x1f),  // SRL
                (0b0100000, 0b101) => ((src1 as SWord) >> (src2 & 0x1f)) as u32,  // SRA
                (0b0000000, 0b110) => src1 | src2,  // OR
                (0b0000000, 0b111) => src1 & src2,  // AND
                // M extension
                (0b0000001, 0b000) => src1.wrapping_mul(src2),  // MUL
                (0b0000001, 0b001) => mulh(src1 as SWord, src2 as SWord) as u32,  // MULH
                (0b0000001, 0b010) => mulhsu(src1 as SWord, src2),  // MULHSU
                (0b0000001, 0b011) => mulhu(src1, src2),  // MULHU
                (0b0000001, 0b100) => {  // DIV
                    if src2 == 0 {
                        0xffffffff
                    } else {
                        ((src1 as SWord).wrapping_div(src2 as SWord)) as u32
                    }
                }
                (0b0000001, 0b101) => {  // DIVU
                    if src2 == 0 {
                        0xffffffff
                    } else {
                        src1 / src2
                    }
                }
                (0b0000001, 0b110) => {  // REM
                    if src2 == 0 {
                        src1
                    } else {
                        ((src1 as SWord).wrapping_rem(src2 as SWord)) as u32
                    }
                }
                (0b0000001, 0b111) => {  // REMU
                    if src2 == 0 {
                        src1
                    } else {
                        src1 % src2
                    }
                }
                _ => {
                    log::error!("Invalid R-type funct7/funct3: 0b{:07b}/0b{:03b}", dec.funct7, dec.funct3);
                    0
                }
            };
            W!(cpu, dec.rd, val);
        }
        // RV32A Atomic instructions
        0b0101111 => {
            dec.decode_r();
            let src1 = R!(cpu, dec.rs1);
            let addr = src1;
            
            match (dec.funct7 >> 2, dec.funct3) {
                (0b00010, 0b010) => {  // LR.W
                    let val = vaddr_read(&*cpu, addr, 4);
                    W!(cpu, dec.rd, val);
                    // TODO: Set reservation
                }
                (0b00011, 0b010) => {  // SC.W
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, src2);
                    W!(cpu, dec.rd, 0);  // Always succeed for now
                    // TODO: Check reservation
                }
                (0b00001, 0b010) => {  // AMOSWAP.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, src2);
                    W!(cpu, dec.rd, t);
                }
                (0b00000, 0b010) => {  // AMOADD.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, t.wrapping_add(src2));
                    W!(cpu, dec.rd, t);
                }
                (0b00100, 0b010) => {  // AMOXOR.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, t ^ src2);
                    W!(cpu, dec.rd, t);
                }
                (0b01100, 0b010) => {  // AMOAND.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, t & src2);
                    W!(cpu, dec.rd, t);
                }
                (0b01000, 0b010) => {  // AMOOR.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    vaddr_write(&*cpu, addr, 4, t | src2);
                    W!(cpu, dec.rd, t);
                }
                (0b10000, 0b010) => {  // AMOMIN.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    let min = if (t as SWord) < (src2 as SWord) { t } else { src2 };
                    vaddr_write(&*cpu, addr, 4, min);
                    W!(cpu, dec.rd, t);
                }
                (0b10100, 0b010) => {  // AMOMAX.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    let max = if (t as SWord) > (src2 as SWord) { t } else { src2 };
                    vaddr_write(&*cpu, addr, 4, max);
                    W!(cpu, dec.rd, t);
                }
                (0b11000, 0b010) => {  // AMOMINU.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    let min = if t < src2 { t } else { src2 };
                    vaddr_write(&*cpu, addr, 4, min);
                    W!(cpu, dec.rd, t);
                }
                (0b11100, 0b010) => {  // AMOMAXU.W
                    let t = vaddr_read(&*cpu, addr, 4);
                    let src2 = R!(cpu, dec.rs2);
                    let max = if t > src2 { t } else { src2 };
                    vaddr_write(&*cpu, addr, 4, max);
                    W!(cpu, dec.rd, t);
                }
                _ => {
                    log::error!("Invalid atomic instruction: funct7={:07b}, funct3={:03b}",
                               dec.funct7, dec.funct3);
                }
            }
        }
        // FENCE (NOP for now)
        0b0001111 => {
            // FENCE/FENCE.I - treated as NOP
        }
        // FENCE.VMA / SFENCE.VMA
         0b0001001 => {
             // SFENCE.VMA - TLB flush, treated as NOP for now (flushing not strictly needed if we don't cache translations persistently across flushes properly yet, or if we just want to proceed)
             // opcode 1110011 (system), funct3 000, funct7 0001001
         }
        // System instructions (0b1110011)
        0b1110011 => {
             // Special check for SFENCE.VMA which is System opcode but funct7=0001001
             if dec.funct7 == 0b0001001 {
                  // SFENCE.VMA
                  // return; // Don't return, let PC update!
             }
             
             match (dec.funct7, dec.rs2, dec.funct3) {
                (0b0000000, 0b00000, 0b000) => {  // ECALL
                    // Determine mode for ECALL cause (User=8, Supervisor=9, Machine=11)
                    let cause = match cpu.mode {
                        crate::common::PrivMode::Machine => 11,
                        crate::common::PrivMode::Supervisor => 9,
                        crate::common::PrivMode::User => 8,
                    };
                    crate::utils::ecall_trace::trace_ecall(pc, cause, cpu.mode as u8);
                    let new_pc = super::system::intr::isa_raise_intr(cpu, cause, pc);
                    cpu.pc = new_pc;
                    return;
                }
                 (0b0000000, 0b00001, 0b000) => {  // EBREAK
                     // EBREAK cause = 3
                     let new_pc = super::system::intr::isa_raise_intr(cpu, 3, pc);
                     cpu.pc = new_pc;
                     return;
                }
                (0b0011000, 0b00010, 0b000) => { // MRET
                    let mstatus = super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_MSTATUS);
                    let mepc = super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_MEPC);
                    
                    // Restore MIE = MPIE
                    let mpie = (mstatus >> 7) & 1;
                    // Restore Priv = MPP
                    let mpp = (mstatus >> 11) & 3;
                     
                    // MIE(3) <- MPIE(7)
                    // MIE = MPIE; MPIE = 1; MPP = U(0);
                    let mut new_mstatus = (mstatus & !(1 << 3)) | (mpie << 3);
                    new_mstatus |= 1 << 7; // MPIE = 1
                    new_mstatus &= !(3 << 11); // MPP = 0 (User)
                    
                    super::system::csr::isa_csr_write(cpu, super::system::csr::CSR_MSTATUS, new_mstatus);
                    
                    cpu.mode = match mpp {
                        3 => crate::common::PrivMode::Machine,
                        1 => crate::common::PrivMode::Supervisor,
                        _ => crate::common::PrivMode::User
                    };
                    
                    cpu.pc = mepc;
                    // dnpc not needed as we update cpu.pc directly and loop continues unless we return?
                    // decode_exec updates cpu.pc = dnpc at end.
                    // We should just return early after setting cpu.pc
                    return;
                }
                (0b0001000, 0b00010, 0b000) => { // SRET
                     // Similar to MRET but for Supervisor
                     let sstatus = super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_SSTATUS); // actually accesses MSTATUS
                     let sepc = super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_SEPC);
                     
                     // Restore SIE = SPIE
                     let spie = (sstatus >> 5) & 1;
                     let spp = (sstatus >> 8) & 1;
                     
                     // SIE(1) <- SPIE(5)
                     let mut new_sstatus = (sstatus & !(1 << 1)) | (spie << 1);
                     new_sstatus |= 1 << 5; // SPIE = 1
                     new_sstatus &= !(1 << 8); // SPP = 0 (User)
                     
                     // Need to write back to MSTATUS (handled by set_csr SSTATUS alias)
                     super::system::csr::isa_csr_write(cpu, super::system::csr::CSR_SSTATUS, new_sstatus);
                     
                     cpu.mode = match spp {
                         1 => crate::common::PrivMode::Supervisor,
                         _ => crate::common::PrivMode::User
                     };
                     
                     cpu.pc = sepc;
                     return;
                }
                _ if dec.funct3 >= 0b001 && dec.funct3 <= 0b111 => {
                    // CSR instructions
                    dec.decode_i();
                    let csr_addr = (dec.imm & 0xfff) as u16;
                    let mut csr_val = super::system::csr::isa_csr_read(&cpu, csr_addr);
                    
                    if csr_addr == crate::isa::riscv32::system::csr::CSR_MIP {
                         csr_val |= crate::device::clint::get_mip_status();
                    }
                    
                    let new_val = match dec.funct3 {
                        0b001 => {  // CSRRW
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, rs1_val);
                            csr_val
                        }
                        0b010 => {  // CSRRS
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val | rs1_val);
                            csr_val
                        }
                        0b011 => {  // CSRRC
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val & !rs1_val);
                            csr_val
                        }
                        0b101 => {  // CSRRWI
                            let zimm = dec.rs1 as u32;
                            super::system::csr::isa_csr_write(cpu, csr_addr, zimm);
                            csr_val
                        }
                        0b110 => {  // CSRRSI
                            let zimm = dec.rs1 as u32;
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val | zimm);
                            csr_val
                        }
                        0b111 => {  // CSRRCI
                            let zimm = dec.rs1 as u32;
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val & !zimm);
                            csr_val
                        }
                        _ => csr_val,
                    };
                    W!(cpu, dec.rd, new_val);
                }
                _ => {
                    log::error!("Invalid system instruction: 0x{:08x}", inst);
                }
            }
        }
        _ => {
            log::error!("Invalid instruction: 0x{:08x} at PC=0x{:08x}", inst, pc);
            set_state(RemuState::Abort);
            return;
        }
    }
    
    // Update PC
    cpu.pc = dnpc;
    
    // Ensure x0 remains 0
    cpu.gpr[0] = 0;
}

// Multiplication helpers (from REMU)
fn mulhu(a: u32, b: u32) -> u32 {
    let t = (a as u64) * (b as u64);
    (t >> 32) as u32
}

fn mulh(a: i32, b: i32) -> i32 {
    let negate = (a < 0) != (b < 0);
    let res = mulhu(a.abs() as u32, b.abs() as u32);
    if negate {
        (!res).wrapping_add((a as i64 * b as i64 == 0) as u32) as i32
    } else {
        res as i32
    }
}

fn mulhsu(a: i32, b: u32) -> u32 {
    let negate = a < 0;
    let res = mulhu(a.abs() as u32, b);
    if negate {
        (!res).wrapping_add((a as i64 * b as i64 == 0) as u32)
    } else {
        res
    }
}
