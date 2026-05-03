// RISC-V instruction execution

use super::decode::DecodedInst;
use crate::common::{as_signed_xlen, mask_xlen, sext32, sign_extend, RemuState, Word};
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

fn read_or_trap(
    cpu: &mut crate::cpu::state::CpuState,
    addr: Word,
    len: usize,
    pc: Word,
) -> Option<Word> {
    match vaddr_read(&*cpu, addr, len) {
        Ok(data) => Some(data),
        Err(cause) => {
            raise_mem_trap(cpu, cause, pc, addr);
            None
        }
    }
}

fn write_or_trap(
    cpu: &mut crate::cpu::state::CpuState,
    addr: Word,
    len: usize,
    data: Word,
    pc: Word,
) -> bool {
    match vaddr_write(&*cpu, addr, len, data) {
        Ok(()) => true,
        Err(cause) => {
            raise_mem_trap(cpu, cause, pc, addr);
            false
        }
    }
}

fn raise_mem_trap(cpu: &mut crate::cpu::state::CpuState, cause: Word, pc: Word, addr: Word) {
    let new_pc = super::system::intr::isa_raise_intr_with_tval(cpu, cause, pc, addr);
    cpu.pc = new_pc;
    cpu.gpr[0] = 0;
}

pub fn decode_exec(cpu: &mut crate::cpu::state::CpuState, inst: Word, pc: Word) {
    if inst & 0b11 != 0b11 {
        exec_compressed(cpu, inst as u16, pc);
        return;
    }

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
                0b000 => src1 == src2,                                 // BEQ
                0b001 => src1 != src2,                                 // BNE
                0b100 => as_signed_xlen(src1) < as_signed_xlen(src2),  // BLT
                0b101 => as_signed_xlen(src1) >= as_signed_xlen(src2), // BGE
                0b110 => src1 < src2,                                  // BLTU
                0b111 => src1 >= src2,                                 // BGEU
                _ => {
                    log::error!("Invalid branch funct3: 0b{:03b}", dec.funct3);
                    false
                }
            };
            if taken {
                dnpc = pc.wrapping_add(dec.imm);
            }
        }
        // Floating-point loads used by hard-float RV64 userland.
        0b0000111 => {
            dec.decode_i();
            let addr = R!(cpu, dec.rs1).wrapping_add(dec.imm);
            match dec.funct3 {
                0b010 => {
                    let Some(val) = read_or_trap(cpu, addr, 4, pc) else {
                        return;
                    };
                    cpu.fpr[dec.rd] = val | 0xffff_ffff_0000_0000;
                }
                0b011 => {
                    let Some(val) = read_or_trap(cpu, addr, 8, pc) else {
                        return;
                    };
                    cpu.fpr[dec.rd] = val;
                }
                _ => {
                    log::error!("Invalid floating load funct3: 0b{:03b}", dec.funct3);
                    set_state(RemuState::Abort);
                    return;
                }
            }
        }
        // Load instructions
        0b0000011 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            let addr = src1.wrapping_add(dec.imm);
            let val = match dec.funct3 {
                0b000 => {
                    // LB
                    let v = match read_or_trap(cpu, addr, 1, pc) {
                        Some(v) => v,
                        None => return,
                    };
                    sign_extend(v & 0xff, 8)
                }
                0b001 => {
                    // LH
                    let v = match read_or_trap(cpu, addr, 2, pc) {
                        Some(v) => v,
                        None => return,
                    };
                    sign_extend(v & 0xffff, 16)
                }
                0b010 => match read_or_trap(cpu, addr, 4, pc) {
                    // LW
                    Some(v) => sext32(v as u32),
                    None => return,
                },
                0b100 => match read_or_trap(cpu, addr, 1, pc) {
                    // LBU
                    Some(v) => v,
                    None => return,
                },
                0b101 => match read_or_trap(cpu, addr, 2, pc) {
                    // LHU
                    Some(v) => v,
                    None => return,
                },
                0b011 if crate::generated::config::RV64 => match read_or_trap(cpu, addr, 8, pc) {
                    // LD
                    Some(v) => v,
                    None => return,
                },
                0b110 if crate::generated::config::RV64 => match read_or_trap(cpu, addr, 4, pc) {
                    // LWU
                    Some(v) => v & 0xffff_ffff,
                    None => return,
                },
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
            let completed = match dec.funct3 {
                0b000 => write_or_trap(cpu, addr, 1, src2, pc), // SB
                0b001 => write_or_trap(cpu, addr, 2, src2, pc), // SH
                0b010 => write_or_trap(cpu, addr, 4, src2, pc), // SW
                0b011 if crate::generated::config::RV64 => write_or_trap(cpu, addr, 8, src2, pc), // SD
                _ => {
                    log::error!("Invalid store funct3: 0b{:03b}", dec.funct3);
                    true
                }
            };
            if !completed {
                return;
            }
        }
        // Floating-point stores used by hard-float RV64 userland.
        0b0100111 => {
            dec.decode_s();
            let addr = R!(cpu, dec.rs1).wrapping_add(dec.imm);
            let completed = match dec.funct3 {
                0b010 => write_or_trap(cpu, addr, 4, cpu.fpr[dec.rs2], pc),
                0b011 => write_or_trap(cpu, addr, 8, cpu.fpr[dec.rs2], pc),
                _ => {
                    log::error!("Invalid floating store funct3: 0b{:03b}", dec.funct3);
                    set_state(RemuState::Abort);
                    return;
                }
            };
            if !completed {
                return;
            }
        }
        // I-type ALU instructions
        0b0010011 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            let val = match dec.funct3 {
                0b000 => src1.wrapping_add(dec.imm), // ADDI
                0b010 => (as_signed_xlen(src1) < as_signed_xlen(dec.imm)) as Word, // SLTI
                0b011 => (src1 < mask_xlen(dec.imm)) as Word, // SLTIU
                0b100 => src1 ^ dec.imm,             // XORI
                0b110 => src1 | dec.imm,             // ORI
                0b111 => src1 & dec.imm,             // ANDI
                0b001 => {
                    // SLLI
                    let shamt = shift_amount(dec.imm);
                    src1 << shamt
                }
                0b101 => {
                    // SRLI / SRAI
                    let shamt = shift_amount(dec.imm);
                    if (dec.imm >> 10) & 1 == 1 {
                        // SRAI
                        (as_signed_xlen(src1) >> shamt) as Word
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
        // RV64 I-type ALU instructions producing 32-bit sign-extended results
        0b0011011 if crate::generated::config::RV64 => {
            dec.decode_i();
            let src1 = R!(cpu, dec.rs1);
            let val = match dec.funct3 {
                0b000 => sext32((src1 as u32).wrapping_add(dec.imm as u32)), // ADDIW
                0b001 => {
                    let shamt = (dec.imm & 0x1f) as u32;
                    sext32((src1 as u32).wrapping_shl(shamt))
                }
                0b101 => {
                    let shamt = (dec.imm & 0x1f) as u32;
                    if (dec.imm >> 10) & 1 == 1 {
                        sext32(((src1 as u32 as i32) >> shamt) as u32) // SRAIW
                    } else {
                        sext32((src1 as u32) >> shamt) // SRLIW
                    }
                }
                _ => {
                    log::error!("Invalid RV64 OP-IMM-32 funct3: 0b{:03b}", dec.funct3);
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
                (0b0000000, 0b000) => src1.wrapping_add(src2), // ADD
                (0b0100000, 0b000) => src1.wrapping_sub(src2), // SUB
                (0b0000000, 0b001) => src1 << shift_amount(src2), // SLL
                (0b0000000, 0b010) => (as_signed_xlen(src1) < as_signed_xlen(src2)) as Word, // SLT
                (0b0000000, 0b011) => (src1 < src2) as Word,   // SLTU
                (0b0000000, 0b100) => src1 ^ src2,             // XOR
                (0b0000000, 0b101) => src1 >> shift_amount(src2), // SRL
                (0b0100000, 0b101) => (as_signed_xlen(src1) >> shift_amount(src2)) as Word, // SRA
                (0b0000000, 0b110) => src1 | src2,             // OR
                (0b0000000, 0b111) => src1 & src2,             // AND
                // M extension
                (0b0000001, 0b000) => src1.wrapping_mul(src2), // MUL
                (0b0000001, 0b001) => mulh_xlen(src1, src2),   // MULH
                (0b0000001, 0b010) => mulhsu_xlen(src1, src2), // MULHSU
                (0b0000001, 0b011) => mulhu_xlen(src1, src2),  // MULHU
                (0b0000001, 0b100) => div_xlen(src1, src2),    // DIV
                (0b0000001, 0b101) => divu_xlen(src1, src2),   // DIVU
                (0b0000001, 0b110) => rem_xlen(src1, src2),    // REM
                (0b0000001, 0b111) => remu_xlen(src1, src2),   // REMU
                _ => {
                    log::error!(
                        "Invalid R-type funct7/funct3: 0b{:07b}/0b{:03b}",
                        dec.funct7,
                        dec.funct3
                    );
                    0
                }
            };
            W!(cpu, dec.rd, val);
        }
        // RV64 R-type ALU instructions producing 32-bit sign-extended results
        0b0111011 if crate::generated::config::RV64 => {
            dec.decode_r();
            let src1 = R!(cpu, dec.rs1);
            let src2 = R!(cpu, dec.rs2);
            let shamt = (src2 & 0x1f) as u32;
            let val = match (dec.funct7, dec.funct3) {
                (0b0000000, 0b000) => sext32((src1 as u32).wrapping_add(src2 as u32)), // ADDW
                (0b0100000, 0b000) => sext32((src1 as u32).wrapping_sub(src2 as u32)), // SUBW
                (0b0000000, 0b001) => sext32((src1 as u32).wrapping_shl(shamt)),       // SLLW
                (0b0000000, 0b101) => sext32((src1 as u32) >> shamt),                  // SRLW
                (0b0100000, 0b101) => sext32(((src1 as u32 as i32) >> shamt) as u32),  // SRAW
                (0b0000001, 0b000) => sext32((src1 as u32).wrapping_mul(src2 as u32)), // MULW
                (0b0000001, 0b100) => divw(src1, src2),                                // DIVW
                (0b0000001, 0b101) => divuw(src1, src2),                               // DIVUW
                (0b0000001, 0b110) => remw(src1, src2),                                // REMW
                (0b0000001, 0b111) => remuw(src1, src2),                               // REMUW
                _ => {
                    log::error!(
                        "Invalid RV64 OP-32 funct7/funct3: 0b{:07b}/0b{:03b}",
                        dec.funct7,
                        dec.funct3
                    );
                    0
                }
            };
            W!(cpu, dec.rd, val);
        }
        // A extension atomic instructions
        0b0101111 => {
            dec.decode_r();
            let src1 = R!(cpu, dec.rs1);
            let addr = src1;
            let len = match dec.funct3 {
                0b010 => 4,
                0b011 if crate::generated::config::RV64 => 8,
                _ => {
                    log::error!(
                        "Invalid atomic width: funct7={:07b}, funct3={:03b}",
                        dec.funct7,
                        dec.funct3
                    );
                    0
                }
            };
            if len == 0 {
                set_state(RemuState::Abort);
                return;
            }

            match dec.funct7 >> 2 {
                0b00010 => {
                    // LR.W/LR.D
                    let val = match read_or_trap(cpu, addr, len, pc) {
                        Some(v) => v,
                        None => return,
                    };
                    W!(cpu, dec.rd, amo_load_result(val, len));
                }
                0b00011 => {
                    // SC.W/SC.D; REMU currently models reservations as always successful.
                    let src2 = R!(cpu, dec.rs2);
                    if !write_or_trap(cpu, addr, len, amo_store_value(src2, len), pc) {
                        return;
                    }
                    W!(cpu, dec.rd, 0);
                }
                op @ (0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000 | 0b10100
                | 0b11000 | 0b11100) => {
                    let old = match read_or_trap(cpu, addr, len, pc) {
                        Some(v) => v,
                        None => return,
                    };
                    let src2 = R!(cpu, dec.rs2);
                    let new_value = match op {
                        0b00001 => amo_store_value(src2, len), // AMOSWAP
                        0b00000 => {
                            amo_store_value(old, len).wrapping_add(amo_store_value(src2, len))
                        } // AMOADD
                        0b00100 => amo_store_value(old, len) ^ amo_store_value(src2, len), // AMOXOR
                        0b01100 => amo_store_value(old, len) & amo_store_value(src2, len), // AMOAND
                        0b01000 => amo_store_value(old, len) | amo_store_value(src2, len), // AMOOR
                        0b10000 => {
                            if amo_signed(old, len) < amo_signed(src2, len) {
                                amo_store_value(old, len)
                            } else {
                                amo_store_value(src2, len)
                            }
                        } // AMOMIN
                        0b10100 => {
                            if amo_signed(old, len) > amo_signed(src2, len) {
                                amo_store_value(old, len)
                            } else {
                                amo_store_value(src2, len)
                            }
                        } // AMOMAX
                        0b11000 => {
                            if amo_store_value(old, len) < amo_store_value(src2, len) {
                                amo_store_value(old, len)
                            } else {
                                amo_store_value(src2, len)
                            }
                        } // AMOMINU
                        0b11100 => {
                            if amo_store_value(old, len) > amo_store_value(src2, len) {
                                amo_store_value(old, len)
                            } else {
                                amo_store_value(src2, len)
                            }
                        } // AMOMAXU
                        _ => unreachable!(),
                    };
                    if !write_or_trap(cpu, addr, len, new_value, pc) {
                        return;
                    }
                    W!(cpu, dec.rd, amo_load_result(old, len));
                }
                _ => {
                    log::error!(
                        "Invalid atomic instruction: funct7={:07b}, funct3={:03b}",
                        dec.funct7,
                        dec.funct3
                    );
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
            if dec.funct7 == 0b0001001 && dec.funct3 == 0 {
                super::system::mmu::flush_tlb();
                cpu.pc = dnpc;
                cpu.gpr[0] = 0;
                return;
            }

            match (dec.funct7, dec.rs2, dec.funct3) {
                (0b0000000, 0b00000, 0b000) => {
                    // ECALL
                    // Determine mode for ECALL cause (User=8, Supervisor=9, Machine=11)
                    let cause = match cpu.mode {
                        crate::common::PrivMode::Machine => 11,
                        crate::common::PrivMode::Supervisor => 9,
                        crate::common::PrivMode::User => 8,
                    };
                    crate::utils::ecall_trace::trace_ecall(
                        pc,
                        cause,
                        cpu.mode as u8,
                        [
                            R!(cpu, 10),
                            R!(cpu, 11),
                            R!(cpu, 12),
                            R!(cpu, 13),
                            R!(cpu, 14),
                            R!(cpu, 15),
                            R!(cpu, 16),
                            R!(cpu, 17),
                        ],
                    );
                    let new_pc = super::system::intr::isa_raise_intr(cpu, cause, pc);
                    cpu.pc = new_pc;
                    return;
                }
                (0b0000000, 0b00001, 0b000) => {
                    // EBREAK
                    // EBREAK cause = 3
                    let new_pc = super::system::intr::isa_raise_intr(cpu, 3, pc);
                    cpu.pc = new_pc;
                    return;
                }
                (0b0011000, 0b00010, 0b000) => {
                    // MRET
                    let mstatus =
                        super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_MSTATUS);
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

                    super::system::csr::isa_csr_write(
                        cpu,
                        super::system::csr::CSR_MSTATUS,
                        new_mstatus,
                    );

                    cpu.mode = match mpp {
                        3 => crate::common::PrivMode::Machine,
                        1 => crate::common::PrivMode::Supervisor,
                        _ => crate::common::PrivMode::User,
                    };

                    cpu.pc = mepc;
                    // dnpc not needed as we update cpu.pc directly and loop continues unless we return?
                    // decode_exec updates cpu.pc = dnpc at end.
                    // We should just return early after setting cpu.pc
                    return;
                }
                (0b0001000, 0b00010, 0b000) => {
                    // SRET
                    // Similar to MRET but for Supervisor
                    let sstatus =
                        super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_SSTATUS); // actually accesses MSTATUS
                    let sepc = super::system::csr::isa_csr_read(&cpu, super::system::csr::CSR_SEPC);

                    // Restore SIE = SPIE
                    let spie = (sstatus >> 5) & 1;
                    let spp = (sstatus >> 8) & 1;

                    // SIE(1) <- SPIE(5)
                    let mut new_sstatus = (sstatus & !(1 << 1)) | (spie << 1);
                    new_sstatus |= 1 << 5; // SPIE = 1
                    new_sstatus &= !(1 << 8); // SPP = 0 (User)

                    // Need to write back to MSTATUS (handled by set_csr SSTATUS alias)
                    super::system::csr::isa_csr_write(
                        cpu,
                        super::system::csr::CSR_SSTATUS,
                        new_sstatus,
                    );

                    cpu.mode = match spp {
                        1 => crate::common::PrivMode::Supervisor,
                        _ => crate::common::PrivMode::User,
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
                        0b001 => {
                            // CSRRW
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, rs1_val);
                            csr_val
                        }
                        0b010 => {
                            // CSRRS
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val | rs1_val);
                            csr_val
                        }
                        0b011 => {
                            // CSRRC
                            let rs1_val = R!(cpu, dec.rs1);
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val & !rs1_val);
                            csr_val
                        }
                        0b101 => {
                            // CSRRWI
                            let zimm = dec.rs1 as Word;
                            super::system::csr::isa_csr_write(cpu, csr_addr, zimm);
                            csr_val
                        }
                        0b110 => {
                            // CSRRSI
                            let zimm = dec.rs1 as Word;
                            super::system::csr::isa_csr_write(cpu, csr_addr, csr_val | zimm);
                            csr_val
                        }
                        0b111 => {
                            // CSRRCI
                            let zimm = dec.rs1 as Word;
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

#[inline]
fn c_bits(inst: u16, hi: u32, lo: u32) -> Word {
    ((inst as Word) >> lo) & ((1_u64 << (hi - lo + 1)) - 1)
}

#[inline]
fn c_bit(inst: u16, bit: u32) -> Word {
    c_bits(inst, bit, bit)
}

#[inline]
fn c_reg3(inst: u16, lo: u32) -> usize {
    (8 + c_bits(inst, lo + 2, lo)) as usize
}

#[inline]
fn c_rd(inst: u16) -> usize {
    c_bits(inst, 11, 7) as usize
}

#[inline]
fn c_rs2(inst: u16) -> usize {
    c_bits(inst, 6, 2) as usize
}

#[inline]
fn c_imm6(inst: u16) -> Word {
    sign_extend((c_bit(inst, 12) << 5) | c_bits(inst, 6, 2), 6)
}

#[inline]
fn c_shamt(inst: u16) -> u32 {
    ((c_bit(inst, 12) << 5) | c_bits(inst, 6, 2)) as u32
}

fn finish_compressed(cpu: &mut crate::cpu::state::CpuState, dnpc: Word) {
    cpu.pc = dnpc;
    cpu.gpr[0] = 0;
}

fn invalid_compressed(inst: u16, pc: Word) {
    log::error!(
        "Invalid compressed instruction: 0x{:04x} at PC=0x{:08x}",
        inst,
        pc
    );
    set_state(RemuState::Abort);
}

fn exec_compressed(cpu: &mut crate::cpu::state::CpuState, inst: u16, pc: Word) {
    let quadrant = inst & 0b11;
    let funct3 = (inst >> 13) & 0b111;
    let next_pc = pc.wrapping_add(2);

    match (quadrant, funct3) {
        // C.ADDI4SPN
        (0b00, 0b000) => {
            let rd = c_reg3(inst, 2);
            let imm = (c_bits(inst, 10, 7) << 6)
                | (c_bits(inst, 12, 11) << 4)
                | (c_bit(inst, 5) << 3)
                | (c_bit(inst, 6) << 2);
            if imm == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            W!(cpu, rd, R!(cpu, 2).wrapping_add(imm));
            finish_compressed(cpu, next_pc);
        }
        // C.FLD
        (0b00, 0b001) if crate::generated::config::RV64 => {
            let rd = c_reg3(inst, 2);
            let rs1 = c_reg3(inst, 7);
            let imm = (c_bits(inst, 6, 5) << 6) | (c_bits(inst, 12, 10) << 3);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 8, pc) else {
                return;
            };
            cpu.fpr[rd] = val;
            finish_compressed(cpu, next_pc);
        }
        // C.LW
        (0b00, 0b010) => {
            let rd = c_reg3(inst, 2);
            let rs1 = c_reg3(inst, 7);
            let imm = (c_bit(inst, 5) << 6) | (c_bits(inst, 12, 10) << 3) | (c_bit(inst, 6) << 2);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 4, pc) else {
                return;
            };
            W!(cpu, rd, sext32(val as u32));
            finish_compressed(cpu, next_pc);
        }
        // C.LD
        (0b00, 0b011) if crate::generated::config::RV64 => {
            let rd = c_reg3(inst, 2);
            let rs1 = c_reg3(inst, 7);
            let imm = (c_bits(inst, 6, 5) << 6) | (c_bits(inst, 12, 10) << 3);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 8, pc) else {
                return;
            };
            W!(cpu, rd, val);
            finish_compressed(cpu, next_pc);
        }
        // C.FSD
        (0b00, 0b101) if crate::generated::config::RV64 => {
            let rs1 = c_reg3(inst, 7);
            let rs2 = c_reg3(inst, 2);
            let imm = (c_bits(inst, 6, 5) << 6) | (c_bits(inst, 12, 10) << 3);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 8, cpu.fpr[rs2], pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.SW
        (0b00, 0b110) => {
            let rs1 = c_reg3(inst, 7);
            let rs2 = c_reg3(inst, 2);
            let imm = (c_bit(inst, 5) << 6) | (c_bits(inst, 12, 10) << 3) | (c_bit(inst, 6) << 2);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 4, R!(cpu, rs2), pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.SD
        (0b00, 0b111) if crate::generated::config::RV64 => {
            let rs1 = c_reg3(inst, 7);
            let rs2 = c_reg3(inst, 2);
            let imm = (c_bits(inst, 6, 5) << 6) | (c_bits(inst, 12, 10) << 3);
            let addr = R!(cpu, rs1).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 8, R!(cpu, rs2), pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.ADDI / C.NOP
        (0b01, 0b000) => {
            let rd = c_rd(inst);
            W!(cpu, rd, R!(cpu, rd).wrapping_add(c_imm6(inst)));
            finish_compressed(cpu, next_pc);
        }
        // RV64 C.ADDIW, RV32 C.JAL
        (0b01, 0b001) if crate::generated::config::RV64 => {
            let rd = c_rd(inst);
            if rd == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            W!(
                cpu,
                rd,
                sext32((R!(cpu, rd).wrapping_add(c_imm6(inst))) as u32)
            );
            finish_compressed(cpu, next_pc);
        }
        (0b01, 0b001) => {
            let target = pc.wrapping_add(c_j_imm(inst));
            W!(cpu, 1, next_pc);
            crate::utils::ftrace::trace_call(pc, target);
            finish_compressed(cpu, target);
        }
        // C.LI
        (0b01, 0b010) => {
            let rd = c_rd(inst);
            if rd == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            W!(cpu, rd, c_imm6(inst));
            finish_compressed(cpu, next_pc);
        }
        // C.ADDI16SP / C.LUI
        (0b01, 0b011) => {
            let rd = c_rd(inst);
            if rd == 2 {
                let imm = sign_extend(
                    (c_bit(inst, 12) << 9)
                        | (c_bits(inst, 4, 3) << 7)
                        | (c_bit(inst, 5) << 6)
                        | (c_bit(inst, 2) << 5)
                        | (c_bit(inst, 6) << 4),
                    10,
                );
                if imm == 0 {
                    invalid_compressed(inst, pc);
                    return;
                }
                W!(cpu, 2, R!(cpu, 2).wrapping_add(imm));
            } else if rd != 0 {
                let imm = sign_extend((c_bit(inst, 12) << 5) | c_bits(inst, 6, 2), 6) << 12;
                if imm == 0 {
                    invalid_compressed(inst, pc);
                    return;
                }
                W!(cpu, rd, imm);
            } else {
                invalid_compressed(inst, pc);
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.SRLI/C.SRAI/C.ANDI/C.SUB/C.XOR/C.OR/C.AND/C.SUBW/C.ADDW
        (0b01, 0b100) => {
            let op = (inst >> 10) & 0b11;
            let rd = c_reg3(inst, 7);
            match op {
                0b00 => W!(cpu, rd, R!(cpu, rd) >> shift_amount(c_shamt(inst) as Word)),
                0b01 => W!(
                    cpu,
                    rd,
                    (as_signed_xlen(R!(cpu, rd)) >> c_shamt(inst)) as Word
                ),
                0b10 => W!(cpu, rd, R!(cpu, rd) & c_imm6(inst)),
                0b11 => {
                    let rs2 = c_reg3(inst, 2);
                    let wide = c_bit(inst, 12) != 0;
                    let val = match (wide, (inst >> 5) & 0b11) {
                        (false, 0b00) => R!(cpu, rd).wrapping_sub(R!(cpu, rs2)),
                        (false, 0b01) => R!(cpu, rd) ^ R!(cpu, rs2),
                        (false, 0b10) => R!(cpu, rd) | R!(cpu, rs2),
                        (false, 0b11) => R!(cpu, rd) & R!(cpu, rs2),
                        (true, 0b00) if crate::generated::config::RV64 => {
                            sext32((R!(cpu, rd) as u32).wrapping_sub(R!(cpu, rs2) as u32))
                        }
                        (true, 0b01) if crate::generated::config::RV64 => {
                            sext32((R!(cpu, rd) as u32).wrapping_add(R!(cpu, rs2) as u32))
                        }
                        _ => {
                            invalid_compressed(inst, pc);
                            return;
                        }
                    };
                    W!(cpu, rd, val);
                }
                _ => unreachable!(),
            }
            finish_compressed(cpu, next_pc);
        }
        // C.J
        (0b01, 0b101) => {
            finish_compressed(cpu, pc.wrapping_add(c_j_imm(inst)));
        }
        // C.BEQZ
        (0b01, 0b110) => {
            let rs1 = c_reg3(inst, 7);
            let target = pc.wrapping_add(c_b_imm(inst));
            finish_compressed(cpu, if R!(cpu, rs1) == 0 { target } else { next_pc });
        }
        // C.BNEZ
        (0b01, 0b111) => {
            let rs1 = c_reg3(inst, 7);
            let target = pc.wrapping_add(c_b_imm(inst));
            finish_compressed(cpu, if R!(cpu, rs1) != 0 { target } else { next_pc });
        }
        // C.SLLI
        (0b10, 0b000) => {
            let rd = c_rd(inst);
            if rd == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            W!(cpu, rd, R!(cpu, rd) << shift_amount(c_shamt(inst) as Word));
            finish_compressed(cpu, next_pc);
        }
        // C.FLDSP
        (0b10, 0b001) if crate::generated::config::RV64 => {
            let rd = c_rd(inst);
            let imm =
                (c_bit(inst, 12) << 5) | (c_bits(inst, 6, 5) << 3) | (c_bits(inst, 4, 2) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 8, pc) else {
                return;
            };
            cpu.fpr[rd] = val;
            finish_compressed(cpu, next_pc);
        }
        // C.LWSP
        (0b10, 0b010) => {
            let rd = c_rd(inst);
            if rd == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            let imm =
                (c_bit(inst, 12) << 5) | (c_bits(inst, 6, 4) << 2) | (c_bits(inst, 3, 2) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 4, pc) else {
                return;
            };
            W!(cpu, rd, sext32(val as u32));
            finish_compressed(cpu, next_pc);
        }
        // C.LDSP
        (0b10, 0b011) if crate::generated::config::RV64 => {
            let rd = c_rd(inst);
            if rd == 0 {
                invalid_compressed(inst, pc);
                return;
            }
            let imm =
                (c_bit(inst, 12) << 5) | (c_bits(inst, 6, 5) << 3) | (c_bits(inst, 4, 2) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            let Some(val) = read_or_trap(cpu, addr, 8, pc) else {
                return;
            };
            W!(cpu, rd, val);
            finish_compressed(cpu, next_pc);
        }
        // C.JR/C.MV/C.EBREAK/C.JALR/C.ADD
        (0b10, 0b100) => {
            let rd = c_rd(inst);
            let rs2 = c_rs2(inst);
            if c_bit(inst, 12) == 0 {
                if rs2 == 0 {
                    if rd == 0 {
                        invalid_compressed(inst, pc);
                        return;
                    }
                    let target = R!(cpu, rd) & !1;
                    if rd == 1 {
                        crate::utils::ftrace::trace_ret(pc);
                    }
                    finish_compressed(cpu, target);
                } else {
                    if rd == 0 {
                        invalid_compressed(inst, pc);
                        return;
                    }
                    W!(cpu, rd, R!(cpu, rs2));
                    finish_compressed(cpu, next_pc);
                }
            } else if rs2 == 0 {
                if rd == 0 {
                    let new_pc = super::system::intr::isa_raise_intr(cpu, 3, pc);
                    cpu.pc = new_pc;
                    cpu.gpr[0] = 0;
                } else {
                    let target = R!(cpu, rd) & !1;
                    W!(cpu, 1, next_pc);
                    crate::utils::ftrace::trace_call(pc, target);
                    finish_compressed(cpu, target);
                }
            } else {
                if rd == 0 {
                    invalid_compressed(inst, pc);
                    return;
                }
                W!(cpu, rd, R!(cpu, rd).wrapping_add(R!(cpu, rs2)));
                finish_compressed(cpu, next_pc);
            }
        }
        // C.FSDSP
        (0b10, 0b101) if crate::generated::config::RV64 => {
            let rs2 = c_rs2(inst);
            let imm = (c_bits(inst, 12, 10) << 3) | (c_bits(inst, 9, 7) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 8, cpu.fpr[rs2], pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.SWSP
        (0b10, 0b110) => {
            let rs2 = c_rs2(inst);
            let imm = (c_bits(inst, 12, 9) << 2) | (c_bits(inst, 8, 7) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 4, R!(cpu, rs2), pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        // C.SDSP
        (0b10, 0b111) if crate::generated::config::RV64 => {
            let rs2 = c_rs2(inst);
            let imm = (c_bits(inst, 12, 10) << 3) | (c_bits(inst, 9, 7) << 6);
            let addr = R!(cpu, 2).wrapping_add(imm);
            if !write_or_trap(cpu, addr, 8, R!(cpu, rs2), pc) {
                return;
            }
            finish_compressed(cpu, next_pc);
        }
        _ => invalid_compressed(inst, pc),
    }
}

fn c_j_imm(inst: u16) -> Word {
    sign_extend(
        (c_bit(inst, 12) << 11)
            | (c_bit(inst, 11) << 4)
            | (c_bits(inst, 10, 9) << 8)
            | (c_bit(inst, 8) << 10)
            | (c_bit(inst, 7) << 6)
            | (c_bit(inst, 6) << 7)
            | (c_bits(inst, 5, 3) << 1)
            | (c_bit(inst, 2) << 5),
        12,
    )
}

fn c_b_imm(inst: u16) -> Word {
    sign_extend(
        (c_bit(inst, 12) << 8)
            | (c_bits(inst, 11, 10) << 3)
            | (c_bits(inst, 6, 5) << 6)
            | (c_bits(inst, 4, 3) << 1)
            | (c_bit(inst, 2) << 5),
        9,
    )
}

fn shift_amount(value: Word) -> u32 {
    if crate::generated::config::RV64 {
        (value & 0x3f) as u32
    } else {
        (value & 0x1f) as u32
    }
}

fn xlen_width() -> u32 {
    if crate::generated::config::RV64 {
        64
    } else {
        32
    }
}

fn unsigned_xlen(value: Word) -> u128 {
    mask_xlen(value) as u128
}

fn signed_xlen_i128(value: Word) -> i128 {
    if crate::generated::config::RV64 {
        value as i64 as i128
    } else {
        value as u32 as i32 as i128
    }
}

fn mulhu_xlen(a: Word, b: Word) -> Word {
    let width = xlen_width();
    ((unsigned_xlen(a) * unsigned_xlen(b)) >> width) as Word
}

fn mulh_xlen(a: Word, b: Word) -> Word {
    let width = xlen_width();
    ((signed_xlen_i128(a) * signed_xlen_i128(b)) >> width) as Word
}

fn mulhsu_xlen(a: Word, b: Word) -> Word {
    let width = xlen_width();
    ((signed_xlen_i128(a) * unsigned_xlen(b) as i128) >> width) as Word
}

fn div_xlen(a: Word, b: Word) -> Word {
    if mask_xlen(b) == 0 {
        return mask_xlen(u64::MAX);
    }
    let lhs = signed_xlen_i128(a);
    let rhs = signed_xlen_i128(b);
    let min = if crate::generated::config::RV64 {
        i64::MIN as i128
    } else {
        i32::MIN as i128
    };
    if lhs == min && rhs == -1 {
        return mask_xlen(a);
    }
    (lhs / rhs) as Word
}

fn divu_xlen(a: Word, b: Word) -> Word {
    let rhs = unsigned_xlen(b);
    if rhs == 0 {
        return mask_xlen(u64::MAX);
    }
    (unsigned_xlen(a) / rhs) as Word
}

fn rem_xlen(a: Word, b: Word) -> Word {
    if mask_xlen(b) == 0 {
        return mask_xlen(a);
    }
    let lhs = signed_xlen_i128(a);
    let rhs = signed_xlen_i128(b);
    let min = if crate::generated::config::RV64 {
        i64::MIN as i128
    } else {
        i32::MIN as i128
    };
    if lhs == min && rhs == -1 {
        return 0;
    }
    (lhs % rhs) as Word
}

fn remu_xlen(a: Word, b: Word) -> Word {
    let rhs = unsigned_xlen(b);
    if rhs == 0 {
        return mask_xlen(a);
    }
    (unsigned_xlen(a) % rhs) as Word
}

fn divw(a: Word, b: Word) -> Word {
    let lhs = a as u32 as i32;
    let rhs = b as u32 as i32;
    if rhs == 0 {
        return sext32(u32::MAX);
    }
    if lhs == i32::MIN && rhs == -1 {
        return sext32(lhs as u32);
    }
    sext32(lhs.wrapping_div(rhs) as u32)
}

fn divuw(a: Word, b: Word) -> Word {
    let lhs = a as u32;
    let rhs = b as u32;
    if rhs == 0 {
        return sext32(u32::MAX);
    }
    sext32(lhs.wrapping_div(rhs))
}

fn remw(a: Word, b: Word) -> Word {
    let lhs = a as u32 as i32;
    let rhs = b as u32 as i32;
    if rhs == 0 {
        return sext32(lhs as u32);
    }
    if lhs == i32::MIN && rhs == -1 {
        return 0;
    }
    sext32(lhs.wrapping_rem(rhs) as u32)
}

fn remuw(a: Word, b: Word) -> Word {
    let lhs = a as u32;
    let rhs = b as u32;
    if rhs == 0 {
        return sext32(lhs);
    }
    sext32(lhs.wrapping_rem(rhs))
}

fn amo_store_value(value: Word, len: usize) -> Word {
    if len == 4 {
        value & 0xffff_ffff
    } else {
        value
    }
}

fn amo_load_result(value: Word, len: usize) -> Word {
    if len == 4 {
        sext32(value as u32)
    } else {
        value
    }
}

fn amo_signed(value: Word, len: usize) -> i64 {
    if len == 4 {
        value as u32 as i32 as i64
    } else {
        value as i64
    }
}
