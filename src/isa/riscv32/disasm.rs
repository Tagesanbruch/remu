// RISC-V32 Disassembler
// Lightweight implementation for RV32IMA instruction set

use crate::common::Word;

/// Register ABI names
const REG_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

#[inline]
fn reg_name(reg: u32) -> &'static str {
    REG_NAMES[(reg & 0x1f) as usize]
}

/// Disassemble a single RISC-V32 instruction
pub fn disasm(inst: Word, _pc: Word) -> String {
    let opcode = inst & 0x7f;
    let rd = (inst >> 7) & 0x1f;
    let funct3 = (inst >> 12) & 0x7;
    let rs1 = (inst >> 15) & 0x1f;
    let rs2 = (inst >> 20) & 0x1f;
    let funct7 = (inst >> 25) & 0x7f;

    // Extract immediates with sign extension
    let imm_i = ((inst as i32) >> 20) as i32;
    let imm_s = (((inst >> 25) << 5) | ((inst >> 7) & 0x1f)) as i32;
    let imm_s = (imm_s << 20) >> 20; // Sign extend
    let imm_b = ((((inst >> 31) & 1) << 12) |
                 (((inst >> 7) & 1) << 11) |
                 (((inst >> 25) & 0x3f) << 5) |
                 (((inst >> 8) & 0xf) << 1)) as i32;
    let imm_b = (imm_b << 19) >> 19; // Sign extend
    let imm_u = (inst & 0xfffff000) as i32;
    let imm_j = ((((inst >> 31) & 1) << 20) |
                 (((inst >> 12) & 0xff) << 12) |
                 (((inst >> 20) & 1) << 11) |
                 (((inst >> 21) & 0x3ff) << 1)) as i32;
    let imm_j = (imm_j << 11) >> 11; // Sign extend

    match opcode {
        0b0110111 => format!("lui\t{}, {:#x}", reg_name(rd), imm_u >> 12),
        0b0010111 => format!("auipc\t{}, {:#x}", reg_name(rd), imm_u >> 12),
        0b1101111 => format!("jal\t{}, {:#x}", reg_name(rd), imm_j),
        0b1100111 => {
            if funct3 == 0 {
                if rd == 0 && rs1 == 1 && imm_i == 0 {
                    "ret".to_string()
                } else if rd == 0 {
                    format!("jr\t{}, {:#x}({})", reg_name(rd), imm_i, reg_name(rs1))
                } else {
                    format!("jalr\t{}, {:#x}({})", reg_name(rd), imm_i, reg_name(rs1))
                }
            } else {
                format!("unknown")
            }
        }
        0b1100011 => {
            let mnem = match funct3 {
                0b000 => "beq",
                0b001 => "bne",
                0b100 => "blt",
                0b101 => "bge",
                0b110 => "bltu",
                0b111 => "bgeu",
                _ => "unknown",
            };
            format!("{}\t{}, {}, {:#x}", mnem, reg_name(rs1), reg_name(rs2), imm_b)
        }
        0b0000011 => {
            let mnem = match funct3 {
                0b000 => "lb",
                0b001 => "lh",
                0b010 => "lw",
                0b100 => "lbu",
                0b101 => "lhu",
                _ => "unknown",
            };
            format!("{}\t{}, {:#x}({})", mnem, reg_name(rd), imm_i, reg_name(rs1))
        }
        0b0100011 => {
            let mnem = match funct3 {
                0b000 => "sb",
                0b001 => "sh",
                0b010 => "sw",
                _ => "unknown",
            };
            format!("{}\t{}, {:#x}({})", mnem, reg_name(rs2), imm_s, reg_name(rs1))
        }
        0b0010011 => {
            let shamt = rs2;
            match funct3 {
                0b000 => format!("addi\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b010 => format!("slti\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b011 => format!("sltiu\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b100 => format!("xori\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b110 => format!("ori\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b111 => format!("andi\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), imm_i),
                0b001 => format!("slli\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), shamt),
                0b101 => {
                    if funct7 == 0 {
                        format!("srli\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), shamt)
                    } else {
                        format!("srai\t{}, {}, {:#x}", reg_name(rd), reg_name(rs1), shamt)
                    }
                }
                _ => "unknown".to_string(),
            }
        }
        0b0110011 => {
            let mnem = match (funct7, funct3) {
                (0b0000000, 0b000) => "add",
                (0b0100000, 0b000) => "sub",
                (0b0000000, 0b001) => "sll",
                (0b0000000, 0b010) => "slt",
                (0b0000000, 0b011) => "sltu",
                (0b0000000, 0b100) => "xor",
                (0b0000000, 0b101) => "srl",
                (0b0100000, 0b101) => "sra",
                (0b0000000, 0b110) => "or",
                (0b0000000, 0b111) => "and",
                // RV32M
                (0b0000001, 0b000) => "mul",
                (0b0000001, 0b001) => "mulh",
                (0b0000001, 0b010) => "mulhsu",
                (0b0000001, 0b011) => "mulhu",
                (0b0000001, 0b100) => "div",
                (0b0000001, 0b101) => "divu",
                (0b0000001, 0b110) => "rem",
                (0b0000001, 0b111) => "remu",
                _ => "unknown",
            };
            format!("{}\t{}, {}, {}", mnem, reg_name(rd), reg_name(rs1), reg_name(rs2))
        }
        0b0001111 => "fence".to_string(),
        0b1110011 => {
            if inst == 0x00000073 {
                "ecall".to_string()
            } else if inst == 0x00100073 {
                "ebreak".to_string()
            } else if inst == 0x30200073 {
                "mret".to_string()
            } else if inst == 0x10200073 {
                "sret".to_string()
            } else if funct7 == 0b0001001 && funct3 == 0 {
                format!("sfence.vma\t{}, {}", reg_name(rs1), reg_name(rs2))
            } else if funct3 >= 1 && funct3 <= 7 {
                let csr = (inst >> 20) & 0xfff;
                let mnem = match funct3 {
                    0b001 => "csrrw",
                    0b010 => "csrrs",
                    0b011 => "csrrc",
                    0b101 => "csrrwi",
                    0b110 => "csrrsi",
                    0b111 => "csrrci",
                    _ => "unknown",
                };
                if funct3 >= 5 {
                    format!("{}\t{}, {:#x}, {:#x}", mnem, reg_name(rd), csr, rs1)
                } else {
                    format!("{}\t{}, {:#x}, {}", mnem, reg_name(rd), csr, reg_name(rs1))
                }
            } else {
                "unknown".to_string()
            }
        }
        0b0101111 => {
            // RV32A
            let funct5 = (funct7 >> 2) & 0x1f;
            let mnem = match funct5 {
                0b00010 => "lr.w",
                0b00011 => "sc.w",
                0b00001 => "amoswap.w",
                0b00000 => "amoadd.w",
                0b00100 => "amoxor.w",
                0b01100 => "amoand.w",
                0b01000 => "amoor.w",
                0b10000 => "amomin.w",
                0b10100 => "amomax.w",
                0b11000 => "amominu.w",
                0b11100 => "amomaxu.w",
                _ => "unknown",
            };
            if funct5 == 0b00010 {
                format!("{}\t{}, ({})", mnem, reg_name(rd), reg_name(rs1))
            } else {
                format!("{}\t{}, {}, ({})", mnem, reg_name(rd), reg_name(rs2), reg_name(rs1))
            }
        }
        _ => format!("unknown {:#x}", inst),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disasm_basic() {
        assert_eq!(disasm(0x00000297, 0), "auipc\tt0, 0x0");
        assert_eq!(disasm(0x00028823, 0), "sb\tzero, 0x10(t0)");
        assert_eq!(disasm(0x0102c503, 0), "lbu\ta0, 0x10(t0)");
        assert_eq!(disasm(0x00100073, 0), "ebreak");
    }
}
