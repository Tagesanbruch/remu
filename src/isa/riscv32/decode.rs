// RISC-V instruction decoder

use crate::common::Word;

#[derive(Debug, Clone, Copy)]
pub enum InstType {
    TypeR,
    TypeI,
    TypeS,
    TypeB,
    TypeU,
    TypeJ,
    TypeN,  // none
}

pub struct DecodedInst {
    pub inst: Word,
    pub opcode: u8,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub funct3: u8,
    pub funct7: u8,
    pub imm: Word,
    pub typ: InstType,
}

#[inline]
fn bits(val: Word, hi: u32, lo: u32) -> Word {
    (val >> lo) & ((1 << (hi - lo + 1)) - 1)
}

#[inline]
fn sext(val: Word, width: u32) -> Word {
    let shift = 32 - width;
    ((val << shift) as i32 >> shift) as u32
}

impl DecodedInst {
    pub fn new(inst: Word) -> Self {
        let opcode = bits(inst, 6, 0) as u8;
        let rd = bits(inst, 11, 7) as usize;
        let rs1 = bits(inst, 19, 15) as usize;
        let rs2 = bits(inst, 24, 20) as usize;
        let funct3 = bits(inst, 14, 12) as u8;
        let funct7 = bits(inst, 31, 25) as u8;

        Self {
            inst,
            opcode,
            rd,
            rs1,
            rs2,
            funct3,
            funct7,
            imm: 0,
            typ: InstType::TypeN,
        }
    }

    pub fn decode_i(&mut self) {
        self.imm = sext(bits(self.inst, 31, 20), 12);
        self.typ = InstType::TypeI;
    }

    pub fn decode_s(&mut self) {
        self.imm = sext((bits(self.inst, 31, 25) << 5) | bits(self.inst, 11, 7), 12);
        self.typ = InstType::TypeS;
    }

    pub fn decode_b(&mut self) {
        self.imm = sext(
            (bits(self.inst, 31, 31) << 12)
                | (bits(self.inst, 7, 7) << 11)
                | (bits(self.inst, 30, 25) << 5)
                | (bits(self.inst, 11, 8) << 1),
            13,
        );
        self.typ = InstType::TypeB;
    }

    pub fn decode_u(&mut self) {
        self.imm = sext(bits(self.inst, 31, 12), 20) << 12;
        self.typ = InstType::TypeU;
    }

    pub fn decode_j(&mut self) {
        self.imm = sext(
            (bits(self.inst, 31, 31) << 20)
                | (bits(self.inst, 19, 12) << 12)
                | (bits(self.inst, 20, 20) << 11)
                | (bits(self.inst, 30, 21) << 1),
            21,
        );
        self.typ = InstType::TypeJ;
    }

    pub fn decode_r(&mut self) {
        self.typ = InstType::TypeR;
    }
}
