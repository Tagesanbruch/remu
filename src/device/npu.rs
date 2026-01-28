//! NPU Device Simulation with Internal SRAM and DMA Engine
//!
//! Memory Map (Base: 0x2100_0000):
//!   0x0000 - 0x00FF: Control Registers
//!   0x1000 - 0x4FFF: Feature SRAM (16KB)
//!   0x5000 - 0x8FFF: Weight SRAM (16KB)
//!   0x9000 - 0xCFFF: Output SRAM (16KB)
//!
//! Register Layout (0x00 - 0xFF):
//!   0x00: CTRL       - Control/Status Register
//!   0x04: STATUS     - Status (Bit0: Busy, Bit1: Done, Bit2: Error)
//!   0x08: DMA_SRC    - DMA MM2S Source Address (DRAM)
//!   0x0C: DMA_DST    - DMA S2MM Dest Address (DRAM)
//!   0x10: DMA_LEN    - DMA Transfer Length (bytes)
//!   0x14: DMA_DIR    - DMA Direction (0: MM2S to Feature, 1: MM2S to Weight, 2: S2MM from Output)
//!   0x18: DMA_CTRL   - DMA Control (Write 1 to start DMA)
//!   0x20: GEMM_M     - Matrix M dimension
//!   0x24: GEMM_N     - Matrix N dimension  
//!   0x28: GEMM_K     - Matrix K dimension
//!   0x2C: GEMM_CTRL  - GEMM Control (Write 1 to start GEMM)
//!   0x30: IM2COL_EN  - Im2Col Enable
//!   0x34: IM2COL_H   - Input Height
//!   0x38: IM2COL_W   - Input Width
//!   0x3C: IM2COL_C   - Input Channels
//!   0x40: IM2COL_KH  - Kernel Height
//!   0x44: IM2COL_KW  - Kernel Width
//!   0x48: IM2COL_PAD - Padding
//!   0x4C: IM2COL_STR - Stride
//!   0x80: PERF_CYCLES- Performance: Active Cycles
//!   0x84: PERF_BYTES - Performance: Memory Traffic (lower 32)
//!   0x88: PERF_BYTES_H - Performance: Memory Traffic (upper 32)

use crate::common::{PAddr, Word};
use crate::memory::mmio::register_mmio;
use std::sync::Mutex;
use lazy_static::lazy_static;

pub const NPU_MMIO_BASE: u32 = 0x21000000;
pub const NPU_MMIO_SIZE: usize = 0x10000; // 64KB total

// SRAM regions (offsets from base)
const SRAM_FEATURE_OFFSET: u32 = 0x1000;
const SRAM_WEIGHT_OFFSET: u32 = 0x5000;
const SRAM_OUTPUT_OFFSET: u32 = 0x9000;
const SRAM_SIZE: usize = 0x4000; // 16KB each

// Register offsets
const REG_CTRL: u32 = 0x00;
const REG_STATUS: u32 = 0x04;
const REG_DMA_SRC: u32 = 0x08;
const REG_DMA_DST: u32 = 0x0C;
const REG_DMA_LEN: u32 = 0x10;
const REG_DMA_DIR: u32 = 0x14;
const REG_DMA_CTRL: u32 = 0x18;
const REG_GEMM_M: u32 = 0x20;
const REG_GEMM_N: u32 = 0x24;
const REG_GEMM_K: u32 = 0x28;
const REG_GEMM_CTRL: u32 = 0x2C;
const REG_IM2COL_EN: u32 = 0x30;
const REG_IM2COL_H: u32 = 0x34;
const REG_IM2COL_W: u32 = 0x38;
const REG_IM2COL_C: u32 = 0x3C;
const REG_IM2COL_KH: u32 = 0x40;
const REG_IM2COL_KW: u32 = 0x44;
const REG_IM2COL_PAD: u32 = 0x48;
const REG_IM2COL_STR: u32 = 0x4C;
const REG_PERF_CYCLES: u32 = 0x80;
const REG_PERF_BYTES: u32 = 0x84;
const REG_PERF_BYTES_H: u32 = 0x88;

// DMA Direction
const DMA_DIR_MM2S_FEATURE: u32 = 0;
const DMA_DIR_MM2S_WEIGHT: u32 = 1;
const DMA_DIR_S2MM_OUTPUT: u32 = 2;

// Status bits
const STATUS_BUSY: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;
const STATUS_ERROR: u32 = 1 << 2;

struct NpuState {
    // Control Registers
    ctrl: u32,
    status: u32,
    
    // DMA Registers
    dma_src: u32,
    dma_dst: u32,
    dma_len: u32,
    dma_dir: u32,
    
    // GEMM Registers
    gemm_m: u32,
    gemm_n: u32,
    gemm_k: u32,
    
    // Im2Col Registers
    im2col_en: u32,
    im2col_h: u32,
    im2col_w: u32,
    im2col_c: u32,
    im2col_kh: u32,
    im2col_kw: u32,
    im2col_pad: u32,
    im2col_str: u32,
    
    // Internal SRAM (acts as scratchpad)
    feature_sram: Vec<u8>,
    weight_sram: Vec<u8>,
    output_sram: Vec<u8>,
    
    // Performance Counters
    perf_cycles: u64,
    perf_bytes: u64,
}

impl NpuState {
    fn new() -> Self {
        Self {
            ctrl: 0,
            status: 0,
            dma_src: 0,
            dma_dst: 0,
            dma_len: 0,
            dma_dir: 0,
            gemm_m: 0,
            gemm_n: 0,
            gemm_k: 0,
            im2col_en: 0,
            im2col_h: 0,
            im2col_w: 0,
            im2col_c: 0,
            im2col_kh: 0,
            im2col_kw: 0,
            im2col_pad: 0,
            im2col_str: 0,
            feature_sram: vec![0u8; SRAM_SIZE],
            weight_sram: vec![0u8; SRAM_SIZE],
            output_sram: vec![0u8; SRAM_SIZE],
            perf_cycles: 0,
            perf_bytes: 0,
        }
    }
}

lazy_static! {
    static ref NPU: Mutex<NpuState> = Mutex::new(NpuState::new());
}

pub fn init_npu() {
    crate::Log!("NPU: Initializing at 0x{:08x}, size 0x{:x}", NPU_MMIO_BASE, NPU_MMIO_SIZE);
    register_mmio("npu", NPU_MMIO_BASE, NPU_MMIO_SIZE as usize, Box::new(npu_callback));
}

fn npu_callback(addr: PAddr, len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - NPU_MMIO_BASE;
    let mut state = NPU.lock().unwrap();
    
    // Check if accessing SRAM regions
    if offset >= SRAM_FEATURE_OFFSET && offset < SRAM_FEATURE_OFFSET + SRAM_SIZE as u32 {
        let sram_offset = (offset - SRAM_FEATURE_OFFSET) as usize;
        return handle_sram_access(&mut state.feature_sram, sram_offset, len, is_write, data);
    }
    if offset >= SRAM_WEIGHT_OFFSET && offset < SRAM_WEIGHT_OFFSET + SRAM_SIZE as u32 {
        let sram_offset = (offset - SRAM_WEIGHT_OFFSET) as usize;
        return handle_sram_access(&mut state.weight_sram, sram_offset, len, is_write, data);
    }
    if offset >= SRAM_OUTPUT_OFFSET && offset < SRAM_OUTPUT_OFFSET + SRAM_SIZE as u32 {
        let sram_offset = (offset - SRAM_OUTPUT_OFFSET) as usize;
        return handle_sram_access(&mut state.output_sram, sram_offset, len, is_write, data);
    }
    
    // Register access
    if is_write {
        handle_reg_write(&mut state, offset, data)
    } else {
        handle_reg_read(&state, offset)
    }
}

fn handle_sram_access(sram: &mut Vec<u8>, offset: usize, len: usize, is_write: bool, data: Word) -> Word {
    if offset + len > sram.len() {
        log::error!("NPU SRAM access out of bounds: offset=0x{:x}, len={}", offset, len);
        return 0;
    }
    
    if is_write {
        match len {
            1 => sram[offset] = data as u8,
            2 => {
                sram[offset] = data as u8;
                sram[offset + 1] = (data >> 8) as u8;
            }
            4 => {
                sram[offset] = data as u8;
                sram[offset + 1] = (data >> 8) as u8;
                sram[offset + 2] = (data >> 16) as u8;
                sram[offset + 3] = (data >> 24) as u8;
            }
            _ => {}
        }
        0
    } else {
        match len {
            1 => sram[offset] as Word,
            2 => (sram[offset] as Word) | ((sram[offset + 1] as Word) << 8),
            4 => {
                (sram[offset] as Word)
                    | ((sram[offset + 1] as Word) << 8)
                    | ((sram[offset + 2] as Word) << 16)
                    | ((sram[offset + 3] as Word) << 24)
            }
            _ => 0,
        }
    }
}

fn handle_reg_write(state: &mut NpuState, offset: u32, data: Word) -> Word {
    match offset {
        REG_CTRL => {
            state.ctrl = data;
            if data & 1 != 0 {
                // Global reset
                state.status = 0;
                state.perf_cycles = 0;
                state.perf_bytes = 0;
            }
        }
        REG_DMA_SRC => state.dma_src = data,
        REG_DMA_DST => state.dma_dst = data,
        REG_DMA_LEN => state.dma_len = data,
        REG_DMA_DIR => state.dma_dir = data,
        REG_DMA_CTRL => {
            if data & 1 != 0 {
                run_dma(state);
            }
        }
        REG_GEMM_M => state.gemm_m = data,
        REG_GEMM_N => state.gemm_n = data,
        REG_GEMM_K => state.gemm_k = data,
        REG_GEMM_CTRL => {
            if data & 1 != 0 {
                run_gemm(state);
            }
        }
        REG_IM2COL_EN => state.im2col_en = data,
        REG_IM2COL_H => state.im2col_h = data,
        REG_IM2COL_W => state.im2col_w = data,
        REG_IM2COL_C => state.im2col_c = data,
        REG_IM2COL_KH => state.im2col_kh = data,
        REG_IM2COL_KW => state.im2col_kw = data,
        REG_IM2COL_PAD => state.im2col_pad = data,
        REG_IM2COL_STR => state.im2col_str = data,
        _ => {}
    }
    0
}

fn handle_reg_read(state: &NpuState, offset: u32) -> Word {
    match offset {
        REG_CTRL => state.ctrl,
        REG_STATUS => state.status,
        REG_DMA_SRC => state.dma_src,
        REG_DMA_DST => state.dma_dst,
        REG_DMA_LEN => state.dma_len,
        REG_DMA_DIR => state.dma_dir,
        REG_GEMM_M => state.gemm_m,
        REG_GEMM_N => state.gemm_n,
        REG_GEMM_K => state.gemm_k,
        REG_IM2COL_EN => state.im2col_en,
        REG_IM2COL_H => state.im2col_h,
        REG_IM2COL_W => state.im2col_w,
        REG_IM2COL_C => state.im2col_c,
        REG_IM2COL_KH => state.im2col_kh,
        REG_IM2COL_KW => state.im2col_kw,
        REG_IM2COL_PAD => state.im2col_pad,
        REG_IM2COL_STR => state.im2col_str,
        REG_PERF_CYCLES => state.perf_cycles as u32,
        REG_PERF_BYTES => state.perf_bytes as u32,
        REG_PERF_BYTES_H => (state.perf_bytes >> 32) as u32,
        _ => 0,
    }
}

/// DMA Engine: Transfer data between DRAM and NPU SRAM
fn run_dma(state: &mut NpuState) {
    let len = state.dma_len as usize;
    if len == 0 {
        return;
    }
    
    state.status |= STATUS_BUSY;
    
    match state.dma_dir {
        DMA_DIR_MM2S_FEATURE => {
            // DRAM -> Feature SRAM
            for i in 0..len {
                if i >= SRAM_SIZE {
                    break;
                }
                let val = crate::memory::paddr::paddr_read(state.dma_src + i as u32, 1) as u8;
                state.feature_sram[i] = val;
            }
            crate::Log!("NPU DMA: MM2S Feature, src=0x{:08x}, len={}", state.dma_src, len);
        }
        DMA_DIR_MM2S_WEIGHT => {
            // DRAM -> Weight SRAM
            for i in 0..len {
                if i >= SRAM_SIZE {
                    break;
                }
                let val = crate::memory::paddr::paddr_read(state.dma_src + i as u32, 1) as u8;
                state.weight_sram[i] = val;
            }
            crate::Log!("NPU DMA: MM2S Weight, src=0x{:08x}, len={}", state.dma_src, len);
        }
        DMA_DIR_S2MM_OUTPUT => {
            // Output SRAM -> DRAM
            for i in 0..len {
                if i >= SRAM_SIZE {
                    break;
                }
                crate::memory::paddr::paddr_write(state.dma_dst + i as u32, 1, state.output_sram[i] as u32);
            }
            crate::Log!("NPU DMA: S2MM Output, dst=0x{:08x}, len={}", state.dma_dst, len);
        }
        _ => {
            log::error!("NPU DMA: Invalid direction {}", state.dma_dir);
            state.status |= STATUS_ERROR;
        }
    }
    
    state.perf_bytes += len as u64;
    state.status &= !STATUS_BUSY;
    state.status |= STATUS_DONE;
}

/// GEMM Engine: Matrix multiplication on internal SRAM
/// A (MxK, i8) from Feature SRAM
/// B (KxN, i8) from Weight SRAM  
/// C (MxN, i32) to Output SRAM
fn run_gemm(state: &mut NpuState) {
    let m = state.gemm_m as usize;
    let n = state.gemm_n as usize;
    let k = state.gemm_k as usize;
    
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    
    // Check bounds
    let size_a = m * k;
    let size_b = k * n;
    let size_c = m * n * 4; // i32 output
    
    if size_a > SRAM_SIZE || size_b > SRAM_SIZE || size_c > SRAM_SIZE {
        log::error!("NPU GEMM: Matrix too large for SRAM");
        state.status |= STATUS_ERROR;
        return;
    }
    
    state.status |= STATUS_BUSY;
    crate::Log!("NPU GEMM: M={}, N={}, K={}", m, n, k);
    
    // Perform GEMM: C = A * B
    for r in 0..m {
        for c in 0..n {
            let mut sum: i32 = 0;
            for i in 0..k {
                let a_val = state.feature_sram[r * k + i] as i8 as i32;
                let b_val = state.weight_sram[i * n + c] as i8 as i32;
                sum += a_val * b_val;
            }
            // Write to output SRAM (little-endian i32)
            let out_idx = (r * n + c) * 4;
            state.output_sram[out_idx] = sum as u8;
            state.output_sram[out_idx + 1] = (sum >> 8) as u8;
            state.output_sram[out_idx + 2] = (sum >> 16) as u8;
            state.output_sram[out_idx + 3] = (sum >> 24) as u8;
        }
    }
    
    // Performance model: cycles = M*N*K / (array_size^2)
    let ops = (m * n * k) as u64;
    let array_size = 16u64; // 16x16 systolic array
    let cycles = ops.div_ceil(array_size * array_size);
    state.perf_cycles += cycles;
    
    state.status &= !STATUS_BUSY;
    state.status |= STATUS_DONE;
}

pub fn dump_npu_profile() -> String {
    let state = NPU.lock().unwrap();
    format!(
        r#"{{
  "NPU Active Cycles": {},
  "Memory Traffic (Bytes)": {},
  "Estimated Utilization": "{:.1}%"
}}"#,
        state.perf_cycles,
        state.perf_bytes,
        if state.perf_cycles > 0 { 100.0 } else { 0.0 }
    )
}