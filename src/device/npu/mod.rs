//! NPU Device Simulation with Internal SRAM and DMA Engine
//!
//! Memory Map (Base: 0x2100_0000):
//!   0x0000 - 0x00FF: Control Registers
//!   0x1000 - 0x4FFF: Feature SRAM (16KB)
//!   0x5000 - 0x8FFF: Weight SRAM (16KB)
//!   0x9000 - 0xCFFF: Output SRAM (16KB)

mod im2col;
mod transposer;

use crate::common::{PAddr, Word};
use crate::memory::mmio::register_mmio;
use lazy_static::lazy_static;
use std::sync::Mutex;
use self::im2col::{run_im2col, Im2ColParams};
use self::transposer::{run_transpose, TransposeParams};

pub const NPU_MMIO_BASE: u32 = 0x21000000;
pub const NPU_MMIO_SIZE: usize = 0x10000; // 64KB

const SRAM_SIZE: usize = 0x4000; // 16KB each

// Register Offsets
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
const REG_GEMM_A_OFFSET: u32 = 0x30;
const REG_GEMM_B_OFFSET: u32 = 0x34;
const REG_GEMM_C_OFFSET: u32 = 0x38;

const REG_ACT_TYPE: u32 = 0x50;
const REG_ACT_LEN: u32 = 0x54;
const REG_ACT_CTRL: u32 = 0x58;
const REG_ACT_PARAM: u32 = 0x5C;

const REG_QUANT_SCALE: u32 = 0x60;
const REG_QUANT_ZERO: u32 = 0x64;
const REG_QUANT_LEN: u32 = 0x68;
const REG_QUANT_CTRL: u32 = 0x6C;

const REG_PERF_CYCLES: u32 = 0x80;
const REG_PERF_BYTES: u32 = 0x84;
const REG_PERF_BYTES_H: u32 = 0x88;
const REG_PERF_GEMM_CNT: u32 = 0x8C;
const REG_PERF_ACT_CNT: u32 = 0x90;
const REG_PERF_DMA_CNT: u32 = 0x94;

// Im2Col Registers
const REG_IM2COL_CTRL: u32 = 0xA0;
const REG_IM2COL_SRC_OFF: u32 = 0xA4;
const REG_IM2COL_DST_OFF: u32 = 0xA8;
const REG_IM2COL_IN_HW: u32 = 0xAC;
const REG_IM2COL_KER_HW: u32 = 0xB0;
const REG_IM2COL_CHANNELS: u32 = 0xB4;
const REG_IM2COL_STRIDE: u32 = 0xB8;
const REG_IM2COL_PADDING: u32 = 0xBC;
const REG_IM2COL_DILATION: u32 = 0xC0;

// Status Bits
const STATUS_BUSY: u32 = 1;
const STATUS_DONE: u32 = 2;
const STATUS_ERROR: u32 = 4;

// Activation Types
const ACT_RELU: u32 = 1;
const ACT_RELU6: u32 = 2;

pub struct NpuState {
    ctrl: u32,
    status: u32,
    dma_src: u32,
    dma_dst: u32,
    dma_len: u32,
    dma_dir: u32,
    gemm_m: u32,
    gemm_n: u32,
    gemm_k: u32,
    gemm_a_off: u32,
    gemm_b_off: u32,
    gemm_c_off: u32,
    act_type: u32,
    act_len: u32,
    act_param: u32,
    quant_scale: u32,
    quant_zero: u32,
    quant_len: u32,
    im2col_src_off: u32,
    im2col_dst_off: u32,
    im2col_in_hw: u32,
    im2col_ker_hw: u32,
    im2col_channels: u32,
    im2col_stride: u32,
    im2col_padding: u32,
    im2col_dilation: u32,
    trans_src_off: u32,
    trans_dst_off: u32,
    trans_dims: u32,
    trans_elem_size: u32,
    feature_sram: Vec<u8>,
    weight_sram: Vec<u8>,
    output_sram: Vec<u8>,
    perf_cycles: u64,
    perf_bytes: u64,
    perf_gemm_cnt: u32,
    perf_act_cnt: u32,
    perf_dma_cnt: u32,
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
            gemm_a_off: 0,
            gemm_b_off: 0,
            gemm_c_off: 0,
            act_type: 0,
            act_len: 0,
            act_param: 0,
            quant_scale: 0,
            quant_zero: 0,
            quant_len: 0,
            im2col_src_off: 0,
            im2col_dst_off: 0,
            im2col_in_hw: 0,
            im2col_ker_hw: 0,
            im2col_channels: 0,
            im2col_stride: 0,
            im2col_padding: 0,
            im2col_dilation: 0,
            trans_src_off: 0,
            trans_dst_off: 0,
            trans_dims: 0,
            trans_elem_size: 0,
            feature_sram: vec![0u8; SRAM_SIZE],
            weight_sram: vec![0u8; SRAM_SIZE],
            output_sram: vec![0u8; SRAM_SIZE],
            perf_cycles: 0,
            perf_bytes: 0,
            perf_gemm_cnt: 0,
            perf_act_cnt: 0,
            perf_dma_cnt: 0,
        }
    }
}

lazy_static! {
    static ref NPU: Mutex<NpuState> = Mutex::new(NpuState::new());
}

pub fn init_npu() {
    register_mmio("NPU", NPU_MMIO_BASE, NPU_MMIO_SIZE, Box::new(npu_mmio_callback));
}

fn npu_mmio_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = (addr - NPU_MMIO_BASE) as u32;
    let mut st = NPU.lock().unwrap();

    if is_write {
        reg_write(&mut st, offset, data)
    } else {
        reg_read(&st, offset)
    }
}

fn reg_write(st: &mut NpuState, off: u32, data: Word) -> Word {
    match off {
        REG_CTRL => {
            st.ctrl = data;
            if data & 1 != 0 {
                st.status = 0;
                st.perf_cycles = 0;
                st.perf_bytes = 0;
                st.perf_gemm_cnt = 0;
                st.perf_act_cnt = 0;
                st.perf_dma_cnt = 0;
            }
        }
        REG_DMA_SRC => st.dma_src = data,
        REG_DMA_DST => st.dma_dst = data,
        REG_DMA_LEN => st.dma_len = data,
        REG_DMA_DIR => st.dma_dir = data,
        REG_DMA_CTRL => {
            if data & 1 != 0 {
                run_dma(st);
            }
        }
        REG_GEMM_M => st.gemm_m = data,
        REG_GEMM_N => st.gemm_n = data,
        REG_GEMM_K => st.gemm_k = data,
        REG_GEMM_CTRL => {
            if data & 1 != 0 {
                run_gemm(st);
            }
        }
        REG_GEMM_A_OFFSET => st.gemm_a_off = data,
        REG_GEMM_B_OFFSET => st.gemm_b_off = data,
        REG_GEMM_C_OFFSET => st.gemm_c_off = data,
        
        REG_ACT_TYPE => st.act_type = data,
        REG_ACT_LEN => st.act_len = data,
        REG_ACT_CTRL => {
            if data & 1 != 0 {
                run_activation(st);
            }
        }
        REG_ACT_PARAM => st.act_param = data,

        REG_QUANT_SCALE => st.quant_scale = data,
        REG_QUANT_ZERO => st.quant_zero = data,
        REG_QUANT_LEN => st.quant_len = data,
        REG_QUANT_CTRL => {
            if data & 1 != 0 {
                run_quantize(st);
            }
        }

        REG_IM2COL_CTRL => {
            if data & 1 != 0 {
                let params = Im2ColParams {
                    src_offset: st.im2col_src_off,
                    dst_offset: st.im2col_dst_off,
                    input_h: st.im2col_in_hw >> 16,
                    input_w: st.im2col_in_hw & 0xFFFF,
                    channels: st.im2col_channels,
                    kernel_h: st.im2col_ker_hw >> 16,
                    kernel_w: st.im2col_ker_hw & 0xFFFF,
                    pad_top: st.im2col_padding >> 16,
                    pad_left: st.im2col_padding & 0xFFFF,
                    stride_h: st.im2col_stride >> 16,
                    stride_w: st.im2col_stride & 0xFFFF,
                    dilation_h: st.im2col_dilation >> 16,
                    dilation_w: st.im2col_dilation & 0xFFFF,
                };
                run_im2col(st, &params);
            }
        }
        REG_IM2COL_SRC_OFF => st.im2col_src_off = data,
        REG_IM2COL_DST_OFF => st.im2col_dst_off = data,
        REG_IM2COL_IN_HW => st.im2col_in_hw = data,
        REG_IM2COL_KER_HW => st.im2col_ker_hw = data,
        REG_IM2COL_CHANNELS => st.im2col_channels = data,
        REG_IM2COL_STRIDE => st.im2col_stride = data,
        REG_IM2COL_PADDING => st.im2col_padding = data,
        REG_IM2COL_DILATION => st.im2col_dilation = data,
        _ => {}
    }
    0
}

fn reg_read(st: &NpuState, off: u32) -> Word {
    match off {
        REG_STATUS => st.status,
        REG_DMA_SRC => st.dma_src,
        REG_DMA_DST => st.dma_dst,
        REG_DMA_LEN => st.dma_len,
        REG_DMA_DIR => st.dma_dir,
        REG_GEMM_M => st.gemm_m,
        REG_GEMM_N => st.gemm_n,
        REG_GEMM_K => st.gemm_k,
        REG_GEMM_A_OFFSET => st.gemm_a_off,
        REG_GEMM_B_OFFSET => st.gemm_b_off,
        REG_GEMM_C_OFFSET => st.gemm_c_off,
        REG_ACT_TYPE => st.act_type,
        REG_ACT_LEN => st.act_len,
        REG_ACT_PARAM => st.act_param,
        REG_QUANT_SCALE => st.quant_scale,
        REG_QUANT_ZERO => st.quant_zero,
        REG_QUANT_LEN => st.quant_len,
        REG_PERF_CYCLES => st.perf_cycles as u32,
        REG_PERF_BYTES => st.perf_bytes as u32,
        REG_PERF_BYTES_H => (st.perf_bytes >> 32) as u32,
        REG_PERF_GEMM_CNT => st.perf_gemm_cnt,
        REG_PERF_ACT_CNT => st.perf_act_cnt,
        REG_PERF_DMA_CNT => st.perf_dma_cnt,
        REG_IM2COL_SRC_OFF => st.im2col_src_off,
        REG_IM2COL_DST_OFF => st.im2col_dst_off,
        REG_IM2COL_IN_HW => st.im2col_in_hw,
        REG_IM2COL_KER_HW => st.im2col_ker_hw,
        REG_IM2COL_CHANNELS => st.im2col_channels,
        REG_IM2COL_STRIDE => st.im2col_stride,
        REG_IM2COL_PADDING => st.im2col_padding,
        REG_IM2COL_DILATION => st.im2col_dilation,
        _ => 0,
    }
}

fn run_dma(st: &mut NpuState) {
    // TODO: Implement DMA logic (Memory to SRAM and vice versa)
    // For now, just mark as done.
    st.status &= !STATUS_BUSY;
    st.status |= STATUS_DONE;
    st.perf_dma_cnt += 1;
}

fn run_gemm(st: &mut NpuState) {
    let m = st.gemm_m as usize;
    let n = st.gemm_n as usize;
    let k = st.gemm_k as usize;
    let a_off = st.gemm_a_off as usize;
    let b_off = st.gemm_b_off as usize;
    let c_off = st.gemm_c_off as usize;

    if m == 0 || n == 0 || k == 0 {
        return;
    }
    
    // Bounds Check
    if a_off + m * k > SRAM_SIZE || b_off + k * n > SRAM_SIZE || c_off + m * n * 4 > SRAM_SIZE {
        st.status |= STATUS_ERROR;
        return;
    }

    st.status |= STATUS_BUSY;

    // C[M,N] = A[M,K] * B[K,N]
    for r in 0..m {
        for c in 0..n {
            let mut sum: i32 = 0;
            for i in 0..k {
                let a = st.feature_sram[a_off + r * k + i] as i8 as i32;
                let b = st.weight_sram[b_off + i * n + c] as i8 as i32;
                sum += a * b;
            }
            let idx = c_off + (r * n + c) * 4;
            st.output_sram[idx] = sum as u8;
            st.output_sram[idx + 1] = (sum >> 8) as u8;
            st.output_sram[idx + 2] = (sum >> 16) as u8;
            st.output_sram[idx + 3] = (sum >> 24) as u8;
        }
    }
    
    // Stats
    let ops = (m * n * k) as u64;
    let cycles = (ops + 255) / 256; 
    st.perf_cycles += cycles;
    st.perf_gemm_cnt += 1;
    st.status &= !STATUS_BUSY;
    st.status |= STATUS_DONE;
}

fn run_activation(st: &mut NpuState) {
    let len = st.act_len as usize;
    if len == 0 || len * 4 > SRAM_SIZE {
        return;
    }

    st.status |= STATUS_BUSY;
    let param = st.act_param as i32;

    for i in 0..len {
        let idx = i * 4;
        let val = i32::from_le_bytes([
            st.output_sram[idx],
            st.output_sram[idx + 1],
            st.output_sram[idx + 2],
            st.output_sram[idx + 3],
        ]);

        let result = match st.act_type {
            ACT_RELU => val.max(0),
            ACT_RELU6 => {
                let max = if param == 0 { 6 << 16 } else { param };
                val.max(0).min(max)
            }
            _ => val,
        };

        st.output_sram[idx] = result as u8;
        st.output_sram[idx + 1] = (result >> 8) as u8;
        st.output_sram[idx + 2] = (result >> 16) as u8;
        st.output_sram[idx + 3] = (result >> 24) as u8;
    }

    st.perf_cycles += len as u64;
    st.perf_act_cnt += 1;
    st.status &= !STATUS_BUSY;
    st.status |= STATUS_DONE;
}

fn run_quantize(st: &mut NpuState) {
    let len = st.quant_len as usize;
    if len == 0 || len * 4 > SRAM_SIZE {
        return;
    }

    st.status |= STATUS_BUSY;
    let scale = st.quant_scale as i32;
    let zero = st.quant_zero as i32;

    let mut temp: Vec<i8> = Vec::with_capacity(len);
    for i in 0..len {
        let idx = i * 4;
        let val = i32::from_le_bytes([
            st.output_sram[idx],
            st.output_sram[idx + 1],
            st.output_sram[idx + 2],
            st.output_sram[idx + 3],
        ]);
        let shifted = if scale > 0 { val >> scale } else { val };
        let q = (shifted + zero).clamp(-128, 127) as i8;
        temp.push(q);
    }

    for (i, &v) in temp.iter().enumerate() {
        st.output_sram[i] = v as u8;
    }

    st.perf_cycles += (len as u64 + 3) / 4;
    st.status &= !STATUS_BUSY;
    st.status |= STATUS_DONE;
}

pub fn dump_npu_profile() -> String {
    let st = NPU.lock().unwrap();
    format!(
        r#"{{
  "npu_active_cycles": {},
  "memory_traffic_bytes": {},
  "gemm_ops": {},
  "activation_ops": {},
  "dma_transfers": {}
}}"#,
        st.perf_cycles,
        st.perf_bytes,
        st.perf_gemm_cnt,
        st.perf_act_cnt,
        st.perf_dma_cnt
    )
}