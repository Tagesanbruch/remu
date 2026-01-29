use crate::device::npu::NpuState;
use crate::device::npu::SRAM_SIZE;

// Register definitions for Im2Col (Relative to Im2Col block base)
// We will assign these in mod.rs later.

pub struct Im2ColParams {
    pub src_offset: u32,
    pub dst_offset: u32,
    pub input_h: u32,
    pub input_w: u32,
    pub channels: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub pad_top: u32,
    pub pad_left: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
}

pub fn run_im2col(st: &mut NpuState, params: &Im2ColParams) {
    let src = &st.feature_sram;
    // We need to write to a temporary buffer first or handle overlapping carefully?
    // Since we mutate st.feature_sram, simple index tracking might be tricky if src/dst overlap.
    // Safe bet: Clone src or write to separate buffer then copy back.
    // Given the memory size (16KB), cloning is cheap.
    let src_buf = src.clone();
    let dst = &mut st.feature_sram;

    let ih = params.input_h as i32;
    let iw = params.input_w as i32;
    let kh = params.kernel_h as i32;
    let kw = params.kernel_w as i32;
    let ic = params.channels as i32;
    let pad_t = params.pad_top as i32;
    let pad_l = params.pad_left as i32;
    let stride_h = params.stride_h as i32;
    let stride_w = params.stride_w as i32;
    let dil_h = params.dilation_h as i32;
    let dil_w = params.dilation_w as i32;

    // Output Dimensions
    let oh = (ih + 2 * pad_t - (dil_h * (kh - 1) + 1)) / stride_h + 1;
    let ow = (iw + 2 * pad_l - (dil_w * (kw - 1) + 1)) / stride_w + 1;

    let mut dst_idx = params.dst_offset as usize;

    // Standard Im2Col:
    // For each output pixel (h, w), extract the patch of size (ic, kh, kw)
    // and flatten it into a row.
    // Layout: The Chisel code suggests it interacts with a Systolic Array.
    // Usually, Systolic Arrays want A (Input) as [M, K].
    // M = OH * OW (Spatial pixels)
    // K = IC * KH * KW (Channels * Kernel)
    // So we iterate M, then K.

    for r in 0..oh {
        for c in 0..ow {
            // This is one "row" of the output matrix (one spatial position)
            // Inner loops iterate over K dimension
            
            // Order of K:
            // Usually (Channels, KH, KW) or (KH, KW, Channels).
            // NCHW data layout usually implies Channels is outer or inner?
            // If Input is NCHW: Input[c][y][x]
            // If Input is NHWC: Input[y][x][c]
            
            // Let's assume NCHW input based on previous npu_conv2d analysis.
            // And usually for GEMM, we want the dot product k-dim to match weights.
            // Weights are usually [OutC, InC, KH, KW] -> Flattened [N, K].
            // If Weights are flattened NCHW-style, then K is C*KH*KW.
            // So we should iterate C, then KH, then KW? Or C is the "outer" of the K-dim?
            
            // Let's stick to (C, KH, KW) order to match NCHW flattening.
            
            for ch in 0..ic {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let cur_y = r * stride_h - pad_t + ky * dil_h;
                        let cur_x = c * stride_w - pad_l + kx * dil_w;

                        let val: u8 = if cur_y >= 0 && cur_y < ih && cur_x >= 0 && cur_x < iw {
                            // Calculate Source Index (NCHW)
                            // Index = ch * (ih * iw) + cur_y * iw + cur_x
                            let src_idx = (params.src_offset as i32
                                + ch * (ih * iw)
                                + cur_y * iw
                                + cur_x) as usize;
                            
                            if src_idx < src_buf.len() {
                                src_buf[src_idx]
                            } else {
                                0
                            }
                        } else {
                            0 // Padding
                        };

                        // Write to Destination
                        if dst_idx < SRAM_SIZE {
                            dst[dst_idx] = val;
                        }
                        dst_idx += 1;
                    }
                }
            }
        }
    }
}
