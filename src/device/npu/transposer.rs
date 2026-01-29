use crate::device::npu::NpuState;
use crate::device::npu::SRAM_SIZE;

pub struct TransposeParams {
    pub src_offset: u32,
    pub dst_offset: u32,
    pub rows: u32,
    pub cols: u32,
    // Assuming 1 byte per element for now (int8). 
    // If we need int32 transpose, we need element_size.
    pub element_size: u32, 
}

pub fn run_transpose(st: &mut NpuState, params: &TransposeParams) {
    // We assume transposition happens typically on WEIGHTS or OUTPUTS?
    // Gemmini transposes Weights? Or Output?
    // Let's assume operation on 'weight_sram' or 'feature_sram'.
    // We should probably allow selecting which SRAM?
    // For now, let's assume it operates on `weight_sram` as weights usually need transpose.
    // Or make it generic.
    // But `NpuState` exposes specific srams.
    // We'll define it to default to `weight_sram` for now, or add a target selector.
    
    // Actually, let's make it operate on a passed buffer slice, but that's hard with NpuState mutex.
    // We'll assume it targets `weight_sram` as that's the primary use case for NPU [K,N] -> [N,K] stuff.
    // OR we provide a register to select the SRAM bank.
    
    // For this generic implementation, let's operate on `weight_sram` to start, 
    // as `conv2d` needs weight transpose.
    
    let src = &st.weight_sram;
    let src_buf = src.clone();
    let dst = &mut st.weight_sram;

    let rows = params.rows as usize;
    let cols = params.cols as usize;
    let es = params.element_size as usize;
    let src_off = params.src_offset as usize;
    let dst_off = params.dst_offset as usize;

    for r in 0..rows {
        for c in 0..cols {
            // Src Index: row-major [r, c]
            let src_idx = src_off + (r * cols + c) * es;
            // Dst Index: row-major [c, r] (which is transposed)
            let dst_idx = dst_off + (c * rows + r) * es;

            if src_idx + es <= SRAM_SIZE && dst_idx + es <= SRAM_SIZE {
                for b in 0..es {
                    dst[dst_idx + b] = src_buf[src_idx + b];
                }
            }
        }
    }
}
