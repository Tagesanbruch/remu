/**
 * NPU Utility Functions
 * 
 * Data layout transformations, quantization, etc.
 */

#include "npu_hw.h"
#include <stdint.h>

// ============================================================================
// Data Layout Transformations
// ============================================================================

/**
 * Transpose matrix: B[n,m] = A[m,n]
 */
void transpose_matrix_i8(int8_t *src, int8_t *dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * m + i] = src[i * n + j];
        }
    }
}

/**
 * Transpose matrix: B[n,m] = A[m,n] (int32)
 */
void transpose_matrix_i32(int32_t *src, int32_t *dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * m + i] = src[i * n + j];
        }
    }
}

/**
 * Extract weight columns for tiled MatMul
 * 
 * From B[k,n] row-major, extract columns [col_start, col_end) as contiguous B_tile[k,n_tile]
 */
void extract_weight_columns_i8(int8_t *src, int8_t *dst, int k, int n, int col_start, int col_end) {
    int n_tile = col_end - col_start;
    for (int ki = 0; ki < k; ki++) {
        for (int ni = 0; ni < n_tile; ni++) {
            dst[ki * n_tile + ni] = src[ki * n + (col_start + ni)];
        }
    }
}

/**
 * Copy output tile to strided destination
 * 
 * From C_tile[m,n_tile] contiguous to C[m,n] at columns [col_start, col_end)
 */
void scatter_output_columns_i32(int32_t *src, int32_t *dst, int m, int n, int col_start, int col_end) {
    int n_tile = col_end - col_start;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n_tile; ni++) {
            dst[mi * n + (col_start + ni)] = src[mi * n_tile + ni];
        }
    }
}

// ============================================================================
// Im2Col for Convolution
// ============================================================================

/**
 * Im2col transformation for a single output position
 * 
 * Extracts a patch from input corresponding to one output pixel and flattens to [K] vector.
 * K = in_c * kh * kw
 */
void im2col_single(int8_t *input, int8_t *col, int in_c, int in_h, int in_w,
                   int kh, int kw, int oh, int ow, int pad, int stride) {
    int K = in_c * kh * kw;
    for (int ic = 0; ic < in_c; ic++) {
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                int ih = oh * stride - pad + ky;
                int iw = ow * stride - pad + kx;
                int k_idx = ic * kh * kw + ky * kw + kx;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    col[k_idx] = input[ic * in_h * in_w + ih * in_w + iw];
                } else {
                    col[k_idx] = 0;  // Zero padding
                }
            }
        }
    }
}

/**
 * Im2col transformation for a tile of output positions
 * 
 * Processes spatial positions [m_start, m_end) and produces [M_cur, K] matrix.
 */
void im2col_tile(int8_t *input, int8_t *col_buf,
                 int in_c, int in_h, int in_w,
                 int kh, int kw, int out_w,
                 int m_start, int m_end, int pad, int stride) {
    int K = in_c * kh * kw;
    int M_cur = m_end - m_start;
    
    for (int m_idx = 0; m_idx < M_cur; m_idx++) {
        int m = m_start + m_idx;
        int oh = m / out_w;
        int ow = m % out_w;
        im2col_single(input, col_buf + m_idx * K, in_c, in_h, in_w,
                      kh, kw, oh, ow, pad, stride);
    }
}

// ============================================================================
// Quantization Utilities
// ============================================================================

/**
 * Requantize int32 to int8 with scale and zero point
 * 
 * output = clamp((input * scale) >> 16 + zero_point, -128, 127)
 */
void requantize_i32_to_i8(int32_t *input, int8_t *output, int len,
                          int32_t scale_q16, int8_t zero_point) {
    for (int i = 0; i < len; i++) {
        int64_t scaled = ((int64_t)input[i] * scale_q16) >> 16;
        int32_t result = (int32_t)scaled + zero_point;
        
        // Clamp to int8 range
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        output[i] = (int8_t)result;
    }
}

/**
 * Dequantize int8 to float (conceptually, returns scaled int32 for fixed-point)
 */
void dequantize_i8_to_i32(int8_t *input, int32_t *output, int len,
                          int32_t scale_q16, int8_t zero_point) {
    for (int i = 0; i < len; i++) {
        int32_t val = input[i] - zero_point;
        output[i] = (val * scale_q16) >> 8;  // Q8 output
    }
}

// ============================================================================
// Clamp and Saturation
// ============================================================================

static inline int32_t clamp_i32(int32_t val, int32_t min_val, int32_t max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline int8_t saturate_i32_to_i8(int32_t val) {
    return (int8_t)clamp_i32(val, -128, 127);
}

/**
 * Saturate int32 array to int8
 */
void saturate_array_i32_to_i8(int32_t *input, int8_t *output, int len) {
    for (int i = 0; i < len; i++) {
        output[i] = saturate_i32_to_i8(input[i]);
    }
}
