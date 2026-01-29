/**
 * NPU Kernel Implementations
 * 
 * P0 Operators: Conv2D, GEMM, MatMul
 * P1 Operators: DepthwiseConv2D, Pooling, Activation
 * P2 Operators: BatchNorm, Add, Mul, Requantize
 */

#include "npu_ops.h"
#include "npu_hw.h"
#include <stdint.h>

// External declarations from npu_tiling.c
extern int calc_matmul_tile_n(int m, int n, int k);
extern int calc_conv2d_tile_m(int M_total, int K, int N);
extern int needs_matmul_tiling(int m, int n, int k);
extern int needs_conv2d_tiling(int out_h, int out_w, int in_c, int out_c, int kh, int kw);
extern int calc_pool_tile_channels(int channels, int in_h, int in_w);
extern int calc_depthwise_conv_tile_channels(int channels, int in_h, int in_w, int kh, int kw);

// External declarations from npu_utils.c
extern void transpose_matrix_i8(int8_t *src, int8_t *dst, int m, int n);
extern void extract_weight_columns_i8(int8_t *src, int8_t *dst, int k, int n, int col_start, int col_end);
extern void scatter_output_columns_i32(int32_t *src, int32_t *dst, int m, int n, int col_start, int col_end);
extern void im2col_tile(int8_t *input, int8_t *col_buf, int in_c, int in_h, int in_w,
                        int kh, int kw, int out_w, int m_start, int m_end, int pad, int stride);
extern void requantize_i32_to_i8(int32_t *input, int8_t *output, int len, int32_t scale_q16, int8_t zero_point);

// ============================================================================
// Static Buffers (to avoid stack overflow)
// ============================================================================

static int8_t _matmul_weight_tile[SRAM_SIZE];
static int32_t _matmul_temp_out[SRAM_SIZE / sizeof(int32_t)];

static int8_t _conv2d_im2col_buf[SRAM_SIZE];
static int8_t _conv2d_weight_t[SRAM_SIZE];
static int32_t _conv2d_gemm_out[SRAM_SIZE / sizeof(int32_t)];

static int8_t _pool_input_buf[SRAM_SIZE];
static int8_t _pool_output_buf[SRAM_SIZE];

static int8_t _dw_input_buf[SRAM_SIZE];
static int8_t _dw_weight_buf[SRAM_SIZE];
static int32_t _dw_output_buf[SRAM_SIZE / sizeof(int32_t)];

// ============================================================================
// P0: Matrix Multiplication with Tiling
// ============================================================================

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    int n_tile = calc_matmul_tile_n(m, n, k);
    
    // If no tiling needed
    if (n_tile >= n && m * k <= (int)SRAM_SIZE) {
        npu_dma_load_feature(a, m * k);
        npu_dma_load_weight(b, k * n);
        npu_gemm(m, n, k);
        npu_dma_store_output(c, m * n * sizeof(int32_t));
        return;
    }
    
    // Load feature once (A[m,k])
    npu_dma_load_feature(a, m * k);
    
    // Process output columns in tiles
    for (int n_start = 0; n_start < n; n_start += n_tile) {
        int n_end = n_start + n_tile;
        if (n_end > n) n_end = n;
        int n_cur = n_end - n_start;
        
        // Extract weight tile: B[:, n_start:n_end] from B[k,n] layout
        extract_weight_columns_i8(b, _matmul_weight_tile, k, n, n_start, n_end);
        npu_dma_load_weight(_matmul_weight_tile, k * n_cur);
        
        // GEMM: [m, k] * [k, n_cur] -> [m, n_cur]
        npu_gemm(m, n_cur, k);
        
        // Store to temp and scatter to output
        npu_dma_store_output(_matmul_temp_out, m * n_cur * sizeof(int32_t));
        scatter_output_columns_i32(_matmul_temp_out, c, m, n, n_start, n_end);
    }
}

// ============================================================================
// P0: Conv2D with Im2col + GEMM and Tiling
// ============================================================================

void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    
    int M_total = out_h * out_w;  // total spatial positions
    int N = out_c;                // filters
    int K = in_c * kh * kw;       // kernel volume
    
    int M_tile = calc_conv2d_tile_m(M_total, K, N);
    
    for (int b = 0; b < batch; b++) {
        int8_t *in_batch = input + b * (in_c * in_h * in_w);
        int32_t *out_batch = output + b * (out_c * out_h * out_w);
        
        // Transpose weight once per batch: [N, K] -> [K, N]
        for (int ni = 0; ni < N; ni++) {
            for (int ki = 0; ki < K; ki++) {
                if (ki * N + ni < (int)SRAM_SIZE) {
                    _conv2d_weight_t[ki * N + ni] = weight[ni * K + ki];
                }
            }
        }
        npu_dma_load_weight(_conv2d_weight_t, K * N);
        
        // Process spatial positions in tiles
        for (int m_start = 0; m_start < M_total; m_start += M_tile) {
            int m_end = m_start + M_tile;
            if (m_end > M_total) m_end = M_total;
            int M_cur = m_end - m_start;
            
            // Im2col for current tile
            im2col_tile(in_batch, _conv2d_im2col_buf, in_c, in_h, in_w,
                        kh, kw, out_w, m_start, m_end, pad, stride);
            
            // Load im2col tile
            npu_dma_load_feature(_conv2d_im2col_buf, M_cur * K);
            
            // GEMM: [M_cur, K] * [K, N] -> [M_cur, N]
            npu_gemm(M_cur, N, K);
            
            // Apply activation if requested
            if (act_type == ACT_RELU) {
                npu_relu(M_cur * N);
            }
            
            // Store tile result
            npu_dma_store_output(_conv2d_gemm_out, M_cur * N * sizeof(int32_t));
            
            // Transpose output tile: [M_cur, N] -> write to [out_c, out_h, out_w]
            for (int m_idx = 0; m_idx < M_cur; m_idx++) {
                int m = m_start + m_idx;
                int oh = m / out_w;
                int ow = m % out_w;
                
                for (int c = 0; c < N; c++) {
                    out_batch[c * out_h * out_w + oh * out_w + ow] = _conv2d_gemm_out[m_idx * N + c];
                }
            }
        }
    }
}

// ============================================================================
// P1: Depthwise Convolution
// ============================================================================

void npu_depthwise_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                          int batch, int channels, int in_h, int in_w,
                          int kh, int kw, int pad, int stride,
                          uint32_t act_type) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    
    // Process channel by channel (depthwise = each channel processed independently)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *wt_ch = weight + c * (kh * kw);
            int32_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            
            // Depthwise conv for single channel (direct implementation)
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += (int32_t)in_ch[ih * in_w + iw] * (int32_t)wt_ch[ky * kw + kx];
                            }
                        }
                    }
                    
                    // Apply activation
                    if (act_type == ACT_RELU && sum < 0) sum = 0;
                    
                    out_ch[oh * out_w + ow] = sum;
                }
            }
        }
    }
}

// ============================================================================
// P1: Max Pooling 2D
// ============================================================================

void npu_maxpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int8_t max_val = -128;  // Min int8
                    
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int8_t val = in_ch[ih * in_w + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out_ch[oh * out_w + ow] = max_val;
                }
            }
        }
    }
}

// ============================================================================
// P1: Average Pooling 2D
// ============================================================================

void npu_avgpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    int pool_size = kh * kw;
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    int count = 0;
                    
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += in_ch[ih * in_w + iw];
                                count++;
                            }
                        }
                    }
                    
                    // Average (rounded)
                    out_ch[oh * out_w + ow] = (int8_t)((sum + count/2) / count);
                }
            }
        }
    }
}

// ============================================================================
// P1: Global Average Pooling 2D
// ============================================================================

void npu_global_avgpool2d(int8_t *input, int32_t *output,
                          int batch, int channels, int in_h, int in_w) {
    
    int spatial = in_h * in_w;
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * spatial) + c * spatial;
            int32_t sum = 0;
            
            for (int i = 0; i < spatial; i++) {
                sum += in_ch[i];
            }
            
            // Return average (rounded)
            int32_t avg = (sum + spatial / 2) / spatial;
            output[b * channels + c] = avg;
        }
    }
}

// ============================================================================
// P1: Activation Functions (Element-wise)
// ============================================================================

void npu_relu_elementwise(void *input, void *output, int len, int dtype) {
    if (dtype == 0) {  // int8
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int i = 0; i < len; i++) {
            out[i] = (in[i] > 0) ? in[i] : 0;
        }
    } else {  // int32
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) {
            out[i] = (in[i] > 0) ? in[i] : 0;
        }
    }
}

void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype, int32_t alpha_q16) {
    if (dtype == 0) {  // int8
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int i = 0; i < len; i++) {
            if (in[i] > 0) {
                out[i] = in[i];
            } else {
                // Leaky: alpha * x (Q16 fixed-point)
                int32_t scaled = ((int32_t)in[i] * alpha_q16) >> 16;
                out[i] = (int8_t)(scaled < -128 ? -128 : scaled);
            }
        }
    } else {  // int32
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) {
            if (in[i] > 0) {
                out[i] = in[i];
            } else {
                out[i] = ((int64_t)in[i] * alpha_q16) >> 16;
            }
        }
    }
}

void npu_clip_elementwise(void *input, void *output, int len, int dtype, int32_t min_val, int32_t max_val) {
    if (dtype == 0) {  // int8
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        int8_t min8 = (int8_t)(min_val < -128 ? -128 : (min_val > 127 ? 127 : min_val));
        int8_t max8 = (int8_t)(max_val < -128 ? -128 : (max_val > 127 ? 127 : max_val));
        for (int i = 0; i < len; i++) {
            int8_t val = in[i];
            if (val < min8) val = min8;
            if (val > max8) val = max8;
            out[i] = val;
        }
    } else {  // int32
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) {
            int32_t val = in[i];
            if (val < min_val) val = min_val;
            if (val > max_val) val = max_val;
            out[i] = val;
        }
    }
}

void npu_relu6_elementwise(void *input, void *output, int len, int dtype) {
    // ReLU6 = Clip(x, 0, 6)
    // For quantized int8, 6 may be scaled differently; assuming direct value here
    npu_clip_elementwise(input, output, len, dtype, 0, 6);
}

// ============================================================================
// P2: Batch Normalization (Inference, Fused)
// ============================================================================

void npu_batchnorm(int8_t *input, int8_t *output,
                   int32_t *gamma, int32_t *beta,
                   int channels, int spatial) {
    
    for (int c = 0; c < channels; c++) {
        int32_t g = gamma[c];  // Q16
        int32_t b = beta[c];   // Q16
        
        for (int s = 0; s < spatial; s++) {
            int idx = c * spatial + s;
            // y = x * gamma + beta (Q16 arithmetic)
            int32_t val = ((int32_t)input[idx] * g) >> 16;
            val = val + (b >> 8);  // Adjust for output scale
            
            // Saturate to int8
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            output[idx] = (int8_t)val;
        }
    }
}

// ============================================================================
// P2: Element-wise Operations
// ============================================================================

void npu_add(int8_t *a, int8_t *b, int8_t *c, int len) {
    for (int i = 0; i < len; i++) {
        int32_t sum = (int32_t)a[i] + (int32_t)b[i];
        // Saturate
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        c[i] = (int8_t)sum;
    }
}

void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len) {
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len) {
    for (int i = 0; i < len; i++) {
        int32_t prod = (int32_t)a[i] * (int32_t)b[i];
        // Scale down and saturate
        prod = prod >> 7;  // Assuming Q7 output
        if (prod > 127) prod = 127;
        if (prod < -128) prod = -128;
        c[i] = (int8_t)prod;
    }
}

// ============================================================================
// P2: Requantize
// ============================================================================

void npu_requantize(int32_t *input, int8_t *output, int len,
                    int32_t scale_q16, int8_t zero_point) {
    requantize_i32_to_i8(input, output, len, scale_q16, zero_point);
}

// Simple requantize with right shift (for common case where scale is power of 2)
void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift) {
    for (int i = 0; i < len; i++) {
        int32_t v = input[i] >> shift;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        output[i] = (int8_t)v;
    }
}
