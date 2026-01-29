/**
 * NPU Operations API for REMU Platform
 * 
 * High-level neural network operators including:
 * - P0: Conv2D, GEMM, MatMul
 * - P1: DepthwiseConv2D, Pooling (Max/Avg/GlobalAvg), Activation (ReLU/LeakyReLU/Clip)
 * 
 * All operations support automatic tiling for SRAM constraints.
 */

#ifndef __NPU_OPS_H__
#define __NPU_OPS_H__

#include <stdint.h>
#include "npu_hw.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// P0 Operators: Core Compute (High Priority for RTL)
// ============================================================================

/**
 * Matrix Multiplication with automatic tiling: C[m,n] = A[m,k] @ B[k,n]
 * 
 * @param a Input matrix A, int8_t[m,k] row-major
 * @param b Input matrix B, int8_t[k,n] row-major
 * @param c Output matrix C, int32_t[m,n] row-major
 * @param m Rows of A and C
 * @param n Columns of B and C
 * @param k Columns of A and rows of B
 */
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);

/**
 * 2D Convolution using im2col + GEMM with automatic tiling
 * 
 * @param input  Input tensor, int8_t[batch, in_c, in_h, in_w]
 * @param weight Filter weights, int8_t[out_c, in_c, kh, kw]
 * @param output Output tensor, int32_t[batch, out_c, out_h, out_w]
 * @param batch  Batch size
 * @param in_c   Input channels
 * @param in_h   Input height
 * @param in_w   Input width
 * @param out_c  Output channels (number of filters)
 * @param kh     Kernel height
 * @param kw     Kernel width
 * @param pad    Padding (same for all sides)
 * @param stride Stride
 * @param act_type Activation type (ACT_NONE, ACT_RELU, etc.)
 */
void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type);

// ============================================================================
// P1 Operators: Recommended for RTL
// ============================================================================

/**
 * Depthwise Convolution (for MobileNet)
 * 
 * Each input channel is convolved with its own set of filters.
 * 
 * @param input  Input tensor, int8_t[batch, channels, in_h, in_w]
 * @param weight Filter weights, int8_t[channels, 1, kh, kw] or int8_t[channels, kh, kw]
 * @param output Output tensor, int32_t[batch, channels, out_h, out_w]
 * @param batch  Batch size
 * @param channels Number of channels (both input and output)
 * @param in_h   Input height
 * @param in_w   Input width
 * @param kh     Kernel height
 * @param kw     Kernel width
 * @param pad    Padding
 * @param stride Stride
 * @param act_type Activation type
 */
void npu_depthwise_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                          int batch, int channels, int in_h, int in_w,
                          int kh, int kw, int pad, int stride,
                          uint32_t act_type);

/**
 * Max Pooling 2D
 * 
 * @param input  Input tensor, int8_t[batch, channels, in_h, in_w]
 * @param output Output tensor, int8_t[batch, channels, out_h, out_w]
 * @param batch  Batch size
 * @param channels Number of channels
 * @param in_h   Input height
 * @param in_w   Input width
 * @param kh     Kernel height
 * @param kw     Kernel width
 * @param stride Stride
 * @param pad    Padding
 */
void npu_maxpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad);

/**
 * Average Pooling 2D
 */
void npu_avgpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad);

/**
 * Global Average Pooling 2D
 * 
 * Reduces spatial dimensions to 1x1 by averaging all values per channel.
 * 
 * @param input  Input tensor, int8_t[batch, channels, in_h, in_w]
 * @param output Output tensor, int32_t[batch, channels] (accumulated sum)
 * @param batch  Batch size
 * @param channels Number of channels
 * @param in_h   Input height
 * @param in_w   Input width
 */
void npu_global_avgpool2d(int8_t *input, int32_t *output,
                          int batch, int channels, int in_h, int in_w);

/**
 * Element-wise ReLU activation
 * 
 * @param input  Input tensor, int8_t or int32_t
 * @param output Output tensor (can be same as input for in-place)
 * @param len    Number of elements
 * @param dtype  0=int8, 1=int32
 */
void npu_relu_elementwise(void *input, void *output, int len, int dtype);

/**
 * Element-wise LeakyReLU activation
 * 
 * @param alpha  Negative slope (Q16 fixed-point, e.g., 0.1 = 6554)
 */
void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype, int32_t alpha_q16);

/**
 * Element-wise Clip activation
 */
void npu_clip_elementwise(void *input, void *output, int len, int dtype, int32_t min_val, int32_t max_val);

/**
 * Element-wise ReLU6 (Clip to [0, 6])
 */
void npu_relu6_elementwise(void *input, void *output, int len, int dtype);

// ============================================================================
// P2 Operators: Optional for RTL (Software Implementation)
// ============================================================================

/**
 * Batch Normalization (inference mode, fused)
 * y = (x - mean) * scale / sqrt(var + eps) + bias
 *   = x * gamma + beta (pre-computed)
 * 
 * @param input  Input tensor
 * @param output Output tensor
 * @param gamma  Pre-computed scale (scale / sqrt(var + eps)), Q16
 * @param beta   Pre-computed bias (bias - mean * gamma), Q16
 * @param channels Number of channels
 * @param spatial  Spatial size (H * W)
 */
void npu_batchnorm(int8_t *input, int8_t *output,
                   int32_t *gamma, int32_t *beta,
                   int channels, int spatial);

/**
 * Element-wise Add
 */
void npu_add(int8_t *a, int8_t *b, int8_t *c, int len);

/**
 * Element-wise Add (int32)
 */
void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len);

/**
 * Element-wise Multiply
 */
void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len);

/**
 * Requantize: Convert int32 accumulator to int8 with scaling
 * 
 * @param input  Input int32 tensor
 * @param output Output int8 tensor
 * @param len    Number of elements
 * @param scale  Scale factor (Q16 fixed-point)
 * @param zero_point Output zero point
 */
void npu_requantize(int32_t *input, int8_t *output, int len,
                    int32_t scale_q16, int8_t zero_point);

/**
 * Simple requantize with right shift (power-of-2 scale)
 * 
 * @param input  Input int32 tensor
 * @param output Output int8 tensor
 * @param len    Number of elements
 * @param shift  Right shift amount (output = input >> shift)
 */
void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate output dimensions for Conv2D
 */
static inline void calc_conv2d_output_size(int in_h, int in_w, int kh, int kw,
                                           int pad, int stride,
                                           int *out_h, int *out_w) {
    *out_h = (in_h + 2 * pad - kh) / stride + 1;
    *out_w = (in_w + 2 * pad - kw) / stride + 1;
}

/**
 * Calculate output dimensions for Pooling
 */
static inline void calc_pool_output_size(int in_h, int in_w, int kh, int kw,
                                         int stride, int pad,
                                         int *out_h, int *out_w) {
    *out_h = (in_h + 2 * pad - kh) / stride + 1;
    *out_w = (in_w + 2 * pad - kw) / stride + 1;
}

#ifdef __cplusplus
}
#endif

#endif // __NPU_OPS_H__
