#ifndef __NPU_H__
#define __NPU_H__

#include <stdint.h>

// NPU Base Address
#define NPU_BASE          0x21000000

// SRAM Regions
#define NPU_SRAM_FEATURE  (NPU_BASE + 0x1000)
#define NPU_SRAM_WEIGHT   (NPU_BASE + 0x5000)
#define NPU_SRAM_OUTPUT   (NPU_BASE + 0x9000)
#define NPU_SRAM_SIZE     0x4000  // 16KB each

// Activation Types
#define NPU_ACT_NONE        0
#define NPU_ACT_RELU        1
#define NPU_ACT_RELU6       2

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Hardware Control
// ============================================================================

void npu_wait(void);
void npu_reset(void);

// ============================================================================
// Performance Counters
// ============================================================================

uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);
uint32_t npu_get_gemm_count(void);
uint32_t npu_get_act_count(void);
uint32_t npu_get_dma_count(void);

// ============================================================================
// DMA APIs
// ============================================================================

void npu_dma_load_feature(void *src, uint32_t len);
void npu_dma_load_weight(void *src, uint32_t len);
void npu_dma_store_output(void *dst, uint32_t len);

// ============================================================================
// P0 Operators: Core Compute
// ============================================================================

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);

void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type);

void npu_gemm(uint32_t m, uint32_t n, uint32_t k);
void npu_relu(uint32_t len);

// ============================================================================
// P1 Operators: Additional Compute
// ============================================================================

void npu_depthwise_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                          int batch, int channels, int in_h, int in_w,
                          int kh, int kw, int pad, int stride,
                          uint32_t act_type);

void npu_maxpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad);

void npu_avgpool2d(int8_t *input, int8_t *output,
                   int batch, int channels, int in_h, int in_w,
                   int kh, int kw, int stride, int pad);

void npu_global_avgpool2d(int8_t *input, int32_t *output,
                          int batch, int channels, int in_h, int in_w);

// ============================================================================
// Activation Functions
// ============================================================================

void npu_relu_elementwise(void *input, void *output, int len, int dtype);
void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype, int32_t alpha_q16);
void npu_clip_elementwise(void *input, void *output, int len, int dtype, int32_t min_val, int32_t max_val);
void npu_relu6_elementwise(void *input, void *output, int len, int dtype);

// ============================================================================
// P2 Operators: Element-wise & Utility
// ============================================================================

void npu_add(int8_t *a, int8_t *b, int8_t *c, int len);
void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len);
void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len);

void npu_batchnorm(int8_t *input, int8_t *output,
                   int32_t *gamma, int32_t *beta,
                   int channels, int spatial);

void npu_requantize(int32_t *input, int8_t *output, int len,
                    int32_t scale_q16, int8_t zero_point);

void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift);

#ifdef __cplusplus
}
#endif

#endif // __NPU_H__
