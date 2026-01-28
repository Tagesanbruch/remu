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

// Core APIs
void npu_reset(void);
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);

// DMA APIs
void npu_dma_load_feature(void *src, uint32_t len);
void npu_dma_load_weight(void *src, uint32_t len);
void npu_dma_store_output(void *dst, uint32_t len);

// GEMM API (operates on SRAM)
void npu_gemm(uint32_t m, uint32_t n, uint32_t k);

// Activation
void npu_relu(uint32_t len);

// Conv2D (im2col + GEMM)
void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type);

// Performance counters
uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);
uint32_t npu_get_gemm_count(void);
uint32_t npu_get_act_count(void);
uint32_t npu_get_dma_count(void);

#endif // __NPU_H__
