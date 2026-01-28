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

// Core APIs
void npu_reset(void);
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);

// DMA APIs
void npu_dma_load_feature(void *src, uint32_t len);
void npu_dma_load_weight(void *src, uint32_t len);
void npu_dma_store_output(void *dst, uint32_t len);

// GEMM API (operates on SRAM)
void npu_gemm(uint32_t m, uint32_t n, uint32_t k);

// Direct SRAM access
void npu_write_feature_byte(uint32_t offset, uint8_t val);
void npu_write_weight_byte(uint32_t offset, uint8_t val);
int32_t npu_read_output_i32(uint32_t idx);

// Performance counters
uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);

#endif // __NPU_H__
