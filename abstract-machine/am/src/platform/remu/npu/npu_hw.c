/**
 * NPU Hardware Abstraction Layer - Implementation
 * 
 * Low-level register access and basic hardware operations.
 */

#include "npu_hw.h"
#include <am.h>
#include <remu.h>

// ============================================================================
// Wait and Reset
// ============================================================================

void npu_wait(void) {
    while (inl(npu_reg(REG_STATUS)) & STATUS_BUSY);
}

void npu_reset(void) {
    outl(npu_reg(REG_CTRL), 1);
    outl(npu_reg(REG_CTRL), 0);
}

// ============================================================================
// DMA Operations
// ============================================================================

void npu_dma_load_feature(void *src, uint32_t len) {
    outl(npu_reg(REG_DMA_SRC), (uintptr_t)src);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_MM2S_FEATURE);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

void npu_dma_load_weight(void *src, uint32_t len) {
    outl(npu_reg(REG_DMA_SRC), (uintptr_t)src);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_MM2S_WEIGHT);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

void npu_dma_store_output(void *dst, uint32_t len) {
    outl(npu_reg(REG_DMA_DST), (uintptr_t)dst);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_S2MM_OUTPUT);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

// ============================================================================
// GEMM Operation
// ============================================================================

void npu_gemm(uint32_t m, uint32_t n, uint32_t k) {
    outl(npu_reg(REG_GEMM_M), m);
    outl(npu_reg(REG_GEMM_N), n);
    outl(npu_reg(REG_GEMM_K), k);
    outl(npu_reg(REG_GEMM_CTRL), 1);
    npu_wait();
}

// ============================================================================
// Activation Operations
// ============================================================================

void npu_activation(uint32_t act_type, uint32_t len) {
    outl(npu_reg(REG_ACT_TYPE), act_type);
    outl(npu_reg(REG_ACT_LEN), len);
    outl(npu_reg(REG_ACT_CTRL), 1);
    npu_wait();
}

void npu_relu(uint32_t len) {
    npu_activation(ACT_RELU, len);
}

void npu_leaky_relu(uint32_t len, int32_t alpha_q16) {
    outl(npu_reg(REG_ACT_PARAM0), (uint32_t)alpha_q16);
    outl(npu_reg(REG_ACT_TYPE), ACT_LEAKY_RELU);
    outl(npu_reg(REG_ACT_LEN), len);
    outl(npu_reg(REG_ACT_CTRL), 1);
    npu_wait();
}

void npu_clip(uint32_t len, int32_t min_val, int32_t max_val) {
    outl(npu_reg(REG_ACT_PARAM0), (uint32_t)min_val);
    outl(npu_reg(REG_ACT_PARAM1), (uint32_t)max_val);
    outl(npu_reg(REG_ACT_TYPE), ACT_CLIP);
    outl(npu_reg(REG_ACT_LEN), len);
    outl(npu_reg(REG_ACT_CTRL), 1);
    npu_wait();
}

// ============================================================================
// Pooling Operation (Hardware)
// ============================================================================

void npu_pooling_hw(uint32_t pool_type, uint32_t in_h, uint32_t in_w,
                    uint32_t kh, uint32_t kw, uint32_t stride) {
    outl(npu_reg(REG_POOL_TYPE), pool_type);
    outl(npu_reg(REG_POOL_IN_H), in_h);
    outl(npu_reg(REG_POOL_IN_W), in_w);
    outl(npu_reg(REG_POOL_KH), kh);
    outl(npu_reg(REG_POOL_KW), kw);
    outl(npu_reg(REG_POOL_STRIDE), stride);
    outl(npu_reg(REG_POOL_CTRL), 1);
    npu_wait();
}

// ============================================================================
// Performance Counters
// ============================================================================

uint32_t npu_get_cycles(void) {
    return inl(npu_reg(REG_PERF_CYCLES));
}

uint32_t npu_get_mem_bytes(void) {
    return inl(npu_reg(REG_PERF_BYTES));
}

uint32_t npu_get_gemm_count(void) {
    return inl(npu_reg(REG_PERF_GEMM));
}

uint32_t npu_get_act_count(void) {
    return inl(npu_reg(REG_PERF_ACT));
}

uint32_t npu_get_dma_count(void) {
    return inl(npu_reg(REG_PERF_DMA));
}
