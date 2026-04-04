/**
 * NPU Hardware Abstraction Layer for REMU Platform
 *
 * Low-level register definitions and DMA control.
 *
 * Hardware Memory Map (Base: 0x2100_0000):
 *   0x0000 - 0x00FF: Control Registers
 *   0x1000 - 0x4FFF: Feature SRAM (16KB)
 *   0x5000 - 0x8FFF: Weight SRAM (16KB)
 *   0x9000 - 0xCFFF: Output SRAM (16KB)
 */

#ifndef __NPU_HW_H__
#define __NPU_HW_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NPU Base Address and Memory Map
// ============================================================================

#define NPU_BASE 0x21000000

// SRAM Offsets and Sizes
#define SRAM_FEATURE 0x1000
#define SRAM_WEIGHT 0x5000
#define SRAM_OUTPUT 0x9000
#define SRAM_SIZE 0x4000 // 16KB each

// ============================================================================
// Register Offsets
// ============================================================================

// Control Registers
#define REG_CTRL 0x00
#define REG_STATUS 0x04

// DMA Registers
#define REG_DMA_SRC 0x08
#define REG_DMA_DST 0x0C
#define REG_DMA_LEN 0x10
#define REG_DMA_DIR 0x14
#define REG_DMA_CTRL 0x18

// GEMM Registers
#define REG_GEMM_M 0x20
#define REG_GEMM_N 0x24
#define REG_GEMM_K 0x28
#define REG_GEMM_CTRL 0x2C
#define REG_GEMM_A_OFFSET 0x30
#define REG_GEMM_B_OFFSET 0x34
#define REG_GEMM_C_OFFSET 0x38

// Pooling Registers 
#define REG_POOL_TYPE 0xF0
#define REG_POOL_IN_H 0xF4
#define REG_POOL_IN_W 0xF8
#define REG_POOL_KH 0xFC
#define REG_POOL_KW 0x100
#define REG_POOL_STRIDE 0x104
#define REG_POOL_CTRL 0x108

// Activation Registers
#define REG_ACT_TYPE 0x50
#define REG_ACT_LEN 0x54
#define REG_ACT_CTRL 0x58
#define REG_ACT_PARAM0 0x5C // For LeakyReLU alpha (Q16)
#define REG_ACT_PARAM1 0x60 // For Clip min/max

// Performance Counters
#define REG_PERF_CYCLES 0x80
#define REG_PERF_BYTES 0x84
#define REG_PERF_GEMM 0x8C
#define REG_PERF_ACT 0x90
#define REG_PERF_DMA 0x94

// Im2Col Registers
#define REG_IM2COL_CTRL 0xA0
#define REG_IM2COL_SRC_OFF 0xA4
#define REG_IM2COL_DST_OFF 0xA8
#define REG_IM2COL_IN_HW 0xAC
#define REG_IM2COL_KER_HW 0xB0
#define REG_IM2COL_CHANNELS 0xB4
#define REG_IM2COL_STRIDE 0xB8
#define REG_IM2COL_PADDING 0xBC
#define REG_IM2COL_DILATION 0xC0

// Transposer Registers
#define REG_TRANS_CTRL 0xD0
#define REG_TRANS_SRC_OFF 0xD4
#define REG_TRANS_DST_OFF 0xD8
#define REG_TRANS_DIMS 0xDC
#define REG_TRANS_ELEM_SIZE 0xE0

// ============================================================================
// DMA Direction
// ============================================================================

#define DMA_MM2S_FEATURE 0
#define DMA_MM2S_WEIGHT 1
#define DMA_S2MM_OUTPUT 2

// ============================================================================
// Activation Types
// ============================================================================

#define ACT_NONE 0
#define ACT_RELU 1
#define ACT_RELU6 2
#define ACT_LEAKY_RELU 3
#define ACT_CLIP 4

// ============================================================================
// Pooling Types
// ============================================================================

#define POOL_MAX 0
#define POOL_AVG 1
#define POOL_GLOBAL_AVG 2

// ============================================================================
// Status Bits
// ============================================================================

#define STATUS_BUSY (1 << 0)

// ============================================================================
// Macros
// ============================================================================

#define npu_reg(off) (NPU_BASE + (off))

// ============================================================================
// Low-level Hardware Functions
// ============================================================================

/**
 * Wait until NPU is not busy
 */
void npu_wait(void);

/**
 * Reset NPU
 */
void npu_reset(void);

/**
 * DMA: Load data to Feature SRAM
 */
void npu_dma_load_feature(void *src, uint32_t len);

/**
 * DMA: Load data to Weight SRAM
 */
void npu_dma_load_weight(void *src, uint32_t len);

/**
 * DMA: Store data from Output SRAM
 */
void npu_dma_store_output(void *dst, uint32_t len);

/**
 * Execute GEMM: C[M,N] = A[M,K] * B[K,N]
 * Data must be loaded to SRAM first
 */
void npu_gemm(uint32_t m, uint32_t n, uint32_t k);

/**
 * Execute Activation on Output SRAM (in-place)
 * @param act_type ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_CLIP
 * @param len Number of elements
 */
void npu_activation(uint32_t act_type, uint32_t len);

/**
 * Execute ReLU on Output SRAM (shorthand)
 */
void npu_relu(uint32_t len);

/**
 * Execute LeakyReLU on Output SRAM
 * @param alpha Slope for negative values (Q16 fixed-point, e.g., 0.1 = 6554)
 */
void npu_leaky_relu(uint32_t len, int32_t alpha_q16);

/**
 * Execute Clip on Output SRAM
 * @param min Minimum value
 * @param max Maximum value
 */
void npu_clip(uint32_t len, int32_t min_val, int32_t max_val);

/**
 * Execute Pooling
 * @param pool_type POOL_MAX, POOL_AVG, POOL_GLOBAL_AVG
 * Data must be loaded to Feature SRAM first
 */
void npu_pooling_hw(uint32_t pool_type, uint32_t in_h, uint32_t in_w,
                    uint32_t kh, uint32_t kw, uint32_t stride);

/**
 * Execute HW Im2Col
 */
void npu_hw_im2col(uint32_t src_offset, uint32_t dst_offset, int input_h,
                   int input_w, int channels, int kernel_h, int kernel_w,
                   int pad_top, int pad_left, int stride_h, int stride_w,
                   int dilation_h, int dilation_w);

/**
 * Execute HW Transpose
 */
void npu_hw_transpose(uint32_t src_offset, uint32_t dst_offset, int rows,
                      int cols, int elem_size);

/**
 * Execute GEMM with Offsets
 */
void npu_gemm_with_offset(uint32_t m, uint32_t n, uint32_t k, uint32_t a_off,
                          uint32_t b_off, uint32_t c_off);

// ============================================================================
// Performance Counters
// ============================================================================

uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);
uint32_t npu_get_gemm_count(void);
uint32_t npu_get_act_count(void);
uint32_t npu_get_dma_count(void);

#ifdef __cplusplus
}
#endif

#endif // __NPU_HW_H__
