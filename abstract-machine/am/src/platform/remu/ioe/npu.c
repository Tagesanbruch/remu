/**
 * NPU Driver for REMU Platform
 * 
 * Hardware Memory Map (Base: 0x2100_0000):
 *   0x0000 - 0x00FF: Control Registers
 *   0x1000 - 0x4FFF: Feature SRAM (16KB)
 *   0x5000 - 0x8FFF: Weight SRAM (16KB)
 *   0x9000 - 0xCFFF: Output SRAM (16KB)
 */

#include <am.h>
#include <klib.h>
#include ISA_H  // For outl/inl definitions

// NPU Base Address
#define NPU_BASE          0x21000000

// SRAM Offsets
#define SRAM_FEATURE      0x1000
#define SRAM_WEIGHT       0x5000
#define SRAM_OUTPUT       0x9000
#define SRAM_SIZE         0x4000  // 16KB each

// Register Offsets
#define REG_CTRL          0x00
#define REG_STATUS        0x04
#define REG_DMA_SRC       0x08
#define REG_DMA_DST       0x0C
#define REG_DMA_LEN       0x10
#define REG_DMA_DIR       0x14
#define REG_DMA_CTRL      0x18
#define REG_GEMM_M        0x20
#define REG_GEMM_N        0x24
#define REG_GEMM_K        0x28
#define REG_GEMM_CTRL     0x2C
#define REG_IM2COL_EN     0x30
#define REG_IM2COL_H      0x34
#define REG_IM2COL_W      0x38
#define REG_IM2COL_C      0x3C
#define REG_IM2COL_KH     0x40
#define REG_IM2COL_KW     0x44
#define REG_IM2COL_PAD    0x48
#define REG_IM2COL_STR    0x4C
#define REG_PERF_CYCLES   0x80
#define REG_PERF_BYTES    0x84

// DMA Direction
#define DMA_MM2S_FEATURE  0
#define DMA_MM2S_WEIGHT   1
#define DMA_S2MM_OUTPUT   2

// Status Bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DONE       (1 << 1)
#define STATUS_ERROR      (1 << 2)

// Helper macros
#define npu_reg(off)      (NPU_BASE + (off))
#define npu_feature(off)  (NPU_BASE + SRAM_FEATURE + (off))
#define npu_weight(off)   (NPU_BASE + SRAM_WEIGHT + (off))
#define npu_output(off)   (NPU_BASE + SRAM_OUTPUT + (off))

/**
 * Reset NPU and clear status
 */
void npu_reset(void) {
    outl(npu_reg(REG_CTRL), 1);  // Reset
    outl(npu_reg(REG_CTRL), 0);  // Clear reset
}

/**
 * Wait for NPU operation to complete
 */
static void npu_wait(void) {
    while (inl(npu_reg(REG_STATUS)) & STATUS_BUSY);
}

/**
 * DMA transfer from DRAM to NPU Feature SRAM
 */
void npu_dma_load_feature(void *src, uint32_t len) {
    outl(npu_reg(REG_DMA_SRC), (uintptr_t)src);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_MM2S_FEATURE);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

/**
 * DMA transfer from DRAM to NPU Weight SRAM
 */
void npu_dma_load_weight(void *src, uint32_t len) {
    outl(npu_reg(REG_DMA_SRC), (uintptr_t)src);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_MM2S_WEIGHT);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

/**
 * DMA transfer from NPU Output SRAM to DRAM
 */
void npu_dma_store_output(void *dst, uint32_t len) {
    outl(npu_reg(REG_DMA_DST), (uintptr_t)dst);
    outl(npu_reg(REG_DMA_LEN), len);
    outl(npu_reg(REG_DMA_DIR), DMA_S2MM_OUTPUT);
    outl(npu_reg(REG_DMA_CTRL), 1);
    npu_wait();
}

/**
 * Execute GEMM: C[MxN] = A[MxK] * B[KxN]
 * Data must already be in Feature/Weight SRAM
 */
void npu_gemm(uint32_t m, uint32_t n, uint32_t k) {
    outl(npu_reg(REG_GEMM_M), m);
    outl(npu_reg(REG_GEMM_N), n);
    outl(npu_reg(REG_GEMM_K), k);
    outl(npu_reg(REG_GEMM_CTRL), 1);
    npu_wait();
}

/**
 * High-level MatMul API
 * A: int8_t[m][k], B: int8_t[k][n], C: int32_t[m][n]
 */
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    // 1. Load A to Feature SRAM
    npu_dma_load_feature(a, m * k);
    
    // 2. Load B to Weight SRAM
    npu_dma_load_weight(b, k * n);
    
    // 3. Execute GEMM
    npu_gemm(m, n, k);
    
    // 4. Store C from Output SRAM
    npu_dma_store_output(c, m * n * sizeof(int32_t));
}

/**
 * Direct SRAM write (for debugging/small data)
 */
void npu_write_feature_byte(uint32_t offset, uint8_t val) {
    outb(npu_feature(offset), val);
}

void npu_write_weight_byte(uint32_t offset, uint8_t val) {
    outb(npu_weight(offset), val);
}

int32_t npu_read_output_i32(uint32_t idx) {
    return inl(npu_output(idx * 4));
}

/**
 * Get performance counters
 */
uint32_t npu_get_cycles(void) {
    return inl(npu_reg(REG_PERF_CYCLES));
}

uint32_t npu_get_mem_bytes(void) {
    return inl(npu_reg(REG_PERF_BYTES));
}

