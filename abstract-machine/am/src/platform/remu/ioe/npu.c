/**
 * NPU Driver for REMU Platform (Simplified)
 * 
 * Hardware Memory Map (Base: 0x2100_0000):
 *   0x0000 - 0x00FF: Control Registers
 *   0x1000 - 0x4FFF: Feature SRAM (16KB)
 *   0x5000 - 0x8FFF: Weight SRAM (16KB)
 *   0x9000 - 0xCFFF: Output SRAM (16KB)
 */

#include <am.h>
#include <klib.h>
#include ISA_H

// NPU Base Address
#define NPU_BASE          0x21000000

// SRAM Offsets
#define SRAM_FEATURE      0x1000
#define SRAM_WEIGHT       0x5000
#define SRAM_OUTPUT       0x9000
#define SRAM_SIZE         0x4000  // 16KB

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
#define REG_ACT_TYPE      0x50
#define REG_ACT_LEN       0x54
#define REG_ACT_CTRL      0x58
#define REG_PERF_CYCLES   0x80
#define REG_PERF_BYTES    0x84
#define REG_PERF_GEMM     0x8C
#define REG_PERF_ACT      0x90
#define REG_PERF_DMA      0x94

// DMA Direction
#define DMA_MM2S_FEATURE  0
#define DMA_MM2S_WEIGHT   1
#define DMA_S2MM_OUTPUT   2

// Activation types
#define ACT_NONE          0
#define ACT_RELU          1
#define ACT_RELU6         2

// Status
#define STATUS_BUSY       (1 << 0)

// Macros
#define npu_reg(off)      (NPU_BASE + (off))

static void npu_wait(void) {
    while (inl(npu_reg(REG_STATUS)) & STATUS_BUSY);
}

void npu_reset(void) {
    outl(npu_reg(REG_CTRL), 1);
    outl(npu_reg(REG_CTRL), 0);
}

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

void npu_gemm(uint32_t m, uint32_t n, uint32_t k) {
    outl(npu_reg(REG_GEMM_M), m);
    outl(npu_reg(REG_GEMM_N), n);
    outl(npu_reg(REG_GEMM_K), k);
    outl(npu_reg(REG_GEMM_CTRL), 1);
    npu_wait();
}

void npu_relu(uint32_t len) {
    outl(npu_reg(REG_ACT_TYPE), ACT_RELU);
    outl(npu_reg(REG_ACT_LEN), len);
    outl(npu_reg(REG_ACT_CTRL), 1);
    npu_wait();
}

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    npu_dma_load_feature(a, m * k);
    npu_dma_load_weight(b, k * n);
    npu_gemm(m, n, k);
    npu_dma_store_output(c, m * n * sizeof(int32_t));
}

/**
 * Simplified Conv2D using im2col + GEMM
 * 
 * Converts convolution to matrix multiplication:
 *   im2col(input) -> feature matrix [out_h*out_w, in_c*kh*kw]
 *   weight -> [out_c, in_c*kh*kw]
 *   GEMM output -> [out_h*out_w, out_c]
 *   Final output -> [out_c, out_h, out_w] (transposed for standard layout)
 */
void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    
    int M = out_h * out_w;  // spatial
    int N = out_c;          // filters
    int K = in_c * kh * kw; // kernel volume
    
    // Temp buffer for im2col (on stack - be careful with size!)
    int8_t im2col_buf[SRAM_SIZE];
    int32_t gemm_out[SRAM_SIZE / sizeof(int32_t)];  // Temp for GEMM output
    
    for (int b = 0; b < batch; b++) {
        int8_t *in_batch = input + b * (in_c * in_h * in_w);
        int32_t *out_batch = output + b * (out_c * out_h * out_w);
        
        // Im2col: convert to [M, K] matrix
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int m = oh * out_w + ow;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            int k_idx = ic * kh * kw + ky * kw + kx;
                            
                            int8_t val = 0;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                val = in_batch[ic * in_h * in_w + ih * in_w + iw];
                            }
                            
                            if (m * K + k_idx < (int)SRAM_SIZE) {
                                im2col_buf[m * K + k_idx] = val;
                            }
                        }
                    }
                }
            }
        }
        
        // Load im2col result
        npu_dma_load_feature(im2col_buf, M * K);
        
        // Transpose weight for GEMM: [N, K] -> [K, N]
        int8_t weight_t[SRAM_SIZE];
        for (int ni = 0; ni < N; ni++) {
            for (int ki = 0; ki < K; ki++) {
                if (ki * N + ni < (int)SRAM_SIZE) {
                    weight_t[ki * N + ni] = weight[ni * K + ki];
                }
            }
        }
        npu_dma_load_weight(weight_t, K * N);
        
        // GEMM: [M, K] * [K, N] -> [M, N] = [out_h*out_w, out_c]
        npu_gemm(M, N, K);
        
        // Apply activation if requested
        if (act_type == ACT_RELU) {
            npu_relu(M * N);
        }
        
        // Store to temp buffer first
        npu_dma_store_output(gemm_out, M * N * sizeof(int32_t));
        
        // Transpose output: [M, N] = [out_h*out_w, out_c] -> [out_c, out_h, out_w]
        for (int c = 0; c < out_c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int m = h * out_w + w;
                    out_batch[c * out_h * out_w + h * out_w + w] = gemm_out[m * N + c];
                }
            }
        }
    }
}

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
