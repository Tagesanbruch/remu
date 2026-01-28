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

/**
 * Matrix multiplication with tiling support: C[m,n] = A[m,k] @ B[k,n]
 * 
 * When k*n > SRAM_SIZE (weight too large), tile along n dimension.
 */
static int32_t _matmul_temp_out[SRAM_SIZE / sizeof(int32_t)];

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    // Check if tiling needed for weight matrix
    int max_n_by_weight = SRAM_SIZE / k;
    int max_n_by_output = SRAM_SIZE / (m * sizeof(int32_t));
    int n_tile = max_n_by_weight < max_n_by_output ? max_n_by_weight : max_n_by_output;
    if (n_tile > n) n_tile = n;
    if (n_tile < 1) n_tile = 1;
    
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
        // B is stored as [k,n] row-major, so column tile is strided
        static int8_t weight_tile[SRAM_SIZE];
        for (int ki = 0; ki < k; ki++) {
            for (int ni = 0; ni < n_cur; ni++) {
                weight_tile[ki * n_cur + ni] = b[ki * n + (n_start + ni)];
            }
        }
        npu_dma_load_weight(weight_tile, k * n_cur);
        
        // GEMM: [m, k] * [k, n_cur] -> [m, n_cur]
        npu_gemm(m, n_cur, k);
        
        // Store to temp and copy to output
        npu_dma_store_output(_matmul_temp_out, m * n_cur * sizeof(int32_t));
        
        // Copy to correct columns of output C[m, n]
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n_cur; ni++) {
                c[mi * n + (n_start + ni)] = _matmul_temp_out[mi * n_cur + ni];
            }
        }
    }
}

/**
 * Simplified Conv2D using im2col + GEMM with tiling support
 * 
 * Converts convolution to matrix multiplication:
 *   im2col(input) -> feature matrix [out_h*out_w, in_c*kh*kw]
 *   weight -> [out_c, in_c*kh*kw]
 *   GEMM output -> [out_h*out_w, out_c]
 *   Final output -> [out_c, out_h, out_w] (transposed for standard layout)
 * 
 * Tiling: When M*K > SRAM_SIZE, process spatial positions in tiles.
 */

// Static buffers to avoid stack overflow
static int8_t _conv2d_im2col_buf[SRAM_SIZE];
static int8_t _conv2d_weight_t[SRAM_SIZE];
static int32_t _conv2d_gemm_out[SRAM_SIZE / sizeof(int32_t)];

void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type) {
    
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    
    int M_total = out_h * out_w;  // total spatial positions
    int N = out_c;                // filters
    int K = in_c * kh * kw;       // kernel volume
    
    // Calculate tile size based on SRAM constraints
    // Need: M_tile * K <= SRAM_SIZE (feature), K * N <= SRAM_SIZE (weight), M_tile * N * 4 <= SRAM_SIZE (output)
    int max_m_by_feature = SRAM_SIZE / K;
    int max_m_by_output = SRAM_SIZE / (N * sizeof(int32_t));
    int M_tile = max_m_by_feature < max_m_by_output ? max_m_by_feature : max_m_by_output;
    if (M_tile > M_total) M_tile = M_total;
    if (M_tile < 1) M_tile = 1;  // At least 1
    
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
            
            // Im2col for current tile: convert spatial positions [m_start, m_end) to [M_cur, K]
            for (int m_idx = 0; m_idx < M_cur; m_idx++) {
                int m = m_start + m_idx;
                int oh = m / out_w;
                int ow = m % out_w;
                
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
                            
                            _conv2d_im2col_buf[m_idx * K + k_idx] = val;
                        }
                    }
                }
            }
            
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
