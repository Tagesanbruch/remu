/**
 * NPU Tiling Utilities
 * 
 * Helper functions for tiling large operations to fit SRAM constraints.
 */

#include "npu_hw.h"
#include <stdint.h>

// ============================================================================
// Tiling Parameters
// ============================================================================

/**
 * Calculate optimal tile sizes for MatMul: C[M,N] = A[M,K] * B[K,N]
 * 
 * Constraints:
 * - Feature SRAM: M_tile * K <= SRAM_SIZE
 * - Weight SRAM:  K * N_tile <= SRAM_SIZE
 * - Output SRAM:  M_tile * N_tile * 4 <= SRAM_SIZE
 * 
 * @return N tile size
 */
int calc_matmul_tile_n(int m, int n, int k) {
    int max_n_by_weight = SRAM_SIZE / k;
    int max_n_by_output = SRAM_SIZE / (m * sizeof(int32_t));
    int n_tile = max_n_by_weight < max_n_by_output ? max_n_by_weight : max_n_by_output;
    if (n_tile > n) n_tile = n;
    if (n_tile < 1) n_tile = 1;
    return n_tile;
}

/**
 * Calculate optimal tile sizes for Conv2D (im2col + GEMM)
 * 
 * GEMM dimensions after im2col: [M, K] * [K, N] -> [M, N]
 * where M = out_h * out_w, K = in_c * kh * kw, N = out_c
 * 
 * @return M tile size (spatial positions per tile)
 */
int calc_conv2d_tile_m(int M_total, int K, int N) {
    int max_m_by_feature = SRAM_SIZE / K;
    int max_m_by_output = SRAM_SIZE / (N * sizeof(int32_t));
    int M_tile = max_m_by_feature < max_m_by_output ? max_m_by_feature : max_m_by_output;
    if (M_tile > M_total) M_tile = M_total;
    if (M_tile < 1) M_tile = 1;
    return M_tile;
}

/**
 * Check if tiling is needed for MatMul
 */
int needs_matmul_tiling(int m, int n, int k) {
    // Check if everything fits in SRAM
    if (m * k > (int)SRAM_SIZE) return 1;  // Feature too large
    if (k * n > (int)SRAM_SIZE) return 1;  // Weight too large
    if (m * n * (int)sizeof(int32_t) > (int)SRAM_SIZE) return 1;  // Output too large
    return 0;
}

/**
 * Check if tiling is needed for Conv2D
 */
int needs_conv2d_tiling(int out_h, int out_w, int in_c, int out_c, int kh, int kw) {
    int M = out_h * out_w;
    int K = in_c * kh * kw;
    int N = out_c;
    
    if (M * K > (int)SRAM_SIZE) return 1;  // im2col too large
    if (K * N > (int)SRAM_SIZE) return 1;  // weight too large
    if (M * N * (int)sizeof(int32_t) > (int)SRAM_SIZE) return 1;  // output too large
    return 0;
}

/**
 * Calculate tile sizes for Pooling
 */
int calc_pool_tile_channels(int channels, int in_h, int in_w) {
    int spatial = in_h * in_w;
    int max_channels = SRAM_SIZE / spatial;
    if (max_channels > channels) max_channels = channels;
    if (max_channels < 1) max_channels = 1;
    return max_channels;
}

/**
 * Calculate tile sizes for Depthwise Conv2D
 */
int calc_depthwise_conv_tile_channels(int channels, int in_h, int in_w, int kh, int kw) {
    // Input tile: channels_tile * in_h * in_w
    // Weight tile: channels_tile * kh * kw
    // Output tile: channels_tile * out_h * out_w
    int max_ch_by_input = SRAM_SIZE / (in_h * in_w);
    int max_ch_by_weight = SRAM_SIZE / (kh * kw);
    int channels_tile = max_ch_by_input < max_ch_by_weight ? max_ch_by_input : max_ch_by_weight;
    if (channels_tile > channels) channels_tile = channels;
    if (channels_tile < 1) channels_tile = 1;
    return channels_tile;
}
