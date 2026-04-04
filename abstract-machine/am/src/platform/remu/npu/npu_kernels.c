/**
 * NPU kernel implementations.
 *
 * Covers matrix multiplication, convolution, pooling, activation,
 * batch normalization, element-wise ops, and requantization.
 */

#include "npu_hw.h"
#include "npu_ops.h"
#include <stdint.h>
#include <stdio.h>

// Tiling helpers.
extern int calc_matmul_tile_n(int m, int n, int k);
extern int calc_conv2d_tile_m(int M_total, int K, int N);
extern int needs_matmul_tiling(int m, int n, int k);
extern int needs_conv2d_tiling(int out_h, int out_w, int in_c, int out_c,
                               int kh, int kw);
extern int calc_pool_tile_channels(int channels, int in_h, int in_w);
extern int calc_depthwise_conv_tile_channels(int channels, int in_h, int in_w,
                                             int kh, int kw);

// Layout and buffer helpers.
extern void transpose_matrix_i8(int8_t *src, int8_t *dst, int m, int n);
extern void extract_weight_columns_i8(int8_t *src, int8_t *dst, int k, int n,
                                      int col_start, int col_end);
extern void scatter_output_columns_i32(int32_t *src, int32_t *dst, int m, int n,
                                       int col_start, int col_end);
extern void im2col_tile(int8_t *input, int8_t *col_buf, int in_c, int in_h,
                        int in_w, int kh, int kw, int out_w, int m_start,
                        int m_end, int pad_top, int pad_left, int stride);
extern void requantize_i32_to_i8(int32_t *input, int8_t *output, int len,
                                 int32_t scale_q16, int8_t zero_point);

// ============================================================================
// Static scratch buffers
// ============================================================================

static int8_t _matmul_weight_tile[SRAM_SIZE];
static int32_t _matmul_temp_out[SRAM_SIZE / sizeof(int32_t)];

static int8_t _conv2d_im2col_buf[SRAM_SIZE];
static int8_t _conv2d_weight_tile[SRAM_SIZE];
static int32_t _conv2d_gemm_out[SRAM_SIZE / sizeof(int32_t)];

static inline int32_t clamp_i32_local(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

static inline void decode_padding(int pad, int *pad_top, int *pad_left,
                                  int *pad_bottom, int *pad_right) {
  uint32_t raw = (uint32_t)pad;
  if (raw <= 0xFFu) {
    int p = (int)raw;
    *pad_top = p;
    *pad_left = p;
    *pad_bottom = p;
    *pad_right = p;
    return;
  }

  *pad_top = (int)(raw & 0xFFu);
  *pad_left = (int)((raw >> 8) & 0xFFu);
  *pad_bottom = (int)((raw >> 16) & 0xFFu);
  *pad_right = (int)((raw >> 24) & 0xFFu);
}

// ============================================================================
// P0: Matrix Multiplication with Tiling
// ============================================================================

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

  int m_tile = m;
  if (m * k > (int)SRAM_SIZE) {
    m_tile = SRAM_SIZE / k;
    if (m_tile < 1)
      m_tile = 1;
  }

  for (int m_start = 0; m_start < m; m_start += m_tile) {
    int m_end = m_start + m_tile;
    if (m_end > m)
      m_end = m;
    int m_cur = m_end - m_start;
    int n_tile = calc_matmul_tile_n(m_cur, n, k);

    npu_dma_load_feature(a + m_start * k, m_cur * k);

    for (int n_start = 0; n_start < n; n_start += n_tile) {
      int n_end = n_start + n_tile;
      if (n_end > n)
        n_end = n;
      int n_cur = n_end - n_start;

      extract_weight_columns_i8(b, _matmul_weight_tile, k, n, n_start, n_end);
      npu_dma_load_weight(_matmul_weight_tile, k * n_cur);
      npu_gemm(m_cur, n_cur, k);
      npu_dma_store_output(_matmul_temp_out, m_cur * n_cur * sizeof(int32_t));
      scatter_output_columns_i32(_matmul_temp_out, c + m_start * n, m_cur, n,
                                 n_start, n_end);
    }
  }
}

// ============================================================================
// P0: Conv2D with Im2col + GEMM and Tiling
// ============================================================================

void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output, int batch,
                int in_c, int in_h, int in_w, int out_c, int kh, int kw,
                int pad, int stride, uint32_t act_type) {

  if (batch <= 0 || in_c <= 0 || in_h <= 0 || in_w <= 0 || out_c <= 0 ||
      kh <= 0 || kw <= 0 || stride <= 0) {
    return;
  }

  int pad_top = 0;
  int pad_left = 0;
  int pad_bottom = 0;
  int pad_right = 0;
  decode_padding(pad, &pad_top, &pad_left, &pad_bottom, &pad_right);

  int out_h = (in_h + pad_top + pad_bottom - kh) / stride + 1;
  int out_w = (in_w + pad_left + pad_right - kw) / stride + 1;

  if (out_h <= 0 || out_w <= 0) {
    return;
  }

  int M_total = out_h * out_w;
  int N = out_c;
  int K = in_c * kh * kw;

  if (K > (int)SRAM_SIZE) {
    printf("npu_conv2d: K=%d exceeds SRAM=%u\n", K, (unsigned)SRAM_SIZE);
    return;
  }

  int M_tile = calc_conv2d_tile_m(M_total, K, N);
  if (M_tile < 1)
    M_tile = 1;

  for (int bidx = 0; bidx < batch; bidx++) {
    int8_t *in_batch = input + bidx * (in_c * in_h * in_w);
    int32_t *out_batch = output + bidx * (out_c * out_h * out_w);

    for (int m_start = 0; m_start < M_total; m_start += M_tile) {
      int m_end = m_start + M_tile;
      if (m_end > M_total)
        m_end = M_total;
      int M_cur = m_end - m_start;

      im2col_tile(in_batch, _conv2d_im2col_buf, in_c, in_h, in_w, kh, kw,
                  out_w, m_start, m_end, pad_top, pad_left, stride);
      npu_dma_load_feature(_conv2d_im2col_buf, M_cur * K);

      int n_tile_by_weight = SRAM_SIZE / K;
      int n_tile_by_output = SRAM_SIZE / (M_cur * (int)sizeof(int32_t));
      int n_tile = n_tile_by_weight < n_tile_by_output ? n_tile_by_weight
                                                        : n_tile_by_output;
      if (n_tile < 1)
        n_tile = 1;
      if (n_tile > N)
        n_tile = N;

      for (int n_start = 0; n_start < N; n_start += n_tile) {
        int n_end = n_start + n_tile;
        if (n_end > N)
          n_end = N;
        int N_cur = n_end - n_start;

        for (int ki = 0; ki < K; ki++) {
          for (int ni = 0; ni < N_cur; ni++) {
            _conv2d_weight_tile[ki * N_cur + ni] =
                weight[(n_start + ni) * K + ki];
          }
        }

        npu_dma_load_weight(_conv2d_weight_tile, K * N_cur);
        npu_gemm(M_cur, N_cur, K);
        if (act_type == ACT_RELU) {
          npu_relu(M_cur * N_cur);
        }
        npu_dma_store_output(_conv2d_gemm_out, M_cur * N_cur * sizeof(int32_t));

        for (int m_idx = 0; m_idx < M_cur; m_idx++) {
          int m = m_start + m_idx;
          int oh = m / out_w;
          int ow = m % out_w;
          for (int oc = 0; oc < N_cur; oc++) {
            out_batch[(n_start + oc) * out_h * out_w + oh * out_w + ow] =
                _conv2d_gemm_out[m_idx * N_cur + oc];
          }
        }
      }
    }
  }
}

// ============================================================================
// P1: Depthwise Convolution
// ============================================================================

void npu_depthwise_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                          int batch, int channels, int in_h, int in_w, int kh,
                          int kw, int pad, int stride, uint32_t act_type) {

  int pad_top = 0;
  int pad_left = 0;
  int pad_bottom = 0;
  int pad_right = 0;
  decode_padding(pad, &pad_top, &pad_left, &pad_bottom, &pad_right);

  int out_h = (in_h + pad_top + pad_bottom - kh) / stride + 1;
  int out_w = (in_w + pad_left + pad_right - kw) / stride + 1;
  int8_t pad_val = npu_get_input_pad_value();

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
      int8_t *wt_ch = weight + c * (kh * kw);
      int32_t *out_ch =
          output + b * (channels * out_h * out_w) + c * (out_h * out_w);

      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int32_t sum = 0;
          for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
              int ih = oh * stride - pad_top + ky;
              int iw = ow * stride - pad_left + kx;
              int32_t in_val = (int32_t)pad_val;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                in_val = (int32_t)in_ch[ih * in_w + iw];
              }
              sum += in_val * (int32_t)wt_ch[ky * kw + kx];
            }
          }

          if (act_type == ACT_RELU && sum < 0)
            sum = 0;

          out_ch[oh * out_w + ow] = sum;
        }
      }
    }
  }
}

// ============================================================================
// P1: Max Pooling 2D
// ============================================================================

void npu_maxpool2d(int8_t *input, int8_t *output, int batch, int channels,
                   int in_h, int in_w, int kh, int kw, int stride, int pad) {

  int out_h = (in_h + 2 * pad - kh) / stride + 1;
  int out_w = (in_w + 2 * pad - kw) / stride + 1;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
      int8_t *out_ch =
          output + b * (channels * out_h * out_w) + c * (out_h * out_w);

      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int8_t max_val = -128;

          for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
              int ih = oh * stride - pad + ky;
              int iw = ow * stride - pad + kx;

              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int8_t val = in_ch[ih * in_w + iw];
                if (val > max_val)
                  max_val = val;
              }
            }
          }
          out_ch[oh * out_w + ow] = max_val;
        }
      }
    }
  }
}

// ============================================================================
// P1: Average Pooling 2D
// ============================================================================

void npu_avgpool2d(int8_t *input, int8_t *output, int batch, int channels,
                   int in_h, int in_w, int kh, int kw, int stride, int pad) {

  int out_h = (in_h + 2 * pad - kh) / stride + 1;
  int out_w = (in_w + 2 * pad - kw) / stride + 1;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
      int8_t *out_ch =
          output + b * (channels * out_h * out_w) + c * (out_h * out_w);

      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int32_t sum = 0;
          int count = 0;

          for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
              int ih = oh * stride - pad + ky;
              int iw = ow * stride - pad + kx;

              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                sum += in_ch[ih * in_w + iw];
                count++;
              }
            }
          }

          if (count == 0) {
            out_ch[oh * out_w + ow] = 0;
          } else {
            int32_t avg = 0;
            if (sum >= 0) {
              avg = (sum + count / 2) / count;
            } else {
              avg = -(((-sum) + count / 2) / count);
            }
            out_ch[oh * out_w + ow] = (int8_t)avg;
          }
        }
      }
    }
  }
}

// ============================================================================
// P1: Global Average Pooling 2D
// ============================================================================

void npu_global_avgpool2d(int8_t *input, int32_t *output, int batch,
                          int channels, int in_h, int in_w) {

  int spatial = in_h * in_w;
  if (spatial <= 0)
    return;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      int8_t *in_ch = input + b * (channels * spatial) + c * spatial;
      int32_t sum = 0;

      for (int i = 0; i < spatial; i++) {
        sum += in_ch[i];
      }
      if (sum >= 0) {
        output[b * channels + c] = (sum + spatial / 2) / spatial;
      } else {
        output[b * channels + c] = -(((-sum) + spatial / 2) / spatial);
      }
    }
  }
}

// ============================================================================
// P1: Activation Functions (Element-wise)
// ============================================================================

void npu_relu_elementwise(void *input, void *output, int len, int dtype) {
  if (dtype == 0) {
    int8_t *in = (int8_t *)input;
    int8_t *out = (int8_t *)output;
    for (int i = 0; i < len; i++) {
      out[i] = (in[i] > 0) ? in[i] : 0;
    }
  } else {
    int32_t *in = (int32_t *)input;
    int32_t *out = (int32_t *)output;
    for (int i = 0; i < len; i++) {
      out[i] = (in[i] > 0) ? in[i] : 0;
    }
  }
}

void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype,
                                int32_t alpha_q16) {
  if (dtype == 0) {
    int8_t *in = (int8_t *)input;
    int8_t *out = (int8_t *)output;
    for (int i = 0; i < len; i++) {
      if (in[i] > 0) {
        out[i] = in[i];
      } else {
        int32_t scaled = ((int32_t)in[i] * alpha_q16) >> 16;
        out[i] = (int8_t)clamp_i32_local(scaled, -128, 127);
      }
    }
  } else {
    int32_t *in = (int32_t *)input;
    int32_t *out = (int32_t *)output;
    for (int i = 0; i < len; i++) {
      if (in[i] > 0) {
        out[i] = in[i];
      } else {
        out[i] = ((int64_t)in[i] * alpha_q16) >> 16;
      }
    }
  }
}

void npu_clip_elementwise(void *input, void *output, int len, int dtype,
                          int32_t min_val, int32_t max_val) {
  if (dtype == 0) {
    int8_t *in = (int8_t *)input;
    int8_t *out = (int8_t *)output;
    int8_t min8 = (int8_t)clamp_i32_local(min_val, -128, 127);
    int8_t max8 = (int8_t)clamp_i32_local(max_val, -128, 127);
    for (int i = 0; i < len; i++) {
      int8_t v = in[i];
      if (v < min8)
        v = min8;
      if (v > max8)
        v = max8;
      out[i] = v;
    }
  } else {
    int32_t *in = (int32_t *)input;
    int32_t *out = (int32_t *)output;
    for (int i = 0; i < len; i++) {
      out[i] = clamp_i32_local(in[i], min_val, max_val);
    }
  }
}

void npu_relu6_elementwise(void *input, void *output, int len, int dtype) {
  npu_clip_elementwise(input, output, len, dtype, 0, 6);
}

// ============================================================================
// P2: Batch Normalization (Inference, Fused)
// ============================================================================

void npu_batchnorm(int8_t *input, int8_t *output, int32_t *gamma,
                   int32_t *beta, int channels, int spatial) {

  for (int c = 0; c < channels; c++) {
    int32_t g = gamma[c];
    int32_t b = beta[c];

    for (int s = 0; s < spatial; s++) {
      int idx = c * spatial + s;
      int32_t val = ((int32_t)input[idx] * g) >> 16;
      val += (b >> 8);
      output[idx] = (int8_t)clamp_i32_local(val, -128, 127);
    }
  }
}

// ============================================================================
// P2: Element-wise Operations
// ============================================================================

void npu_add(int8_t *a, int8_t *b, int8_t *c, int len) {
  for (int i = 0; i < len; i++) {
    int32_t sum = (int32_t)a[i] + (int32_t)b[i];
    c[i] = (int8_t)clamp_i32_local(sum, -128, 127);
  }
}

void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len) {
  for (int i = 0; i < len; i++) {
    c[i] = a[i] + b[i];
  }
}

void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len) {
  for (int i = 0; i < len; i++) {
    int32_t prod = ((int32_t)a[i] * (int32_t)b[i]) >> 7;
    c[i] = (int8_t)clamp_i32_local(prod, -128, 127);
  }
}

// ============================================================================
// P2: Requantize
// ============================================================================

void npu_requantize(int32_t *input, int8_t *output, int len,
                    int32_t scale_q16, int8_t zero_point) {
  requantize_i32_to_i8(input, output, len, scale_q16, zero_point);
}

void npu_requantize_q31(int32_t *input, int8_t *output, int len,
                        int32_t scale_q31, int8_t zero_point) {
  for (int i = 0; i < len; i++) {
    int64_t prod = (int64_t)input[i] * (int64_t)scale_q31;
    int sign = (prod < 0) ? -1 : 1;
    uint64_t abs_prod = (prod < 0) ? (uint64_t)(-prod) : (uint64_t)prod;
    uint64_t q = abs_prod >> 31;
    uint64_t rem = abs_prod & 0x7FFFFFFFu;
    if (rem > 0x40000000u || (rem == 0x40000000u && (q & 1u))) {
      q++;
    }
    int64_t scaled = (sign > 0) ? (int64_t)q : -(int64_t)q;
    int32_t result = (int32_t)scaled + zero_point;
    output[i] = (int8_t)clamp_i32_local(result, -128, 127);
  }
}

void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift) {
  for (int i = 0; i < len; i++) {
    int32_t v = input[i] >> shift;
    output[i] = (int8_t)clamp_i32_local(v, -128, 127);
  }
}

void npu_requantize_auto(int32_t *input, int8_t *output, int len) {
  if (len <= 0)
    return;

  int32_t max_abs = 0;
  for (int i = 0; i < len; i++) {
    int32_t v = input[i];
    if (v < 0) {
      if (v == INT32_MIN) {
        v = INT32_MAX;
      } else {
        v = -v;
      }
    }
    if (v > max_abs)
      max_abs = v;
  }

  if (max_abs == 0) {
    for (int i = 0; i < len; i++)
      output[i] = 0;
    return;
  }

  int32_t divisor = (max_abs + 126) / 127;
  if (divisor < 1)
    divisor = 1;

  for (int i = 0; i < len; i++) {
    int32_t v = input[i];
    if (v >= 0) {
      v = (v + divisor / 2) / divisor;
    } else {
      v = -(((-v) + divisor / 2) / divisor);
    }
    output[i] = (int8_t)clamp_i32_local(v, -128, 127);
  }
}
