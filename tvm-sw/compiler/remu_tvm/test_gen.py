import os
import numpy as np

def generate_test_data(onnx_model, test_dir, input_shape, input_scale):
    """Generate test input/output data for verification."""
    import onnxruntime
    from onnx import numpy_helper
    
    # Create deterministic random input
    np.random.seed(42)
    input_data = (np.random.randn(*input_shape).astype(np.float32) * 50)

    # Build initializer map for quantization parameter lookup.
    initializer_map = {}
    for init in onnx_model.graph.initializer:
        try:
            initializer_map[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    # Try to detect explicit model-input quantization:
    #   input(float) -> QuantizeLinear(scale, zero_point) -> int8 graph input
    # If present, generate test_input_data with the same quantization rule.
    input_q = None
    input_quant_scale = None
    input_quant_zero_point = None
    model_input_name = onnx_model.graph.input[0].name if onnx_model.graph.input else None
    if model_input_name:
        for node in onnx_model.graph.node:
            if node.op_type != "QuantizeLinear":
                continue
            if len(node.input) < 3 or node.input[0] != model_input_name:
                continue
            scale_name = node.input[1]
            zp_name = node.input[2]
            if scale_name in initializer_map and zp_name in initializer_map:
                scale_arr = np.asarray(initializer_map[scale_name]).reshape(-1)
                zp_arr = np.asarray(initializer_map[zp_name]).reshape(-1)
                if scale_arr.size > 0 and zp_arr.size > 0:
                    input_quant_scale = float(scale_arr[0])
                    input_quant_zero_point = int(zp_arr[0])
                    if input_quant_scale != 0.0:
                        q = np.round(input_data / input_quant_scale) + input_quant_zero_point
                        input_q = np.clip(q, -128, 127).astype(np.int8)
                        break

    # Fallback: dynamic abs-max quantization for non-qnn model inputs.
    if input_q is None:
        abs_max = max(abs(input_data.min()), abs(input_data.max()))
        qmax = 127
        input_scale = abs_max / qmax if abs_max != 0 else 1.0
        input_q = np.clip(np.round(input_data / input_scale), -128, 127).astype(np.int8)
    
    # Run ONNX Runtime inference for reference
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_data})[0]
    
    # Quantize output to INT32
    output_max = max(abs(output.min()), abs(output.max()))
    output_scale = output_max / (2**30)
    output_q = (output / output_scale).astype(np.int32)
    
    # Get top-5
    top5_idx = np.argsort(output[0])[-5:][::-1]
    
    # Generate C header with embedded data
    header_path = os.path.join(test_dir, 'test_data.h')
    with open(header_path, 'w') as f:
        f.write("/**\n * Test data for REMU verification\n */\n\n")
        f.write("#ifndef __TEST_DATA_H__\n#define __TEST_DATA_H__\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define TEST_INPUT_SIZE {input_q.size}\n")
        f.write(f"#define TEST_OUTPUT_SIZE {output.shape[1]}\n")
        f.write(f"#define EXPECTED_CLASS {top5_idx[0]}\n\n")
        
        # Embed test input array
        f.write(f"static const int8_t test_input_data[{input_q.size}] = {{\n")
        for i in range(0, input_q.size, 16):
            chunk = input_q.flat[i:min(i+16, input_q.size)]
            f.write("    " + ", ".join(f"{x:4d}" for x in chunk))
            f.write(",\n" if i + 16 < input_q.size else "\n")
        f.write("};\n\n")
        
        # Embed reference output
        f.write(f"static const int32_t test_output_ref[{output.shape[1]}] = {{\n")
        for i in range(0, output.shape[1], 8):
            chunk = output_q[0][i:min(i+8, output.shape[1])]
            f.write("    " + ", ".join(f"{x:12d}" for x in chunk))
            f.write(",\n" if i + 8 < output.shape[1] else "\n")
        f.write("};\n\n")
        
        f.write("#endif\n")
    
    print(f"  Generated test data: {header_path}")
    print(f"    Expected class: {top5_idx[0]}, Top-5: {top5_idx.tolist()}")
    if input_quant_scale is not None:
        print(
            f"    Input quantization: QuantizeLinear(scale={input_quant_scale:.8g}, "
            f"zero_point={input_quant_zero_point})"
        )
    else:
        print("    Input quantization: abs-max dynamic fallback")


def generate_test_program(test_c, model_name, input_shape):
    """Generate test program."""
    with open(test_c, 'w') as f:
        f.write(f"""/**
 * {model_name.upper()} Inference Test for REMU
 */

#ifdef HOST_NATIVE
#include \"native_compat/am.h\"
#include \"native_compat/klib.h\"
#else
#include <am.h>
#include <klib.h>
#endif

// External inference function
extern int {model_name}_inference(const int8_t* input, int32_t* output);

// Test data - included from generated header
#include "test_data/test_data.h"

// Output buffer
static int32_t output[TEST_OUTPUT_SIZE] __attribute__((aligned(4)));

// Helper: Find top-k indices
static void find_topk(const int32_t* scores, int n, int k, int* indices) {{
    for (int i = 0; i < k; i++) {{
        indices[i] = -1;
    }}
    for (int i = 0; i < n; i++) {{
        for (int j = 0; j < k; j++) {{
            if (indices[j] < 0 || scores[i] > scores[indices[j]]) {{
                for (int m = k - 1; m > j; m--) {{
                    indices[m] = indices[m - 1];
                }}
                indices[j] = i;
                break;
            }}
        }}
    }}
}}

int main() {{
    printf("=== {model_name.upper()} Inference Test ===\\n");
    printf("Input size: %d bytes\\n", TEST_INPUT_SIZE);
    printf("Output size: %d elements\\n", TEST_OUTPUT_SIZE);
    printf("\\n");
    
    // Run inference
    printf("Running inference...\\n");
    int ret = {model_name}_inference(test_input_data, output);
    if (ret != 0) {{
        printf("ERROR: Inference failed with code %d\\n", ret);
        return 1;
    }}
    printf("Inference completed!\\n\\n");

    // Basic output stats
    int32_t min_val = output[0];
    int32_t max_val = output[0];
    int nonzero = 0;
    int64_t max_abs_diff = 0;
    for (int i = 0; i < TEST_OUTPUT_SIZE; i++) {{
        int32_t v = output[i];
        if (v != 0) nonzero++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        int64_t diff = (int64_t)v - (int64_t)test_output_ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_abs_diff) max_abs_diff = diff;
    }}
    printf("Output stats: min=%d max=%d nonzero=%d/%d\\n", min_val, max_val, nonzero, TEST_OUTPUT_SIZE);
    printf("Max abs diff vs ref: %ld\\n\\n", (long)max_abs_diff);
    
    // Find top-5 predictions
    int top5[5];
    find_topk(output, TEST_OUTPUT_SIZE, 5, top5);
    
    printf("Top-5 Predictions:\\n");
    for (int i = 0; i < 5; i++) {{
        printf("  #%d: Class %d (score: %d)\\n", 
               i + 1, top5[i], output[top5[i]]);
    }}
    printf("\\n");
    
    // Compare with expected
    printf("Expected: Class %d\\n", EXPECTED_CLASS);
    printf("Got:      Class %d\\n", top5[0]);
    
    if (top5[0] == EXPECTED_CLASS) {{
        printf("\\n✓ PASS: Top-1 prediction matches!\\n");
        return 0;
    }} else {{
        printf("\\n✗ FAIL: Top-1 mismatch\\n");
        return 1;
    }}
}}
""")
    print(f"  Generated test program: {test_c}")


def generate_makefile(makefile_path, model_name):
    """Generate Makefile for building and running."""
    with open(makefile_path, 'w') as f:
        f.write(f"""# {model_name.upper()} Inference Test for REMU
# Build: make ARCH=riscv32-remu BATCH=1 run

NAME = {model_name}
SRCS = {model_name}_weights.c {model_name}_inference.c test_{model_name}.c
LIBS = klib

include $(REMU_AM_HOME)/Makefile
""")
    print(f"  Generated Makefile: {makefile_path}")


def generate_native_runtime(output_dir, model_name):
        """Generate a lightweight host-native runtime and compatibility headers."""
        compat_dir = os.path.join(output_dir, "native_compat")
        os.makedirs(compat_dir, exist_ok=True)

        am_h = os.path.join(compat_dir, "am.h")
        with open(am_h, "w") as f:
                f.write("""#ifndef __REMU_NATIVE_AM_H__
#define __REMU_NATIVE_AM_H__

#include <stdint.h>
#include <stddef.h>

static inline void ioe_init(void) {}

#endif
""")

        klib_h = os.path.join(compat_dir, "klib.h")
        with open(klib_h, "w") as f:
                f.write("""#ifndef __REMU_NATIVE_KLIB_H__
#define __REMU_NATIVE_KLIB_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif
""")

        npu_h = os.path.join(compat_dir, "npu.h")
        with open(npu_h, "w") as f:
                f.write("""#ifndef __REMU_NATIVE_NPU_H__
#define __REMU_NATIVE_NPU_H__

#include <stdint.h>

#define NPU_ACT_NONE 0
#define NPU_ACT_RELU 1
#define NPU_ACT_LEAKY_RELU 2
#define NPU_ACT_RELU6 3

void npu_reset(void);
uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);
uint32_t npu_get_gemm_count(void);
uint32_t npu_get_act_count(void);
uint32_t npu_get_dma_count(void);

#endif
""")

        npu_ops_h = os.path.join(compat_dir, "npu_ops.h")
        with open(npu_ops_h, "w") as f:
                f.write("""#ifndef __REMU_NATIVE_NPU_OPS_H__
#define __REMU_NATIVE_NPU_OPS_H__

#include <stdint.h>

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);
void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                                int batch, int in_c, int in_h, int in_w,
                                int out_c, int kh, int kw, int pad, int stride,
                                uint32_t act_type);
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
void npu_relu_elementwise(void *input, void *output, int len, int dtype);
void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype, int32_t alpha_q16);
void npu_clip_elementwise(void *input, void *output, int len, int dtype, int32_t min_val, int32_t max_val);
void npu_relu6_elementwise(void *input, void *output, int len, int dtype);
void npu_batchnorm(int8_t *input, int8_t *output,
                                     int32_t *gamma, int32_t *beta,
                                     int channels, int spatial);
void npu_add(int8_t *a, int8_t *b, int8_t *c, int len);
void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len);
void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len);
void npu_requantize(int32_t *input, int8_t *output, int len,
                                        int32_t scale_q16, int8_t zero_point);
void npu_requantize_q31(int32_t *input, int8_t *output, int len,
                                            int32_t scale_q31, int8_t zero_point);
void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift);
void npu_requantize_auto(int32_t *input, int8_t *output, int len);
void npu_set_input_pad_value(int8_t pad_value);

#endif
""")

        runtime_c = os.path.join(output_dir, "native_runtime.c")
        with open(runtime_c, "w") as f:
                f.write("""#include \"native_compat/npu.h\"
#include \"native_compat/npu_ops.h\"

static uint64_t g_cycles = 0;
static uint64_t g_mem_bytes = 0;
static uint64_t g_gemm_cnt = 0;
static uint64_t g_act_cnt = 0;
static uint64_t g_dma_cnt = 0;
static int8_t g_input_pad_value = 0;

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline void decode_padding(int pad, int *pt, int *pl, int *pb, int *pr) {
    uint32_t raw = (uint32_t)pad;
    if (raw <= 0xFFu) {
        int p = (int)raw;
        *pt = p; *pl = p; *pb = p; *pr = p;
        return;
    }
    *pt = (int)(raw & 0xFFu);
    *pl = (int)((raw >> 8) & 0xFFu);
    *pb = (int)((raw >> 16) & 0xFFu);
    *pr = (int)((raw >> 24) & 0xFFu);
}

void npu_reset(void) {
    g_cycles = 0;
    g_mem_bytes = 0;
    g_gemm_cnt = 0;
    g_act_cnt = 0;
    g_dma_cnt = 0;
}

uint32_t npu_get_cycles(void) { return (uint32_t)g_cycles; }
uint32_t npu_get_mem_bytes(void) { return (uint32_t)g_mem_bytes; }
uint32_t npu_get_gemm_count(void) { return (uint32_t)g_gemm_cnt; }
uint32_t npu_get_act_count(void) { return (uint32_t)g_act_cnt; }
uint32_t npu_get_dma_count(void) { return (uint32_t)g_dma_cnt; }

void npu_set_input_pad_value(int8_t pad_value) {
    g_input_pad_value = pad_value;
}

void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            for (int p = 0; p < k; p++) {
                sum += (int32_t)a[i * k + p] * (int32_t)b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    g_gemm_cnt += (uint64_t)m * (uint64_t)n;
    g_cycles += (uint64_t)m * (uint64_t)n * (uint64_t)k;
}

void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                                int batch, int in_c, int in_h, int in_w,
                                int out_c, int kh, int kw, int pad, int stride,
                                uint32_t act_type) {
    int pt, pl, pb, pr;
    decode_padding(pad, &pt, &pl, &pb, &pr);
    int out_h = (in_h + pt + pb - kh) / stride + 1;
    int out_w = (in_w + pl + pr - kw) / stride + 1;

    for (int b = 0; b < batch; b++) {
        int8_t *in_b = input + b * (in_c * in_h * in_w);
        int32_t *out_b = output + b * (out_c * out_h * out_w);
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int ih = oh * stride - pt + ky;
                                int iw = ow * stride - pl + kx;
                                int32_t in_val = (int32_t)g_input_pad_value;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ic * in_h * in_w + ih * in_w + iw;
                                    in_val = (int32_t)in_b[in_idx];
                                }
                                int w_idx = ((oc * in_c + ic) * kh + ky) * kw + kx;
                                sum += in_val * (int32_t)weight[w_idx];
                            }
                        }
                    }
                    if (act_type == NPU_ACT_RELU && sum < 0) sum = 0;
                    out_b[oc * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }

    g_gemm_cnt += (uint64_t)batch * (uint64_t)out_c * (uint64_t)out_h * (uint64_t)out_w;
    g_cycles += (uint64_t)batch * (uint64_t)out_c * (uint64_t)out_h * (uint64_t)out_w *
                            (uint64_t)in_c * (uint64_t)kh * (uint64_t)kw;
}

void npu_depthwise_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                                                    int batch, int channels, int in_h, int in_w,
                                                    int kh, int kw, int pad, int stride,
                                                    uint32_t act_type) {
    int pt, pl, pb, pr;
    decode_padding(pad, &pt, &pl, &pb, &pr);
    int out_h = (in_h + pt + pb - kh) / stride + 1;
    int out_w = (in_w + pl + pr - kw) / stride + 1;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *w_ch = weight + c * (kh * kw);
            int32_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pt + ky;
                            int iw = ow * stride - pl + kx;
                            int32_t in_val = (int32_t)g_input_pad_value;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                in_val = (int32_t)in_ch[ih * in_w + iw];
                            }
                            sum += in_val * (int32_t)w_ch[ky * kw + kx];
                        }
                    }
                    if (act_type == NPU_ACT_RELU && sum < 0) sum = 0;
                    out_ch[oh * out_w + ow] = sum;
                }
            }
        }
    }

    g_gemm_cnt += (uint64_t)batch * (uint64_t)channels * (uint64_t)out_h * (uint64_t)out_w;
}

void npu_maxpool2d(int8_t *input, int8_t *output,
                                     int batch, int channels, int in_h, int in_w,
                                     int kh, int kw, int stride, int pad) {
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int8_t mx = -128;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                            int8_t v = in_ch[ih * in_w + iw];
                            if (v > mx) mx = v;
                        }
                    }
                    out_ch[oh * out_w + ow] = mx;
                }
            }
        }
    }
}

void npu_avgpool2d(int8_t *input, int8_t *output,
                                     int batch, int channels, int in_h, int in_w,
                                     int kh, int kw, int stride, int pad) {
    int out_h = (in_h + 2 * pad - kh) / stride + 1;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * in_h * in_w) + c * (in_h * in_w);
            int8_t *out_ch = output + b * (channels * out_h * out_w) + c * (out_h * out_w);
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    int cnt = 0;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int ih = oh * stride - pad + ky;
                            int iw = ow * stride - pad + kx;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                            sum += in_ch[ih * in_w + iw];
                            cnt++;
                        }
                    }
                    if (cnt == 0) {
                        out_ch[oh * out_w + ow] = 0;
                    } else {
                        int32_t avg = 0;
                        if (sum >= 0) avg = (sum + cnt / 2) / cnt;
                        else avg = -(((-sum) + cnt / 2) / cnt);
                        out_ch[oh * out_w + ow] = (int8_t)avg;
                    }
                }
            }
        }
    }
}

void npu_global_avgpool2d(int8_t *input, int32_t *output,
                                                    int batch, int channels, int in_h, int in_w) {
    int spatial = in_h * in_w;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int8_t *in_ch = input + b * (channels * spatial) + c * spatial;
            int32_t sum = 0;
            for (int i = 0; i < spatial; i++) sum += in_ch[i];
            if (sum >= 0) output[b * channels + c] = (sum + spatial / 2) / spatial;
            else output[b * channels + c] = -(((-sum) + spatial / 2) / spatial);
        }
    }
}

void npu_relu_elementwise(void *input, void *output, int len, int dtype) {
    if (dtype == 0) {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int i = 0; i < len; i++) out[i] = in[i] > 0 ? in[i] : 0;
    } else {
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) out[i] = in[i] > 0 ? in[i] : 0;
    }
    g_act_cnt += (uint64_t)len;
}

void npu_leaky_relu_elementwise(void *input, void *output, int len, int dtype, int32_t alpha_q16) {
    if (dtype == 0) {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int i = 0; i < len; i++) {
            if (in[i] > 0) out[i] = in[i];
            else out[i] = (int8_t)clamp_i32(((int32_t)in[i] * alpha_q16) >> 16, -128, 127);
        }
    } else {
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) {
            if (in[i] > 0) out[i] = in[i];
            else out[i] = (int32_t)(((int64_t)in[i] * alpha_q16) >> 16);
        }
    }
    g_act_cnt += (uint64_t)len;
}

void npu_clip_elementwise(void *input, void *output, int len, int dtype, int32_t min_val, int32_t max_val) {
    if (dtype == 0) {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        int8_t lo = (int8_t)clamp_i32(min_val, -128, 127);
        int8_t hi = (int8_t)clamp_i32(max_val, -128, 127);
        for (int i = 0; i < len; i++) {
            int8_t v = in[i];
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            out[i] = v;
        }
    } else {
        int32_t *in = (int32_t *)input;
        int32_t *out = (int32_t *)output;
        for (int i = 0; i < len; i++) out[i] = clamp_i32(in[i], min_val, max_val);
    }
    g_act_cnt += (uint64_t)len;
}

void npu_relu6_elementwise(void *input, void *output, int len, int dtype) {
    npu_clip_elementwise(input, output, len, dtype, 0, 6);
}

void npu_batchnorm(int8_t *input, int8_t *output,
                                     int32_t *gamma, int32_t *beta,
                                     int channels, int spatial) {
    for (int c = 0; c < channels; c++) {
        int32_t g = gamma[c];
        int32_t b = beta[c];
        for (int s = 0; s < spatial; s++) {
            int idx = c * spatial + s;
            int32_t val = ((int32_t)input[idx] * g) >> 16;
            val += (b >> 8);
            output[idx] = (int8_t)clamp_i32(val, -128, 127);
        }
    }
}

void npu_add(int8_t *a, int8_t *b, int8_t *c, int len) {
    for (int i = 0; i < len; i++) c[i] = (int8_t)clamp_i32((int32_t)a[i] + (int32_t)b[i], -128, 127);
}

void npu_add_i32(int32_t *a, int32_t *b, int32_t *c, int len) {
    for (int i = 0; i < len; i++) c[i] = a[i] + b[i];
}

void npu_mul(int8_t *a, int8_t *b, int8_t *c, int len) {
    for (int i = 0; i < len; i++) {
        int32_t v = ((int32_t)a[i] * (int32_t)b[i]) >> 7;
        c[i] = (int8_t)clamp_i32(v, -128, 127);
    }
}

void npu_requantize(int32_t *input, int8_t *output, int len,
                                        int32_t scale_q16, int8_t zero_point) {
    for (int i = 0; i < len; i++) {
        int64_t prod = (int64_t)input[i] * scale_q16;
        int sign = (prod < 0) ? -1 : 1;
        uint64_t abs_prod = (prod < 0) ? (uint64_t)(-prod) : (uint64_t)prod;
        uint64_t q = abs_prod >> 16;
        uint64_t rem = abs_prod & 0xFFFFu;
        if (rem > 0x8000u || (rem == 0x8000u && (q & 1u))) {
            q++;
        }
        int64_t scaled = (sign > 0) ? (int64_t)q : -(int64_t)q;
        int32_t result = (int32_t)scaled + zero_point;
        output[i] = (int8_t)clamp_i32(result, -128, 127);
    }
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
        output[i] = (int8_t)clamp_i32(result, -128, 127);
    }
}

void npu_requantize_shift(int32_t *input, int8_t *output, int len, int shift) {
    for (int i = 0; i < len; i++) {
        int32_t v = input[i] >> shift;
        output[i] = (int8_t)clamp_i32(v, -128, 127);
    }
}

void npu_requantize_auto(int32_t *input, int8_t *output, int len) {
    if (len <= 0) return;
    int32_t max_abs = 0;
    for (int i = 0; i < len; i++) {
        int32_t v = input[i];
        if (v == INT32_MIN) v = INT32_MAX;
        if (v < 0) v = -v;
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0) {
        for (int i = 0; i < len; i++) output[i] = 0;
        return;
    }
    int32_t div = (max_abs + 126) / 127;
    if (div < 1) div = 1;
    for (int i = 0; i < len; i++) {
        int32_t v = input[i];
        if (v >= 0) v = (v + div / 2) / div;
        else v = -(((-v) + div / 2) / div);
        output[i] = (int8_t)clamp_i32(v, -128, 127);
    }
}
""")

        print(f"  Generated native runtime: {runtime_c}")


def generate_native_makefile(makefile_path, model_name):
        """Generate host-native Makefile."""
        with open(makefile_path, "w") as f:
                f.write(f"""# {model_name.upper()} Host-native validation build

CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra

NAME = {model_name}_native
SRCS = {model_name}_weights.c {model_name}_inference.c test_{model_name}.c native_runtime.c
INCLUDES = -I. -Inative_compat

.PHONY: all run clean

all: $(NAME)

$(NAME): $(SRCS)
	$(CC) $(CFLAGS) -DHOST_NATIVE $(INCLUDES) -o $@ $(SRCS)

run: $(NAME)
	./$(NAME)

clean:
	rm -f $(NAME)
""")
        print(f"  Generated native Makefile: {makefile_path}")


def generate_compare_script(script_path):
        """Generate helper script for native/remu side-by-side validation."""
        with open(script_path, "w") as f:
                f.write("""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"
cd \"$SCRIPT_DIR\"

echo "[1/2] Running host-native"
set +e
make -f Makefile.native run | tee native_run.log
native_ec=$?
set -e

echo "[2/2] Running REMU (BATCH=1)"
set +e
make ARCH=riscv32-remu BATCH=1 run | tee remu_run.log
remu_ec=$?
set -e

extract_class() {
    local key="$1"
    local file="$2"
    grep -E "^${key}" "$file" | tail -n1 | awk '{print $3}'
}

native_expected=$(extract_class 'Expected:' native_run.log || true)
native_top1=$(extract_class 'Got:' native_run.log || true)
remu_expected=$(extract_class 'Expected:' remu_run.log || true)
remu_top1=$(extract_class 'Got:' remu_run.log || true)

native_pass=0
remu_pass=0
if [[ -n "${native_expected:-}" && -n "${native_top1:-}" && "${native_expected}" == "${native_top1}" ]]; then
    native_pass=1
fi
if [[ -n "${remu_expected:-}" && -n "${remu_top1:-}" && "${remu_expected}" == "${remu_top1}" ]]; then
    remu_pass=1
fi

echo ""
echo "Native Expected: ${native_expected:-N/A}"
echo "Native Top-1: ${native_top1:-N/A}"
echo "REMU   Expected: ${remu_expected:-N/A}"
echo "REMU   Top-1: ${remu_top1:-N/A}"

if [[ -n "${native_top1:-}" && -n "${remu_top1:-}" && "${native_top1}" == "${remu_top1}" ]]; then
    echo "Native and REMU Top-1 are aligned."
else
    echo "Native and REMU Top-1 differ."
fi

if [[ ${native_ec} -ne 0 ]]; then
    echo "Native run exited with code ${native_ec}."
fi
if [[ ${remu_ec} -ne 0 ]]; then
    echo "REMU run exited with code ${remu_ec}."
fi

if [[ ${native_pass} -eq 1 && ${remu_pass} -eq 1 && -n "${native_top1:-}" && -n "${remu_top1:-}" && "${native_top1}" == "${remu_top1}" ]]; then
    echo "Both native and REMU match expected class."
    exit 0
fi

echo "Validation failed: expected-class check and/or native-remu alignment did not pass."
exit 1
""")
        os.chmod(script_path, 0o755)
        print(f"  Generated compare script: {script_path}")
