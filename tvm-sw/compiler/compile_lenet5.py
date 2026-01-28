#!/usr/bin/env python3
"""
Compile trained LeNet-5 ONNX model for REMU NPU.
Generates:
  - model_weights.h: Quantized weights as C arrays
  - lenet5_inference.c: Inference code using NPU API
"""

import os
import numpy as np
import onnx
from onnx import numpy_helper

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
ONNX_PATH = os.path.join(OUTPUT_DIR, 'lenet5.onnx')

# Target directory for generated files (will be linked/included by am-kernels)
TARGET_DIR = os.path.join(SCRIPT_DIR, 'output')


def quantize_weights(weights: np.ndarray, name: str) -> tuple:
    """Quantize float32 weights to int8 (symmetric)."""
    abs_max = max(abs(weights.min()), abs(weights.max()))
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    print(f"  {name}: shape={weights.shape}, scale={scale:.6f}, range=[{weights.min():.4f}, {weights.max():.4f}]")
    return quantized, scale


def main():
    print(f"Loading ONNX model from {ONNX_PATH}")
    model = onnx.load(ONNX_PATH)
    
    # Extract weights
    weights = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        weights[init.name] = arr
    
    print(f"\nFound {len(weights)} weight tensors:")
    for name, arr in weights.items():
        print(f"  {name}: {arr.shape}, dtype={arr.dtype}")
    
    # Quantize all weights
    print("\nQuantizing weights...")
    quantized_weights = {}
    scales = {}
    
    for name, arr in weights.items():
        q, s = quantize_weights(arr, name)
        quantized_weights[name] = q
        scales[name] = s
    
    # Generate model_weights.h
    print(f"\nGenerating model_weights.h...")
    
    header_lines = [
        "// Auto-generated LeNet-5 weights (quantized to int8)",
        "// Generated from ONNX model",
        "#ifndef __MODEL_WEIGHTS_H__",
        "#define __MODEL_WEIGHTS_H__",
        "",
        "#include <stdint.h>",
        "",
        "// Quantization scales (for dequantization if needed)",
    ]
    
    for name, scale in scales.items():
        safe_name = name.replace(".", "_")
        header_lines.append(f"#define SCALE_{safe_name.upper()} {scale}f")
    
    header_lines.append("")
    
    # Generate weight arrays
    for name, arr in quantized_weights.items():
        safe_name = name.replace(".", "_")
        flat = arr.flatten()
        
        header_lines.append(f"// {name}: shape={list(arr.shape)}")
        header_lines.append(f"static const int8_t weight_{safe_name}[{len(flat)}] = {{")
        
        # Write in rows of 16
        for i in range(0, len(flat), 16):
            row = flat[i:i+16]
            row_str = ", ".join(f"{v:4d}" for v in row)
            header_lines.append(f"    {row_str},")
        
        header_lines.append("};")
        header_lines.append("")
    
    header_lines.append("#endif // __MODEL_WEIGHTS_H__")
    
    header_path = os.path.join(TARGET_DIR, 'model_weights.h')
    with open(header_path, 'w') as f:
        f.write('\n'.join(header_lines))
    print(f"Saved weights to {header_path}")
    
    # Generate inference code
    print(f"\nGenerating lenet5_inference.c...")
    generate_inference_code(model, quantized_weights, scales)
    
    print("\nDone!")


def generate_inference_code(model, weights, scales):
    """Generate C inference code for REMU NPU."""
    
    # Analyze model structure
    print("\nModel structure:")
    for node in model.graph.node:
        inputs = ", ".join(node.input[:2])  # First 2 inputs
        print(f"  {node.op_type}: {node.name} <- ({inputs})")
    
    code = '''/**
 * LeNet-5 Inference for REMU NPU
 * Auto-generated from ONNX model
 * 
 * Architecture:
 *   Input: 1x28x28 (int8, normalized)
 *   Conv1: 6x28x28 -> Pool -> 6x14x14
 *   Conv2: 16x10x10 -> Pool -> 16x5x5
 *   FC1: 400 -> 120
 *   FC2: 120 -> 84
 *   FC3: 84 -> 10
 */

#include <am.h>
#include <klib.h>
#include "npu.h"
#include "model_weights.h"
#include "test_images.h"

// Layer dimensions
#define INPUT_H     28
#define INPUT_W     28
#define INPUT_C     1

#define CONV1_OUT_C 6
#define CONV1_K     5
#define CONV1_OUT_H 28  // With padding=2
#define CONV1_OUT_W 28
#define POOL1_H     14
#define POOL1_W     14

#define CONV2_OUT_C 16
#define CONV2_K     5
#define CONV2_OUT_H 10
#define CONV2_OUT_W 10
#define POOL2_H     5
#define POOL2_W     5

#define FC1_IN      (CONV2_OUT_C * POOL2_H * POOL2_W)  // 400
#define FC1_OUT     120
#define FC2_OUT     84
#define FC3_OUT     10

// Intermediate buffers
static int32_t conv1_out[CONV1_OUT_C * CONV1_OUT_H * CONV1_OUT_W];
static int8_t pool1_out[CONV1_OUT_C * POOL1_H * POOL1_W];
static int32_t conv2_out[CONV2_OUT_C * CONV2_OUT_H * CONV2_OUT_W];
static int8_t pool2_out[CONV2_OUT_C * POOL2_H * POOL2_W];
static int8_t flatten_buf[FC1_IN];
static int32_t fc1_out[FC1_OUT];
static int8_t fc1_out_q[FC1_OUT];
static int32_t fc2_out[FC2_OUT];
static int8_t fc2_out_q[FC2_OUT];
static int32_t fc3_out[FC3_OUT];

// ReLU on int32 array
static void relu_i32(int32_t *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

// Max pooling 2x2, int32 -> int8 with quantization
static void maxpool2x2(int32_t *in, int8_t *out, int c, int h, int w, int scale_shift) {
    int oh = h / 2;
    int ow = w / 2;
    
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int base = ch * h * w;
                int32_t v0 = in[base + (y*2)*w + (x*2)];
                int32_t v1 = in[base + (y*2)*w + (x*2+1)];
                int32_t v2 = in[base + (y*2+1)*w + (x*2)];
                int32_t v3 = in[base + (y*2+1)*w + (x*2+1)];
                
                int32_t maxv = v0;
                if (v1 > maxv) maxv = v1;
                if (v2 > maxv) maxv = v2;
                if (v3 > maxv) maxv = v3;
                
                // Quantize
                int32_t q = maxv >> scale_shift;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out[ch * oh * ow + y * ow + x] = (int8_t)q;
            }
        }
    }
}

// Quantize int32 -> int8
static void quantize(int32_t *in, int8_t *out, int n, int scale_shift) {
    for (int i = 0; i < n; i++) {
        int32_t q = in[i] >> scale_shift;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

// Run inference on a single image
int lenet5_inference(const int8_t *input) {
    // Conv1: 1x28x28 -> 6x28x28 (with padding=2)
    npu_conv2d((int8_t*)input, (int8_t*)weight_conv1_weight, conv1_out,
               1, INPUT_C, INPUT_H, INPUT_W,
               CONV1_OUT_C, CONV1_K, CONV1_K, 2, 1, NPU_ACT_RELU);
    
    // Add bias (conv1.bias)
    for (int c = 0; c < CONV1_OUT_C; c++) {
        int32_t bias = (int32_t)weight_conv1_bias[c] << 7;  // Scale bias
        for (int i = 0; i < CONV1_OUT_H * CONV1_OUT_W; i++) {
            conv1_out[c * CONV1_OUT_H * CONV1_OUT_W + i] += bias;
        }
    }
    
    // Pool1: 6x28x28 -> 6x14x14
    maxpool2x2(conv1_out, pool1_out, CONV1_OUT_C, CONV1_OUT_H, CONV1_OUT_W, 8);
    
    // Conv2: 6x14x14 -> 16x10x10
    npu_conv2d(pool1_out, (int8_t*)weight_conv2_weight, conv2_out,
               1, CONV1_OUT_C, POOL1_H, POOL1_W,
               CONV2_OUT_C, CONV2_K, CONV2_K, 0, 1, NPU_ACT_RELU);
    
    // Add bias (conv2.bias)
    for (int c = 0; c < CONV2_OUT_C; c++) {
        int32_t bias = (int32_t)weight_conv2_bias[c] << 7;
        for (int i = 0; i < CONV2_OUT_H * CONV2_OUT_W; i++) {
            conv2_out[c * CONV2_OUT_H * CONV2_OUT_W + i] += bias;
        }
    }
    
    // Pool2: 16x10x10 -> 16x5x5
    maxpool2x2(conv2_out, pool2_out, CONV2_OUT_C, CONV2_OUT_H, CONV2_OUT_W, 8);
    
    // Flatten: copy to flatten_buf
    memcpy(flatten_buf, pool2_out, FC1_IN);
    
    // FC1: 400 -> 120
    npu_matmul(flatten_buf, (int8_t*)weight_fc1_weight, fc1_out, 1, FC1_OUT, FC1_IN);
    
    // Add bias and ReLU
    for (int i = 0; i < FC1_OUT; i++) {
        fc1_out[i] += (int32_t)weight_fc1_bias[i] << 7;
        if (fc1_out[i] < 0) fc1_out[i] = 0;
    }
    quantize(fc1_out, fc1_out_q, FC1_OUT, 8);
    
    // FC2: 120 -> 84
    npu_matmul(fc1_out_q, (int8_t*)weight_fc2_weight, fc2_out, 1, FC2_OUT, FC1_OUT);
    for (int i = 0; i < FC2_OUT; i++) {
        fc2_out[i] += (int32_t)weight_fc2_bias[i] << 7;
        if (fc2_out[i] < 0) fc2_out[i] = 0;
    }
    quantize(fc2_out, fc2_out_q, FC2_OUT, 8);
    
    // FC3: 84 -> 10
    npu_matmul(fc2_out_q, (int8_t*)weight_fc3_weight, fc3_out, 1, FC3_OUT, FC2_OUT);
    for (int i = 0; i < FC3_OUT; i++) {
        fc3_out[i] += (int32_t)weight_fc3_bias[i] << 7;
    }
    
    // Find argmax
    int max_idx = 0;
    int32_t max_val = fc3_out[0];
    for (int i = 1; i < FC3_OUT; i++) {
        if (fc3_out[i] > max_val) {
            max_val = fc3_out[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

int main() {
    ioe_init();
    npu_reset();
    
    printf("=== LeNet-5 ONNX Inference Test ===\\n\\n");
    
    int correct = 0;
    int total = NUM_CLASSES;
    
    for (int digit = 0; digit < NUM_CLASSES; digit++) {
        const int8_t *img = test_images[digit];
        int predicted = lenet5_inference(img);
        int expected = expected_labels[digit];
        
        printf("Digit %d: predicted=%d, expected=%d -> %s\\n",
               digit, predicted, expected,
               (predicted == expected) ? "PASS" : "FAIL");
        
        if (predicted == expected) correct++;
    }
    
    printf("\\n=== Results ===\\n");
    printf("Accuracy: %d/%d = %d%%\\n", correct, total, correct * 100 / total);
    
    printf("\\n=== NPU Performance ===\\n");
    printf("Cycles:      %u\\n", npu_get_cycles());
    printf("Memory:      %u bytes\\n", npu_get_mem_bytes());
    printf("GEMM ops:    %u\\n", npu_get_gemm_count());
    printf("Activations: %u\\n", npu_get_act_count());
    printf("DMA:         %u\\n", npu_get_dma_count());
    
    if (correct >= 7) {  // At least 70% accuracy
        printf("\\n=== TEST PASS ===\\n");
        return 0;
    } else {
        printf("\\n=== TEST FAIL ===\\n");
        return 1;
    }
}
'''
    
    code_path = os.path.join(TARGET_DIR, 'lenet5_inference.c')
    with open(code_path, 'w') as f:
        f.write(code)
    print(f"Saved inference code to {code_path}")


if __name__ == "__main__":
    main()
