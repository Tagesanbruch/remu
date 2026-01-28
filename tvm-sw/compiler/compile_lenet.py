#!/usr/bin/env python3
"""
LeNet ONNX Model Analyzer and Compiler for REMU NPU

1. Analyze model structure
2. Extract and quantize weights  
3. Generate C code for inference
4. Generate test digit images
"""

import os
import sys
import json
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Installing onnx...")
    os.system("uv add onnx")
    import onnx
    from onnx import numpy_helper

try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    os.system("uv add onnxruntime")
    import onnxruntime as ort


def analyze_model(model_path):
    """Analyze ONNX model and print structure."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print('='*60)
    
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"\nModel: {model.graph.name}")
    print(f"Opset: {[op.version for op in model.opset_import]}")
    
    # Inputs
    print("\n--- Inputs ---")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param 
                 for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  {inp.name}: shape={shape}, dtype={dtype}")
    
    # Outputs
    print("\n--- Outputs ---")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param 
                 for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: shape={shape}")
    
    # Layers
    print("\n--- Layers ---")
    for i, node in enumerate(graph.node):
        attrs = {a.name: a for a in node.attribute}
        info = ""
        
        if node.op_type == "Conv":
            ks = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else "?"
            st = list(attrs["strides"].ints) if "strides" in attrs else [1,1]
            info = f"kernel={ks}, stride={st}"
        elif node.op_type == "Gemm":
            info = f"inputs={node.input}"
        elif node.op_type == "MaxPool":
            ks = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else "?"
            info = f"kernel={ks}"
        
        print(f"  [{i:2d}] {node.op_type:15s} {info}")
    
    # Weights
    print("\n--- Weights ---")
    total_params = 0
    weights_info = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        weights_info[init.name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size)
        }
        total_params += arr.size
        print(f"  {init.name}: shape={arr.shape}, dtype={arr.dtype}")
    
    print(f"\nTotal parameters: {total_params:,}")
    
    return model, weights_info


def run_reference_inference(model_path, input_data):
    """Run inference using ONNX Runtime for reference."""
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    output = sess.run(None, {input_name: input_data})[0]
    return output


def create_digit_images():
    """Create simple 28x28 digit images (0-9) for testing."""
    digits = {}
    
    # Simple hand-crafted digit patterns (28x28)
    # Each digit is represented as a simplified pattern
    
    patterns = {
        0: """
           ......######......
           ....##......##....
           ...##........##...
           ...#..........#...
           ...#..........#...
           ...#..........#...
           ...#..........#...
           ...##........##...
           ....##......##....
           ......######......
        """,
        1: """
           .......##.........
           ......###.........
           .....#.##.........
           .......##.........
           .......##.........
           .......##.........
           .......##.........
           .......##.........
           .......##.........
           .....######.......
        """,
        2: """
           .....######.......
           ...##......##.....
           ...........##.....
           ..........##......
           .........##.......
           ........##........
           .......##.........
           ......##..........
           .....##...........
           ....##########....
        """,
        3: """
           .....######.......
           ...##......##.....
           ...........##.....
           ...........##.....
           .......####.......
           ...........##.....
           ...........##.....
           ...##......##.....
           .....######.......
           ..................
        """,
        7: """
           ....##########....
           ............##....
           ...........##.....
           ..........##......
           .........##.......
           ........##........
           .......##.........
           ......##..........
           .....##...........
           ....##............
        """,
    }
    
    for digit, pattern in patterns.items():
        # Parse pattern to 28x28 image
        lines = [l.strip() for l in pattern.strip().split('\n') if l.strip()]
        img = np.zeros((28, 28), dtype=np.float32)
        
        # Center the pattern
        start_y = (28 - len(lines)) // 2
        for y, line in enumerate(lines):
            start_x = (28 - len(line)) // 2
            for x, ch in enumerate(line):
                if ch == '#':
                    img[start_y + y, start_x + x] = 1.0
        
        # Add some blur/smoothing effect
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.5)
        
        digits[digit] = img
    
    return digits


def create_simple_digit_images():
    """Create simple digit images without scipy dependency."""
    digits = {}
    
    # Digit 0
    img0 = np.zeros((28, 28), dtype=np.float32)
    for i in range(8, 20):
        img0[6, i] = img0[21, i] = 1.0
        img0[i-2, 6] = img0[i-2, 21] = 1.0
    digits[0] = img0
    
    # Digit 1
    img1 = np.zeros((28, 28), dtype=np.float32)
    for i in range(6, 22):
        img1[i, 14] = 1.0
    img1[21, 10:19] = 1.0
    img1[7, 12:15] = 1.0
    digits[1] = img1
    
    # Digit 2
    img2 = np.zeros((28, 28), dtype=np.float32)
    img2[6, 8:20] = 1.0
    for i in range(6, 12):
        img2[i, 20] = 1.0
    img2[12, 8:21] = 1.0
    for i in range(12, 22):
        img2[i, 8] = 1.0
    img2[21, 8:21] = 1.0
    digits[2] = img2
    
    # Digit 7
    img7 = np.zeros((28, 28), dtype=np.float32)
    img7[6, 6:22] = 1.0
    for i in range(6, 22):
        x = 21 - (i - 6) // 2
        img7[i, x] = 1.0
        img7[i, x-1] = 0.5
    digits[7] = img7
    
    return digits


def quantize_weight(weight, bits=8):
    """Quantize float32 weight to int8."""
    abs_max = max(abs(weight.min()), abs(weight.max()))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    quantized = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
    return quantized, float(scale)


def generate_c_header(model_path, output_dir):
    """Generate C header files with weights and test data."""
    
    model = onnx.load(model_path)
    graph = model.graph
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weights
    weights = {}
    scales = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            q, s = quantize_weight(arr)
            weights[init.name] = q
            scales[init.name] = s
        else:
            weights[init.name] = arr
            scales[init.name] = 1.0
    
    # Generate weights header
    weights_h = [
        "// Auto-generated LeNet weights",
        "// Quantized to int8",
        "#ifndef __LENET_WEIGHTS_H__",
        "#define __LENET_WEIGHTS_H__",
        "",
        "#include <stdint.h>",
        "",
    ]
    
    # Weight arrays
    for name, weight in weights.items():
        safe_name = name.replace(".", "_").replace("/", "_")
        flat = weight.flatten()
        
        weights_h.append(f"// {name}: shape={list(weight.shape)}, scale={scales[name]:.6f}")
        weights_h.append(f"static const int8_t {safe_name}[{len(flat)}] = {{")
        
        # Format in rows of 16
        for i in range(0, len(flat), 16):
            row = flat[i:i+16]
            row_str = ", ".join(f"{int(v):4d}" for v in row)
            weights_h.append(f"    {row_str},")
        
        weights_h.append("};")
        weights_h.append(f"#define {safe_name.upper()}_SCALE {scales[name]:.6f}f")
        weights_h.append("")
    
    weights_h.append("#endif // __LENET_WEIGHTS_H__")
    
    weights_path = os.path.join(output_dir, "lenet_weights.h")
    with open(weights_path, 'w') as f:
        f.write('\n'.join(weights_h))
    print(f"Generated: {weights_path}")
    
    # Generate test images header
    digits = create_simple_digit_images()
    
    test_h = [
        "// Auto-generated test digit images",
        "// 28x28 grayscale, quantized to int8",
        "#ifndef __LENET_TEST_H__",
        "#define __LENET_TEST_H__",
        "",
        "#include <stdint.h>",
        "",
        "#define TEST_IMAGE_SIZE (28 * 28)",
        "",
    ]
    
    # Get reference predictions
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    for digit, img in digits.items():
        # Normalize and quantize
        img_q = np.clip(img * 127, -128, 127).astype(np.int8)
        flat = img_q.flatten()
        
        # Get reference prediction
        input_data = img.reshape(1, 1, 28, 28).astype(np.float32)
        output = sess.run(None, {input_name: input_data})[0]
        pred = int(np.argmax(output))
        confidence = float(np.max(output))
        
        test_h.append(f"// Digit {digit}, predicted={pred}, conf={confidence:.4f}")
        test_h.append(f"static const int8_t test_digit_{digit}[{len(flat)}] = {{")
        
        for i in range(0, len(flat), 16):
            row = flat[i:i+16]
            row_str = ", ".join(f"{int(v):4d}" for v in row)
            test_h.append(f"    {row_str},")
        
        test_h.append("};")
        test_h.append(f"#define TEST_DIGIT_{digit}_LABEL {digit}")
        test_h.append("")
    
    # Test array
    test_digits = list(digits.keys())
    test_h.append(f"#define NUM_TEST_DIGITS {len(test_digits)}")
    test_h.append("static const int8_t* test_digits[] = {")
    for d in test_digits:
        test_h.append(f"    test_digit_{d},")
    test_h.append("};")
    test_h.append(f"static const int test_labels[] = {{ {', '.join(str(d) for d in test_digits)} }};")
    test_h.append("")
    test_h.append("#endif // __LENET_TEST_H__")
    
    test_path = os.path.join(output_dir, "lenet_test.h")
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_h))
    print(f"Generated: {test_path}")
    
    return weights, scales, digits


def generate_inference_code(model_path, output_dir):
    """Generate C inference code."""
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # Analyze layer structure
    layers = []
    for node in graph.node:
        attrs = {a.name: a for a in node.attribute}
        layer = {"op": node.op_type, "name": node.name, "inputs": list(node.input), "outputs": list(node.output)}
        
        if node.op_type == "Conv":
            layer["kernel"] = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [3,3]
            layer["stride"] = list(attrs["strides"].ints) if "strides" in attrs else [1,1]
            layer["pads"] = list(attrs["pads"].ints) if "pads" in attrs else [0,0,0,0]
        elif node.op_type == "MaxPool":
            layer["kernel"] = list(attrs["kernel_shape"].ints)
            layer["stride"] = list(attrs["strides"].ints) if "strides" in attrs else layer["kernel"]
        elif node.op_type == "Gemm":
            layer["transB"] = attrs["transB"].i if "transB" in attrs else 0
        
        layers.append(layer)
    
    # Get weight shapes
    weight_shapes = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        weight_shapes[init.name] = list(arr.shape)
    
    # Generate inference code
    code = [
        "/**",
        " * Auto-generated LeNet inference code for REMU NPU",
        " */",
        "",
        "#include <am.h>",
        "#include <klib.h>",
        '#include "npu.h"',
        '#include "lenet_weights.h"',
        '#include "lenet_test.h"',
        "",
        "// Intermediate buffers",
        "static int8_t input_buf[1 * 28 * 28];",
        "static int32_t conv1_out[6 * 24 * 24];",
        "static int8_t pool1_out[6 * 12 * 12];",
        "static int32_t conv2_out[16 * 8 * 8];",
        "static int8_t pool2_out[16 * 4 * 4];",
        "static int32_t fc1_out[120];",
        "static int8_t fc1_out_q[120];",
        "static int32_t fc2_out[84];",
        "static int8_t fc2_out_q[84];",
        "static int32_t fc3_out[10];",
        "",
        "// ReLU activation",
        "static void relu_i32(int32_t *data, int n) {",
        "    for (int i = 0; i < n; i++) {",
        "        if (data[i] < 0) data[i] = 0;",
        "    }",
        "}",
        "",
        "// Max pooling 2x2 with quantization",
        "static void maxpool2x2_q(int32_t *in, int8_t *out, int c, int h, int w, int shift) {",
        "    int oh = h / 2, ow = w / 2;",
        "    for (int ch = 0; ch < c; ch++) {",
        "        for (int y = 0; y < oh; y++) {",
        "            for (int x = 0; x < ow; x++) {",
        "                int32_t v0 = in[ch*h*w + (y*2)*w + (x*2)];",
        "                int32_t v1 = in[ch*h*w + (y*2)*w + (x*2+1)];",
        "                int32_t v2 = in[ch*h*w + (y*2+1)*w + (x*2)];",
        "                int32_t v3 = in[ch*h*w + (y*2+1)*w + (x*2+1)];",
        "                int32_t mx = v0;",
        "                if (v1 > mx) mx = v1;",
        "                if (v2 > mx) mx = v2;",
        "                if (v3 > mx) mx = v3;",
        "                int32_t q = mx >> shift;",
        "                if (q > 127) q = 127;",
        "                if (q < -128) q = -128;",
        "                out[ch*oh*ow + y*ow + x] = (int8_t)q;",
        "            }",
        "        }",
        "    }",
        "}",
        "",
        "// Quantize i32 to i8",
        "static void quantize(int32_t *in, int8_t *out, int n, int shift) {",
        "    for (int i = 0; i < n; i++) {",
        "        int32_t q = in[i] >> shift;",
        "        if (q > 127) q = 127;",
        "        if (q < -128) q = -128;",
        "        out[i] = (int8_t)q;",
        "    }",
        "}",
        "",
        "// LeNet inference",
        "int lenet_inference(const int8_t *input) {",
        "    npu_reset();",
        "",
        "    // Copy input",
        "    memcpy(input_buf, input, 28 * 28);",
        "",
    ]
    
    # Generate layer calls based on model structure
    # Conv1: 1x28x28 -> 6x24x24
    code.extend([
        "    // Conv1: 1x28x28 -> 6x24x24, kernel=5x5",
        "    npu_conv2d(input_buf, (int8_t*)conv1_weight, conv1_out,",
        "               1, 1, 28, 28, 6, 5, 5, 0, 1, NPU_ACT_RELU);",
        "",
        "    // Pool1: 6x24x24 -> 6x12x12",
        "    maxpool2x2_q(conv1_out, pool1_out, 6, 24, 24, 8);",
        "",
    ])
    
    # Conv2: 6x12x12 -> 16x8x8
    code.extend([
        "    // Conv2: 6x12x12 -> 16x8x8, kernel=5x5",
        "    npu_conv2d(pool1_out, (int8_t*)conv2_weight, conv2_out,",
        "               1, 6, 12, 12, 16, 5, 5, 0, 1, NPU_ACT_RELU);",
        "",
        "    // Pool2: 16x8x8 -> 16x4x4",
        "    maxpool2x2_q(conv2_out, pool2_out, 16, 8, 8, 8);",
        "",
    ])
    
    # FC layers
    code.extend([
        "    // FC1: 256 -> 120",
        "    npu_matmul(pool2_out, (int8_t*)fc1_weight, fc1_out, 1, 120, 256);",
        "    relu_i32(fc1_out, 120);",
        "    quantize(fc1_out, fc1_out_q, 120, 8);",
        "",
        "    // FC2: 120 -> 84",
        "    npu_matmul(fc1_out_q, (int8_t*)fc2_weight, fc2_out, 1, 84, 120);",
        "    relu_i32(fc2_out, 84);",
        "    quantize(fc2_out, fc2_out_q, 84, 8);",
        "",
        "    // FC3: 84 -> 10",
        "    npu_matmul(fc2_out_q, (int8_t*)fc3_weight, fc3_out, 1, 10, 84);",
        "",
        "    // Find argmax",
        "    int max_idx = 0;",
        "    int32_t max_val = fc3_out[0];",
        "    for (int i = 1; i < 10; i++) {",
        "        if (fc3_out[i] > max_val) {",
        "            max_val = fc3_out[i];",
        "            max_idx = i;",
        "        }",
        "    }",
        "",
        "    return max_idx;",
        "}",
        "",
        "void print_npu_stats(void) {",
        '    printf("NPU Stats:\\n");',
        '    printf("  Cycles:      %u\\n", npu_get_cycles());',
        '    printf("  Mem Traffic: %u bytes\\n", npu_get_mem_bytes());',
        '    printf("  GEMM Ops:    %u\\n", npu_get_gemm_count());',
        '    printf("  Activations: %u\\n", npu_get_act_count());',
        '    printf("  DMA Xfers:   %u\\n", npu_get_dma_count());',
        "}",
        "",
        "int main() {",
        "    ioe_init();",
        "",
        '    printf("=== LeNet ONNX Inference Test ===\\n\\n");',
        "",
        "    int correct = 0;",
        "    for (int i = 0; i < NUM_TEST_DIGITS; i++) {",
        "        int pred = lenet_inference(test_digits[i]);",
        "        int label = test_labels[i];",
        '        printf("Test %d: label=%d, pred=%d %s\\n", ',
        '               i, label, pred, (pred == label) ? "[OK]" : "[FAIL]");',
        "        if (pred == label) correct++;",
        "    }",
        "",
        '    printf("\\nAccuracy: %d/%d\\n\\n", correct, NUM_TEST_DIGITS);',
        "",
        "    print_npu_stats();",
        "",
        "    if (correct == NUM_TEST_DIGITS) {",
        '        printf("\\n=== ALL TESTS PASSED ===\\n");',
        "        return 0;",
        "    } else {",
        '        printf("\\n=== SOME TESTS FAILED ===\\n");',
        "        return 1;",
        "    }",
        "}",
    ])
    
    code_path = os.path.join(output_dir, "lenet_inference.c")
    with open(code_path, 'w') as f:
        f.write('\n'.join(code))
    print(f"Generated: {code_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
    model_path = os.path.join(repo_root, "tvm-sw/onnx/lenet.onnx")
    output_dir = os.path.join(repo_root, "am-kernels/tests/npu-tests/tests/lenet_onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Analyze model
    model, weights_info = analyze_model(model_path)
    
    # Generate files
    print(f"\n{'='*60}")
    print("Generating C files...")
    print('='*60)
    
    generate_c_header(model_path, output_dir)
    generate_inference_code(model_path, output_dir)
    
    print(f"\n{'='*60}")
    print("Done! Generated files in:", output_dir)
    print('='*60)


if __name__ == "__main__":
    main()
