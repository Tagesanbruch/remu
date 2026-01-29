import os
import numpy as np

def generate_test_data(onnx_model, test_dir, input_shape, input_scale):
    """Generate test input/output data for verification."""
    import onnxruntime
    
    # Create deterministic random input
    np.random.seed(42)
    input_data = (np.random.randn(*input_shape).astype(np.float32) * 50)
    
    # Quantize input to INT8
    abs_max = max(abs(input_data.min()), abs(input_data.max()))
    qmax = 127
    input_scale = abs_max / qmax
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


def generate_test_program(test_c, model_name, input_shape):
    """Generate test program."""
    with open(test_c, 'w') as f:
        f.write(f"""/**
 * {model_name.upper()} Inference Test for REMU
 */

#include <am.h>
#include <klib.h>

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
# Build: make ARCH=riscv32-remu run

NAME = {model_name}
SRCS = {model_name}_inference.c test_{model_name}.c
LIBS = klib

include $(REMU_AM_HOME)/Makefile
""")
    print(f"  Generated Makefile: {makefile_path}")
