#!/usr/bin/env python3
"""
Generate test data for REMU verification.

This script:
1. Creates a test input (random or from image)
2. Runs TVM inference to get reference output
3. Saves quantized input and reference output as binary files

Usage:
    python generate_remu_test.py --model mobilenetv2-7.onnx --output_dir ./test_data/

Output files:
    - test_input.bin: INT8 quantized input [N,C,H,W]
    - test_output.bin: INT32 reference output [N, num_classes]
    - test_meta.txt: Metadata (shapes, scales, top-k predictions)
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Patch ml_dtypes
def _patch_ml_dtypes():
    try:
        import ml_dtypes
        if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
            ml_dtypes.float8_e4m3fnuz = ml_dtypes.float8_e4m3fn
        if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
            ml_dtypes.float8_e5m2fnuz = ml_dtypes.float8_e5m2
        if not hasattr(ml_dtypes, 'int4'):
            ml_dtypes.int4 = np.int8
        if not hasattr(ml_dtypes, 'uint4'):
            ml_dtypes.uint4 = np.uint8
        if not hasattr(ml_dtypes, 'float4_e2m1fn'):
            ml_dtypes.float4_e2m1fn = ml_dtypes.bfloat16
    except:
        pass
_patch_ml_dtypes()

import onnx
import onnxruntime
import tvm
from tvm import relay


def quantize_symmetric(data, bits=8):
    """Symmetric INT8 quantization."""
    abs_max = max(abs(data.min()), abs(data.max()))
    if abs_max < 1e-10:
        return np.zeros_like(data, dtype=np.int8), 1.0
    qmax = (1 << (bits - 1)) - 1  # 127
    scale = abs_max / qmax
    quantized = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
    return quantized, scale


def create_test_input(image_path=None, shape=(1, 3, 224, 224), seed=42):
    """Create test input data."""
    if image_path and os.path.exists(image_path):
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((shape[3], shape[2]), Image.BILINEAR)
        img_data = np.array(img, dtype=np.float32)
        
        # ImageNet normalization
        mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img_data = img_data - mean
        
        # NHWC -> NCHW
        img_data = np.expand_dims(img_data, axis=0)
        img_data = img_data.transpose(0, 3, 1, 2)
        return img_data
    else:
        # Create deterministic random input
        np.random.seed(seed)
        # Create realistic image-like data
        data = np.random.randn(*shape).astype(np.float32) * 50
        return data


def run_tvm_inference(model_path, input_data):
    """Run TVM inference and return output."""
    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: input_data.shape}
    
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        target = "llvm"
        lib = relay.build(mod, target=target, params=params)
    
    from tvm.contrib import graph_executor
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, input_data)
    module.run()
    
    output = module.get_output(0).numpy()
    return output


def main():
    parser = argparse.ArgumentParser(description='Generate REMU test data')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--image', help='Path to test image (optional)')
    parser.add_argument('--output_dir', default='./test_data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("REMU Test Data Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    
    # Create test input
    print("\n[1/4] Creating test input...")
    input_data = create_test_input(args.image, seed=args.seed)
    print(f"  Input shape: {input_data.shape}")
    print(f"  Input range: [{input_data.min():.2f}, {input_data.max():.2f}]")
    
    # Quantize input to INT8
    print("\n[2/4] Quantizing input to INT8...")
    input_q, input_scale = quantize_symmetric(input_data)
    print(f"  Input scale: {input_scale:.6e}")
    print(f"  Quantized range: [{input_q.min()}, {input_q.max()}]")
    
    # Run TVM inference with float input
    print("\n[3/4] Running TVM inference...")
    output = run_tvm_inference(args.model, input_data)
    print(f"  Output shape: {output.shape}")
    
    # Get top-5 predictions
    top5_idx = np.argsort(output[0])[-5:][::-1]
    top5_scores = output[0][top5_idx]
    print(f"  Top-5 classes: {top5_idx.tolist()}")
    print(f"  Top-5 scores: {top5_scores.tolist()}")
    
    # Quantize output for comparison (NPU outputs int32)
    # Convert float scores to int32 (scale to preserve precision)
    output_max = max(abs(output.min()), abs(output.max()))
    output_scale = output_max / (2**30)  # Keep within int32 range
    output_q = (output / output_scale).astype(np.int32)
    
    # Save files
    print("\n[4/4] Saving test data...")
    
    # Save quantized input
    input_path = os.path.join(args.output_dir, 'test_input.bin')
    input_q.tofile(input_path)
    print(f"  Saved: {input_path} ({input_q.nbytes} bytes)")
    
    # Save quantized output
    output_path = os.path.join(args.output_dir, 'test_output.bin')
    output_q.tofile(output_path)
    print(f"  Saved: {output_path} ({output_q.nbytes} bytes)")
    
    # Save float output for reference
    output_float_path = os.path.join(args.output_dir, 'test_output_float.bin')
    output.astype(np.float32).tofile(output_float_path)
    print(f"  Saved: {output_float_path}")
    
    # Save metadata
    meta_path = os.path.join(args.output_dir, 'test_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"# REMU Test Metadata\n")
        f.write(f"model={args.model}\n")
        f.write(f"input_shape={input_data.shape}\n")
        f.write(f"input_scale={input_scale:.10e}\n")
        f.write(f"output_shape={output.shape}\n")
        f.write(f"output_scale={output_scale:.10e}\n")
        f.write(f"top5_classes={top5_idx.tolist()}\n")
        f.write(f"top5_scores_float={top5_scores.tolist()}\n")
        f.write(f"expected_class={top5_idx[0]}\n")
    print(f"  Saved: {meta_path}")
    
    # Generate C header for test data
    header_path = os.path.join(args.output_dir, 'test_data.h')
    with open(header_path, 'w') as f:
        f.write("/**\n")
        f.write(" * Test data header for REMU verification\n")
        f.write(" * Generated by generate_remu_test.py\n")
        f.write(" */\n\n")
        f.write("#ifndef __TEST_DATA_H__\n")
        f.write("#define __TEST_DATA_H__\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define TEST_INPUT_N {input_data.shape[0]}\n")
        f.write(f"#define TEST_INPUT_C {input_data.shape[1]}\n")
        f.write(f"#define TEST_INPUT_H {input_data.shape[2]}\n")
        f.write(f"#define TEST_INPUT_W {input_data.shape[3]}\n")
        f.write(f"#define TEST_INPUT_SIZE {input_q.size}\n")
        f.write(f"#define TEST_INPUT_SCALE {input_scale:.10e}f\n\n")
        f.write(f"#define TEST_OUTPUT_SIZE {output.shape[1]}\n")
        f.write(f"#define TEST_OUTPUT_SCALE {output_scale:.10e}f\n\n")
        f.write(f"#define EXPECTED_CLASS {top5_idx[0]}\n")
        f.write(f"#define EXPECTED_TOP5 {{{', '.join(map(str, top5_idx))}}}\n\n")
        
        # Generate test input array (split into smaller chunks for compilation)
        f.write("// Test input data as C array\n")
        f.write(f"static const int8_t test_input_data[{input_q.size}] = {{\n")
        # Split into rows of 16 elements for readability
        for i in range(0, input_q.size, 16):
            chunk = input_q.flat[i:min(i+16, input_q.size)]
            f.write("    " + ", ".join(f"{x:4d}" for x in chunk))
            if i + 16 < input_q.size:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        # Generate expected output array (reference for comparison)
        f.write("// Reference output (INT32)\n")
        f.write(f"static const int32_t test_output_ref[{output.shape[1]}] = {{\n")
        for i in range(0, output.shape[1], 8):
            chunk = output_q[0][i:min(i+8, output.shape[1])]
            f.write("    " + ", ".join(f"{x:12d}" for x in chunk))
            if i + 8 < output.shape[1]:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("#endif\n")
    print(f"  Saved: {header_path}")
    
    print("\n" + "=" * 60)
    print("Test data generation complete!")
    print("=" * 60)
    print(f"\nExpected classification: class {top5_idx[0]}")
    print(f"To verify on REMU:")
    print(f"  1. Copy test_input.bin and test_output.bin to Flash")
    print(f"  2. Run inference and compare top-k predictions")


if __name__ == "__main__":
    main()
