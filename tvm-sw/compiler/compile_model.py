#!/usr/bin/env python3
"""
TVM Model Compiler for REMU NPU

This script compiles ONNX models to C code that can run on the REMU platform
with NPU acceleration.

Usage:
    python compile_model.py --model model.onnx --output build/
"""

import argparse
import os
import sys
import json

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    # TVM is optional for now - we'll generate manual code
    try:
        import tvm
        HAS_TVM = True
    except ImportError:
        HAS_TVM = False
        print("Warning: TVM not found, using manual code generation")
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def load_onnx_model(model_path):
    """Load and analyze ONNX model"""
    import onnx
    
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    
    # Basic model info
    print(f"  IR Version: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Opset: {[op.version for op in model.opset_import]}")
    
    # Analyze graph
    graph = model.graph
    print(f"  Nodes: {len(graph.node)}")
    print(f"  Inputs: {[i.name for i in graph.input]}")
    print(f"  Outputs: {[o.name for o in graph.output]}")
    
    # Count operation types
    op_counts = {}
    for node in graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    print("  Operations:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op}: {count}")
    
    return model

def extract_weights(model, output_dir):
    """Extract weights from ONNX model"""
    import onnx
    import numpy as np
    
    weights = {}
    total_size = 0
    
    for initializer in model.graph.initializer:
        name = initializer.name
        tensor = onnx.numpy_helper.to_array(initializer)
        weights[name] = tensor
        total_size += tensor.nbytes
        print(f"  Weight: {name}, shape={tensor.shape}, dtype={tensor.dtype}")
    
    print(f"  Total weights size: {total_size / 1024 / 1024:.2f} MB")
    
    # Save weights as binary
    weights_path = os.path.join(output_dir, "model_params.bin")
    with open(weights_path, 'wb') as f:
        # Simple format: count, then for each: name_len, name, shape_dims, shape, dtype, data
        import struct
        f.write(struct.pack('I', len(weights)))
        
        for name, tensor in weights.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            shape = tensor.shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))
            
            # Dtype as string
            dtype_str = str(tensor.dtype).encode('utf-8')
            f.write(struct.pack('I', len(dtype_str)))
            f.write(dtype_str)
            
            # Data
            f.write(tensor.tobytes())
    
    print(f"  Saved weights to: {weights_path}")
    return weights

def generate_c_code(model, weights, output_dir):
    """Generate C code for inference"""
    
    # Analyze model to find GEMM/Conv operations
    gemm_ops = []
    conv_ops = []
    
    for node in model.graph.node:
        if node.op_type in ['Gemm', 'MatMul']:
            gemm_ops.append(node)
        elif node.op_type == 'Conv':
            conv_ops.append(node)
    
    print(f"  Found {len(gemm_ops)} GEMM ops, {len(conv_ops)} Conv ops")
    
    # Generate header
    header_path = os.path.join(output_dir, "model.h")
    with open(header_path, 'w') as f:
        f.write("""#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdint.h>

// Model inference function
int model_inference(int8_t *input, int32_t *output);

// Model info
#define MODEL_INPUT_SIZE  (224 * 224 * 3)
#define MODEL_OUTPUT_SIZE 1000

// Weight loading
int model_load_weights(void *flash_base);

#endif // __MODEL_H__
""")
    print(f"  Generated: {header_path}")
    
    # Generate implementation stub
    impl_path = os.path.join(output_dir, "model.c")
    with open(impl_path, 'w') as f:
        f.write("""/**
 * Auto-generated model inference code
 * TODO: Full TVM codegen integration
 */

#include "model.h"
#include <npu.h>

// Placeholder implementation
int model_inference(int8_t *input, int32_t *output) {
    // TODO: Implement full inference pipeline
    // For now, just a simple test
    
    npu_reset();
    
    // Example: first GEMM layer
    // npu_matmul(input, weights, output, M, N, K);
    
    return 0;
}

int model_load_weights(void *flash_base) {
    // TODO: Parse weight format and load to NPU SRAM
    return 0;
}
""")
    print(f"  Generated: {impl_path}")
    
    # Generate model info JSON
    info_path = os.path.join(output_dir, "model_info.json")
    info = {
        "gemm_ops": len(gemm_ops),
        "conv_ops": len(conv_ops),
        "total_weights_mb": sum(w.nbytes for w in weights.values()) / 1024 / 1024,
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Generated: {info_path}")

def main():
    parser = argparse.ArgumentParser(description="Compile ONNX model for REMU NPU")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="build", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = load_onnx_model(args.model)
    
    # Extract weights
    weights = extract_weights(model, args.output)
    
    # Generate C code
    generate_c_code(model, weights, args.output)
    
    print("\nCompilation complete!")
    print(f"Output files in: {args.output}/")

if __name__ == "__main__":
    main()
