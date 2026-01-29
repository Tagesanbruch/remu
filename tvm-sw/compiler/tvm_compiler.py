#!/usr/bin/env python3
"""
TVM REMU NPU Compiler - CLI Wrapper
"""
import sys
from remu_tvm.core import compile_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tvm_compiler.py <model.onnx> [output_dir] [model_name]")
        print("")
        print("Examples:")
        print("  python tvm_compiler.py mobilenet.onnx ./output mobilenet")
        print("  python tvm_compiler.py resnet50.onnx ./output/resnet resnet")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "model"
    
    result = compile_model(model_path, output_dir, model_name)
    
    print("\nSummary:")
    print(f"  Parameters: {result['total_params']:,}")
    print(f"  Weights: {result['weights_size']:,} bytes")
    print(f"  Layers: {result['num_layers']}")
