#!/usr/bin/env python3
"""
Analyze ONNX model operations to determine NPU hardware requirements.
"""

import sys
import onnx

def analyze_model(model_path):
    model = onnx.load(model_path)
    
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    
    print(f"\n=== {model_path} ===")
    print(f"Total nodes: {len(model.graph.node)}")
    print("\nOperations:")
    for op, cnt in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {op}: {cnt}")
    
    return ops

if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["onnx/mobilenetv2-7.onnx"]
    
    all_ops = set()
    for m in models:
        try:
            ops = analyze_model(m)
            all_ops.update(ops.keys())
        except Exception as e:
            print(f"Error loading {m}: {e}")
    
    print("\n=== All unique operations ===")
    for op in sorted(all_ops):
        print(f"  {op}")
