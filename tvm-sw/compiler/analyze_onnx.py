#!/usr/bin/env python3
"""Analyze ONNX model structure"""
import sys
import onnx

def analyze_model(path):
    model = onnx.load(path)
    print(f"Model: {path}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    
    print("\n=== Inputs ===")
    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    print("\n=== Outputs ===")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    print("\n=== Layers (first 10) ===")
    for i, node in enumerate(model.graph.node[:10]):
        attrs = {a.name: a for a in node.attribute}
        info = ""
        if node.op_type == "Conv":
            if "kernel_shape" in attrs:
                info = f"kernel={list(attrs['kernel_shape'].ints)}"
        print(f"  [{i}] {node.op_type}: {info}")
    
    print(f"\nTotal nodes: {len(model.graph.node)}")
    print(f"Total initializers: {len(model.graph.initializer)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "onnx/lenet.onnx"
    analyze_model(path)
