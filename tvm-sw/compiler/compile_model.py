#!/usr/bin/env python3
"""
TVM Model Compiler for REMU NPU

Compiles ONNX models to C code that runs on the REMU platform with NPU acceleration.

Usage:
    uv run python compile_model.py --model ../onnx/lenet.onnx --output build/
    uv run python compile_model.py --model ../onnx/mobilenetv2-7.onnx --analyze-only
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package not found. Run: uv add onnx")
    sys.exit(1)

# NPU hardware constraints
NPU_SRAM_SIZE = 16 * 1024  # 16KB per SRAM
NPU_TILE_MAX = 64


def analyze_onnx_model(model_path: str) -> Dict[str, Any]:
    """Analyze ONNX model to extract layer information."""
    model = onnx.load(model_path)
    graph = model.graph
    
    # Get initializers (weights)
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    
    layers = []
    total_params = 0
    
    for node in graph.node:
        layer_info = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {},
        }
        
        # Extract attributes
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                layer_info["attributes"][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                layer_info["attributes"][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOATS:
                layer_info["attributes"][attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.FLOAT:
                layer_info["attributes"][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                layer_info["attributes"][attr.name] = attr.s.decode('utf-8')
        
        # Get weight info for relevant ops
        if node.op_type == "Conv" and len(node.input) > 1:
            weight_name = node.input[1]
            if weight_name in initializers:
                weight = initializers[weight_name]
                layer_info["weight_shape"] = list(weight.shape)
                layer_info["params"] = int(weight.size)
                total_params += weight.size
                if len(weight.shape) == 4:
                    layer_info["oc"], layer_info["ic"] = int(weight.shape[0]), int(weight.shape[1])
                    layer_info["kh"], layer_info["kw"] = int(weight.shape[2]), int(weight.shape[3])
        
        elif node.op_type in ("Gemm", "MatMul") and len(node.input) > 1:
            weight_name = node.input[1]
            if weight_name in initializers:
                weight = initializers[weight_name]
                layer_info["weight_shape"] = list(weight.shape)
                layer_info["params"] = int(weight.size)
                total_params += weight.size
        
        layers.append(layer_info)
    
    # Count ops by type
    op_counts = {}
    for layer in layers:
        op = layer["op_type"]
        op_counts[op] = op_counts.get(op, 0) + 1
    
    return {
        "model_path": model_path,
        "layers": layers,
        "op_counts": op_counts,
        "total_params": int(total_params),
        "total_layers": len(layers),
    }


def quantize_weights(weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Quantize float32 weights to int8 (symmetric)."""
    abs_max = max(abs(weights.min()), abs(weights.max()))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    return quantized, float(scale), 0


def generate_weight_file(model_path: str, output_dir: str) -> Dict[str, Any]:
    """Extract, quantize, and save weights."""
    model = onnx.load(model_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    
    weight_info = {}
    all_weights = []
    offset = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, weight in initializers.items():
        if weight.dtype in (np.float32, np.float64):
            quantized, scale, zero = quantize_weights(weight)
        else:
            quantized = weight.astype(np.int8)
            scale, zero = 1.0, 0
        
        weight_info[name] = {
            "offset": offset,
            "size": int(quantized.size),
            "shape": [int(d) for d in quantized.shape],
            "scale": scale,
            "zero_point": zero,
        }
        
        all_weights.append(quantized.flatten())
        offset += quantized.size
    
    # Concatenate and save
    all_weights = np.concatenate(all_weights)
    
    weights_path = os.path.join(output_dir, "model_weights.bin")
    all_weights.tofile(weights_path)
    
    meta_path = os.path.join(output_dir, "weight_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(weight_info, f, indent=2)
    
    print(f"Written {len(all_weights):,} bytes of weights to {weights_path}")
    return weight_info


def safe_name(s: str) -> str:
    """Convert string to valid C identifier."""
    return s.replace(".", "_").replace("/", "_").replace(":", "_").replace("-", "_")


def generate_inference_code(model_path: str, weight_info: Dict, output_dir: str):
    """Generate C code for inference using NPU."""
    model = onnx.load(model_path)
    graph = model.graph
    model_name = Path(model_path).stem
    
    lines = [
        "/**",
        f" * Auto-generated inference code for {model_name}",
        " * Compiled by TVM-REMU NPU Codegen",
        " */",
        "",
        "#include <am.h>",
        "#include <klib.h>",
        '#include "npu.h"',
        "",
        "// Flash base for model weights",
        "#define FLASH_BASE 0x30000000",
        "",
        "// Buffer pool",
        "#define BUF_SIZE (64 * 1024)",
        "static int8_t g_input[BUF_SIZE];",
        "static int32_t g_output[BUF_SIZE / sizeof(int32_t)];",
        "static int32_t g_temp[BUF_SIZE / sizeof(int32_t)];",
        "",
    ]
    
    # Weight offset defines
    lines.append("// Weight offsets in Flash")
    for name, info in weight_info.items():
        sn = safe_name(name).upper()
        lines.append(f"#define W_{sn}_OFF {info['offset']}")
        lines.append(f"#define W_{sn}_SZ {info['size']}")
    lines.append("")
    
    # Generate layer stubs
    lines.append("// Layer implementations")
    conv_idx, gemm_idx, relu_idx = 0, 0, 0
    
    for node in graph.node:
        if node.op_type == "Conv" and len(node.input) > 1:
            weight_name = node.input[1]
            if weight_name in weight_info:
                info = weight_info[weight_name]
                shape = info["shape"]
                if len(shape) == 4:
                    oc, ic, kh, kw = shape
                else:
                    continue
                
                sn = safe_name(weight_name).upper()
                attrs = {a.name: a for a in node.attribute}
                strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
                pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
                
                lines.extend([
                    "",
                    f"// Conv{conv_idx}: {node.name}",
                    f"// Weight: [{oc}, {ic}, {kh}, {kw}], stride={strides[0]}, pad={pads[0]}",
                    f"static void conv_{conv_idx}(int8_t *in, int32_t *out, int h, int w) {{",
                    f"    int8_t *weight = (int8_t*)(FLASH_BASE + W_{sn}_OFF);",
                    f"    npu_conv2d(in, weight, out, 1, {ic}, h, w, {oc}, {kh}, {kw}, {pads[0]}, {strides[0]}, NPU_ACT_NONE);",
                    "}",
                ])
                conv_idx += 1
        
        elif node.op_type in ("Gemm", "MatMul") and len(node.input) > 1:
            weight_name = node.input[1]
            if weight_name in weight_info:
                info = weight_info[weight_name]
                shape = info["shape"]
                if len(shape) >= 2:
                    n, k = shape[0], shape[1]
                else:
                    n, k = shape[0], 1
                
                sn = safe_name(weight_name).upper()
                lines.extend([
                    "",
                    f"// Gemm{gemm_idx}: {node.name}",
                    f"// Weight: {shape}",
                    f"static void gemm_{gemm_idx}(int8_t *in, int32_t *out) {{",
                    f"    int8_t *weight = (int8_t*)(FLASH_BASE + W_{sn}_OFF);",
                    f"    npu_matmul(in, weight, out, 1, {n}, {k});",
                    "}",
                ])
                gemm_idx += 1
        
        elif node.op_type == "Relu":
            lines.extend([
                "",
                f"// ReLU{relu_idx}: {node.name}",
                f"static void relu_{relu_idx}(int32_t *data, int n) {{",
                f"    for (int i = 0; i < n; i++) if (data[i] < 0) data[i] = 0;",
                "}",
            ])
            relu_idx += 1
    
    # Main inference function
    lines.extend([
        "",
        "// Main inference",
        "void run_inference(int8_t *input, int size) {",
        "    npu_reset();",
        "    printf(\"Starting inference for " + model_name + "...\\n\");",
        "",
        "    // TODO: Call layers in topological order",
        "    // (Manual scheduling or TVM graph executor)",
        "",
        "    printf(\"=== NPU Performance Report ===\\n\");",
        "    printf(\"NPU Cycles:     %u\\n\", npu_get_cycles());",
        "    printf(\"Memory Traffic: %u bytes\\n\", npu_get_mem_bytes());",
        "    printf(\"GEMM Ops:       %u\\n\", npu_get_gemm_count());",
        "    printf(\"Activation Ops: %u\\n\", npu_get_act_count());",
        "    printf(\"DMA Transfers:  %u\\n\", npu_get_dma_count());",
        "}",
        "",
        "int main() {",
        "    ioe_init();",
        "    memset(g_input, 0, sizeof(g_input));",
        "    run_inference(g_input, sizeof(g_input));",
        "    return 0;",
        "}",
    ])
    
    output_path = os.path.join(output_dir, f"{model_name}_inference.c")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated: {output_path}")
    print(f"  Conv layers: {conv_idx}")
    print(f"  Gemm layers: {gemm_idx}")
    print(f"  ReLU layers: {relu_idx}")


def main():
    parser = argparse.ArgumentParser(description="TVM to REMU NPU Compiler")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="./build", help="Output directory")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze model")
    args = parser.parse_args()
    
    print(f"Analyzing: {args.model}")
    analysis = analyze_onnx_model(args.model)
    
    print(f"\n=== Model Summary ===")
    print(f"Layers: {analysis['total_layers']}")
    print(f"Parameters: {analysis['total_params']:,}")
    print(f"Ops by type:")
    for op, cnt in sorted(analysis['op_counts'].items(), key=lambda x: -x[1]):
        print(f"  {op}: {cnt}")
    
    if args.analyze_only:
        # Save analysis
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to {args.output}/analysis.json")
        return
    
    print(f"\n=== Extracting Weights ===")
    weight_info = generate_weight_file(args.model, args.output)
    
    print(f"\n=== Generating Code ===")
    generate_inference_code(args.model, weight_info, args.output)
    
    print(f"\n=== Done ===")
    print(f"Files in {args.output}/")


if __name__ == "__main__":
    main()
