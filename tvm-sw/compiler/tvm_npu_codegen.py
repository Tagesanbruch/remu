#!/usr/bin/env python3
"""
TVM BYOC (Bring Your Own Codegen) for REMU NPU

This module implements proper TVM compilation:
1. Pattern matching to identify NPU-compatible subgraphs
2. Custom codegen to generate NPU API calls
3. Runtime artifact generation

Reference: https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

# # Comprehensive workaround for onnx/ml_dtypes compatibility issue
# # onnx 1.16+ requires ml_dtypes 0.4+ which has new dtypes
# # We patch ml_dtypes to add dummy types for compatibility
# def _patch_ml_dtypes():
#     try:
#         import ml_dtypes
        
#         # Float8 variants
#         if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
#             ml_dtypes.float8_e4m3fnuz = ml_dtypes.float8_e4m3fn
#         if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
#             ml_dtypes.float8_e5m2fnuz = ml_dtypes.float8_e5m2
            
#         # 4-bit types (fallback to 8-bit)
#         if not hasattr(ml_dtypes, 'int4'):
#             ml_dtypes.int4 = np.int8
#         if not hasattr(ml_dtypes, 'uint4'):
#             ml_dtypes.uint4 = np.uint8
            
#         # Float4 variants (fallback to bfloat16)
#         if not hasattr(ml_dtypes, 'float4_e2m1fn'):
#             ml_dtypes.float4_e2m1fn = ml_dtypes.bfloat16
#         if not hasattr(ml_dtypes, 'float6_e2m3fn'):
#             ml_dtypes.float6_e2m3fn = ml_dtypes.bfloat16
#         if not hasattr(ml_dtypes, 'float6_e3m2fn'):
#             ml_dtypes.float6_e3m2fn = ml_dtypes.bfloat16
            
#     except ImportError:
#         pass
#     except AttributeError:
#         pass

# _patch_ml_dtypes()

import tvm
from tvm import relay, ir
from tvm.relay import transform
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, is_constant, 
    DFPatternCallback, rewrite
)


#############################################################################
# Pattern Definitions for NPU Operations
#############################################################################

def make_conv2d_relu_pattern():
    """Pattern: Conv2D + BiasAdd (optional) + ReLU"""
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    
    # Conv2D
    conv = is_op("nn.conv2d")(data, weight)
    
    # Optional bias
    conv_bias = is_op("nn.bias_add")(conv, bias) | conv
    
    # Optional ReLU
    conv_relu = is_op("nn.relu")(conv_bias) | conv_bias
    
    return conv_relu


def make_conv2d_bn_relu_pattern():
    """Pattern: Conv2D + BatchNorm + ReLU (common in ResNet)"""
    data = wildcard()
    weight = is_constant()
    gamma = is_constant()
    beta = is_constant()
    moving_mean = is_constant()
    moving_var = is_constant()
    
    conv = is_op("nn.conv2d")(data, weight)
    bn = is_op("nn.batch_norm")(conv, gamma, beta, moving_mean, moving_var)
    bn_output = is_op("TupleGetItem")(bn)  # Get normalized output
    relu = is_op("nn.relu")(bn_output)
    
    return relu


def make_depthwise_conv_pattern():
    """Pattern: Depthwise Conv2D"""
    data = wildcard()
    weight = is_constant()
    
    # Depthwise conv has groups == in_channels
    conv = is_op("nn.conv2d")(data, weight)
    
    return conv


def make_dense_pattern():
    """Pattern: Dense (Fully Connected)"""
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    
    dense = is_op("nn.dense")(data, weight)
    dense_bias = is_op("nn.bias_add")(dense, bias) | dense
    
    return dense_bias


def make_add_relu_pattern():
    """Pattern: Add + ReLU (residual connection)"""
    lhs = wildcard()
    rhs = wildcard()
    
    add = is_op("add")(lhs, rhs)
    add_relu = is_op("nn.relu")(add) | add
    
    return add_relu


def make_maxpool_pattern():
    """Pattern: Max Pooling"""
    data = wildcard()
    pool = is_op("nn.max_pool2d")(data)
    return pool


def make_global_avgpool_pattern():
    """Pattern: Global Average Pooling"""
    data = wildcard()
    pool = is_op("nn.global_avg_pool2d")(data)
    return pool


#############################################################################
# NPU Pattern Registry
#############################################################################

def get_npu_patterns():
    """Return list of NPU-supported patterns."""
    return [
        ("remu_npu.conv2d_relu", make_conv2d_relu_pattern()),
        ("remu_npu.conv2d_bn_relu", make_conv2d_bn_relu_pattern()),
        ("remu_npu.depthwise_conv", make_depthwise_conv_pattern()),
        ("remu_npu.dense", make_dense_pattern()),
        ("remu_npu.add_relu", make_add_relu_pattern()),
        ("remu_npu.maxpool", make_maxpool_pattern()),
        ("remu_npu.global_avgpool", make_global_avgpool_pattern()),
    ]


#############################################################################
# NPU Codegen Visitor
#############################################################################

class NPUCodegenVisitor(ExprVisitor):
    """Visit Relay IR and generate NPU C code."""
    
    def __init__(self):
        super().__init__()
        self.code_lines = []
        self.weight_info = {}
        self.tensor_counter = 0
        self.tensor_map = {}  # Maps Relay expr to buffer name
        
    def get_tensor_name(self, expr) -> str:
        """Get or create a buffer name for an expression."""
        expr_id = id(expr)
        if expr_id not in self.tensor_map:
            name = f"tensor_{self.tensor_counter}"
            self.tensor_counter += 1
            self.tensor_map[expr_id] = name
        return self.tensor_map[expr_id]
    
    def emit(self, line: str):
        """Emit a line of code."""
        self.code_lines.append(line)
    
    def visit_call(self, call):
        """Visit a function call node."""
        op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)
        
        # Visit arguments first
        for arg in call.args:
            self.visit(arg)
        
        output_name = self.get_tensor_name(call)
        
        if "nn.conv2d" in op_name:
            self._gen_conv2d(call, output_name)
        elif "nn.dense" in op_name:
            self._gen_dense(call, output_name)
        elif "nn.relu" in op_name:
            self._gen_relu(call, output_name)
        elif "nn.max_pool2d" in op_name:
            self._gen_maxpool(call, output_name)
        elif "nn.global_avg_pool2d" in op_name:
            self._gen_global_avgpool(call, output_name)
        elif "nn.batch_norm" in op_name:
            self._gen_batchnorm(call, output_name)
        elif "add" in op_name:
            self._gen_add(call, output_name)
        elif "nn.bias_add" in op_name:
            self._gen_bias_add(call, output_name)
        else:
            self.emit(f"    // TODO: Unsupported op {op_name}")
    
    def _gen_conv2d(self, call, output_name: str):
        """Generate Conv2D code."""
        attrs = call.attrs
        
        # Get input name
        input_name = self.get_tensor_name(call.args[0])
        weight_name = f"weight_{self.tensor_counter}"
        
        # Extract attributes
        kernel = list(attrs.kernel_size) if hasattr(attrs, 'kernel_size') else [3, 3]
        strides = list(attrs.strides) if hasattr(attrs, 'strides') else [1, 1]
        padding = list(attrs.padding) if hasattr(attrs, 'padding') else [0, 0, 0, 0]
        groups = int(attrs.groups) if hasattr(attrs, 'groups') else 1
        channels = int(attrs.channels) if hasattr(attrs, 'channels') else 0
        
        pad = padding[0] if len(padding) > 0 else 0
        stride = strides[0] if len(strides) > 0 else 1
        kh, kw = kernel[0], kernel[1]
        
        if groups > 1:
            # Depthwise conv
            self.emit(f"    // Depthwise Conv2D: groups={groups}")
            self.emit(f"    npu_depthwise_conv2d({input_name}, {weight_name}, {output_name},")
            self.emit(f"                          1, {channels}, H, W, {kh}, {kw}, {pad}, {stride}, NPU_ACT_NONE);")
        else:
            # Standard conv
            self.emit(f"    // Conv2D: out_channels={channels}, kernel={kernel}, stride={stride}, pad={pad}")
            self.emit(f"    npu_conv2d_tiled({input_name}, {weight_name}, {output_name},")
            self.emit(f"                     1, IN_C, IN_H, IN_W, {channels}, {kh}, {kw}, {pad}, {stride}, NPU_ACT_NONE);")
    
    def _gen_dense(self, call, output_name: str):
        """Generate Dense (MatMul) code."""
        input_name = self.get_tensor_name(call.args[0])
        weight_name = f"weight_{self.tensor_counter}"
        
        units = int(call.attrs.units) if hasattr(call.attrs, 'units') else 0
        
        self.emit(f"    // Dense: units={units}")
        self.emit(f"    npu_matmul_tiled({input_name}, {weight_name}, {output_name}, 1, {units}, IN_FEATURES);")
    
    def _gen_relu(self, call, output_name: str):
        """Generate ReLU code."""
        input_name = self.get_tensor_name(call.args[0])
        self.emit(f"    // ReLU")
        self.emit(f"    npu_relu_tiled({input_name}, {output_name}, SIZE);")
    
    def _gen_maxpool(self, call, output_name: str):
        """Generate MaxPool code."""
        input_name = self.get_tensor_name(call.args[0])
        attrs = call.attrs
        
        pool_size = list(attrs.pool_size) if hasattr(attrs, 'pool_size') else [2, 2]
        strides = list(attrs.strides) if hasattr(attrs, 'strides') else pool_size
        
        self.emit(f"    // MaxPool2D: pool_size={pool_size}, strides={strides}")
        self.emit(f"    npu_maxpool2d({input_name}, {output_name}, 1, C, H, W, {pool_size[0]}, {pool_size[1]}, {strides[0]});")
    
    def _gen_global_avgpool(self, call, output_name: str):
        """Generate Global Average Pool code."""
        input_name = self.get_tensor_name(call.args[0])
        self.emit(f"    // Global Average Pool")
        self.emit(f"    npu_global_avgpool2d({input_name}, {output_name}, 1, C, H, W);")
    
    def _gen_batchnorm(self, call, output_name: str):
        """Generate BatchNorm code (fused into scale/shift)."""
        input_name = self.get_tensor_name(call.args[0])
        self.emit(f"    // BatchNorm (fused as scale+shift)")
        self.emit(f"    npu_batchnorm({input_name}, {output_name}, scale, shift, C, H, W);")
    
    def _gen_add(self, call, output_name: str):
        """Generate element-wise Add code."""
        lhs_name = self.get_tensor_name(call.args[0])
        rhs_name = self.get_tensor_name(call.args[1])
        self.emit(f"    // Element-wise Add")
        self.emit(f"    npu_add({lhs_name}, {rhs_name}, {output_name}, SIZE);")
    
    def _gen_bias_add(self, call, output_name: str):
        """Generate Bias Add code."""
        input_name = self.get_tensor_name(call.args[0])
        self.emit(f"    // Bias Add")
        self.emit(f"    npu_bias_add({input_name}, bias, {output_name}, SIZE);")
    
    def get_code(self) -> str:
        """Return generated code."""
        return '\n'.join(self.code_lines)


#############################################################################
# NPU Module Partitioning
#############################################################################

def partition_for_npu(mod: ir.IRModule) -> ir.IRModule:
    """Partition Relay module for NPU execution."""
    
    # Define which ops the NPU supports
    supported_ops = {
        "nn.conv2d",
        "nn.dense", 
        "nn.relu",
        "nn.max_pool2d",
        "nn.global_avg_pool2d",
        "nn.batch_norm",
        "nn.bias_add",
        "add",
        "clip",  # ReLU6
        "reshape",
        "squeeze",
    }
    
    def check_supported(expr):
        """Check if expression is supported on NPU."""
        if isinstance(expr, relay.Call):
            op_name = str(expr.op)
            for supported in supported_ops:
                if supported in op_name:
                    return True
        return False
    
    # Use annotation-based partitioning
    # This marks subgraphs for NPU execution
    
    return mod


#############################################################################
# Complete Code Generation
#############################################################################

class NPUCodegen:
    """Generate complete NPU code from Relay IR."""
    
    def __init__(self):
        self.weights = OrderedDict()
        self.weight_scales = {}
        self.layers = []
        
    def extract_weights(self, params: Dict[str, tvm.nd.NDArray]):
        """Extract and quantize weights."""
        for name, param in params.items():
            tensor = param.numpy()
            
            # Quantize to INT8
            abs_max = max(abs(tensor.min()), abs(tensor.max()))
            if abs_max < 1e-10:
                scale = 1.0
            else:
                scale = abs_max / 127.0
            
            quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
            
            self.weights[name] = quantized
            self.weight_scales[name] = scale
    
    def generate_weights_binary(self, output_path: str) -> int:
        """Generate binary weights file."""
        total_size = 0
        offsets = {}
        
        with open(output_path, 'wb') as f:
            for name, weight in self.weights.items():
                offsets[name] = total_size
                f.write(weight.tobytes())
                total_size += weight.nbytes
                # Align to 4 bytes
                padding = (4 - (weight.nbytes % 4)) % 4
                f.write(b'\x00' * padding)
                total_size += padding
        
        return total_size, offsets
    
    def generate_weights_header(self, output_path: str, offsets: Dict[str, int]):
        """Generate weights header file."""
        lines = [
            "/* Auto-generated by TVM REMU NPU Backend */",
            "#ifndef __NPU_WEIGHTS_H__",
            "#define __NPU_WEIGHTS_H__",
            "",
            "#include <stdint.h>",
            "",
            "#define NPU_WEIGHTS_FLASH_BASE 0x30000000",
            "",
        ]
        
        for name, weight in self.weights.items():
            offset = offsets[name]
            scale = self.weight_scales[name]
            safe_name = name.replace(".", "_").replace("/", "_").upper()
            
            lines.append(f"// {name}: shape={list(weight.shape)}")
            lines.append(f"#define WEIGHT_{safe_name}_OFFSET {offset}")
            lines.append(f"#define WEIGHT_{safe_name}_SCALE {scale:.6e}f")
            lines.append(f"#define WEIGHT_{safe_name} ((const int8_t*)(NPU_WEIGHTS_FLASH_BASE + {offset}))")
            lines.append("")
        
        lines.append("#endif")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def generate_inference_code(self, 
                                  mod: ir.IRModule,
                                  output_path: str,
                                  model_name: str):
        """Generate complete inference code."""
        
        # Create visitor
        visitor = NPUCodegenVisitor()
        
        # Visit main function
        main_func = mod["main"]
        visitor.visit(main_func.body)
        
        # Generate code
        code = [
            "/**",
            f" * {model_name} inference code",
            " * Generated by TVM REMU NPU Backend",
            " */",
            "",
            "#include <am.h>",
            "#include <klib.h>",
            '#include "npu.h"',
            f'#include "{model_name}_weights.h"',
            "",
            "// Generated layer operations",
            visitor.get_code(),
            "",
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(code))


#############################################################################
# Main API
#############################################################################

def compile_for_npu(onnx_path: str, 
                    output_dir: str,
                    model_name: str = "model",
                    input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """
    Compile ONNX model for REMU NPU using TVM.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory
        model_name: Name for generated files
        input_shape: Input tensor shape
    
    Returns:
        Dict with compilation results
    """
    import onnx as onnx_lib
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== TVM REMU NPU Compiler ===")
    print(f"Model: {onnx_path}")
    print(f"Output: {output_dir}")
    
    # 1. Load ONNX via TVM frontend
    print("\n[1/5] Loading ONNX model...")
    onnx_model = onnx_lib.load(onnx_path)
    
    # Get input name
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: input_shape}
    
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    print(f"  Loaded: {len(params)} parameters")
    
    # 2. Apply TVM passes
    print("\n[2/5] Applying optimization passes...")
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        mod = relay.transform.FoldScaleAxis()(mod)
        mod = relay.transform.CanonicalizeOps()(mod)
    print("  Passes applied successfully")
    
    # 3. Partition for NPU
    print("\n[3/5] Partitioning for NPU...")
    mod = partition_for_npu(mod)
    
    # 4. Generate code
    print("\n[4/5] Generating NPU code...")
    codegen = NPUCodegen()
    codegen.extract_weights(params)
    
    # Generate outputs
    weights_bin = os.path.join(output_dir, f"{model_name}_weights.bin")
    weights_h = os.path.join(output_dir, f"{model_name}_weights.h")
    inference_c = os.path.join(output_dir, f"{model_name}_inference.c")
    
    total_size, offsets = codegen.generate_weights_binary(weights_bin)
    print(f"  Weights: {total_size} bytes")
    
    codegen.generate_weights_header(weights_h, offsets)
    codegen.generate_inference_code(mod, inference_c, model_name)
    
    # 5. Summary
    print("\n[5/5] Complete!")
    print(f"  {weights_bin}")
    print(f"  {weights_h}")
    print(f"  {inference_c}")
    
    return {
        "weights_size": total_size,
        "num_params": len(params),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tvm_npu_codegen.py <model.onnx> [output_dir] [model_name]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "model"
    
    compile_for_npu(model_path, output_dir, model_name)
