#!/usr/bin/env python3
"""
TVM REMU NPU Compiler - Complete Implementation

This implements a proper TVM-based compilation flow for REMU NPU:
1. Load ONNX model using TVM Relay frontend
2. Apply TVM optimization passes (InferType, FoldConstant, FuseOps, etc.)
3. Analyze the optimized Relay IR to extract layer information
4. Extract and quantize weights with scale tracking
5. Generate complete executable C inference code with NPU API calls

Uses TVM 0.12.0 API

Author: TVM REMU NPU Compiler Team
"""

import os
import sys
import json
import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field

# # Workaround for onnx/ml_dtypes compatibility (TVM 0.12 + onnx 1.15)
# def _patch_ml_dtypes():
#     try:
#         import ml_dtypes
#         if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
#             ml_dtypes.float8_e4m3fnuz = ml_dtypes.float8_e4m3fn
#         if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
#             ml_dtypes.float8_e5m2fnuz = ml_dtypes.float8_e5m2
#         if not hasattr(ml_dtypes, 'int4'):
#             ml_dtypes.int4 = np.int8
#         if not hasattr(ml_dtypes, 'uint4'):
#             ml_dtypes.uint4 = np.uint8
#         if not hasattr(ml_dtypes, 'float4_e2m1fn'):
#             ml_dtypes.float4_e2m1fn = ml_dtypes.bfloat16
#     except:
#         pass
# _patch_ml_dtypes()

import tvm
from tvm import relay, ir, te
from tvm.relay import transform
from tvm.relay.expr import Constant, Var, Call, TupleGetItem, Tuple as RelayTuple
from tvm.relay.function import Function
from tvm.relay.expr_functor import ExprVisitor

import onnx
from onnx import numpy_helper


#############################################################################
# NPU Configuration
#############################################################################

@dataclass
class NPUConfig:
    """REMU NPU hardware configuration."""
    feature_sram_size: int = 16 * 1024  # 16KB
    weight_sram_size: int = 16 * 1024   # 16KB
    output_sram_size: int = 16 * 1024   # 16KB
    gemm_m_max: int = 256
    gemm_n_max: int = 256
    gemm_k_max: int = 256
    flash_base: int = 0x30000000
    mmio_base: int = 0x21000000
    sram_feature: int = 0x21001000
    sram_weight: int = 0x21005000
    sram_output: int = 0x21009000

NPU_CONFIG = NPUConfig()


#############################################################################
# Layer Information
#############################################################################

@dataclass
class LayerInfo:
    """Information about a neural network layer."""
    idx: int
    op_type: str
    name: str
    attrs: Dict[str, Any]
    input_shape: List[int]
    output_shape: List[int]
    weight_name: Optional[str] = None
    bias_name: Optional[str] = None
    
    def to_dict(self):
        return {
            "idx": self.idx,
            "op_type": self.op_type,
            "name": self.name,
            "attrs": self.attrs,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "weight_name": self.weight_name,
            "bias_name": self.bias_name,
        }


#############################################################################
# Weight Quantization
#############################################################################

@dataclass
class QuantizedWeight:
    """Quantized weight tensor with metadata."""
    name: str
    data: np.ndarray  # int8 quantized
    shape: Tuple[int, ...]
    scale: float
    zero_point: int
    original_dtype: str
    offset: int = 0
    
def quantize_symmetric(tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float]:
    """
    Symmetric INT8 quantization.
    
    Args:
        tensor: Float tensor to quantize
        bits: Number of bits (default 8)
        
    Returns:
        (quantized_tensor, scale)
    """
    abs_max = max(abs(tensor.min()), abs(tensor.max()))
    if abs_max < 1e-10:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    
    qmax = (1 << (bits - 1)) - 1  # 127 for int8
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), -qmax - 1, qmax).astype(np.int8)
    return quantized, float(scale)


def quantize_asymmetric(tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, int]:
    """
    Asymmetric INT8 quantization.
    
    Args:
        tensor: Float tensor to quantize
        bits: Number of bits (default 8)
        
    Returns:
        (quantized_tensor, scale, zero_point)
    """
    qmin, qmax = 0, (1 << bits) - 1  # 0-255 for uint8
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    if tensor_max - tensor_min < 1e-10:
        return np.zeros_like(tensor, dtype=np.uint8), 1.0, 0
    
    scale = (tensor_max - tensor_min) / (qmax - qmin)
    zero_point = int(round(qmin - tensor_min / scale))
    zero_point = np.clip(zero_point, qmin, qmax)
    
    quantized = np.clip(np.round(tensor / scale) + zero_point, qmin, qmax).astype(np.uint8)
    return quantized, float(scale), int(zero_point)


#############################################################################
# Relay IR Analyzer
#############################################################################

class RelayAnalyzer(ExprVisitor):
    """
    Analyze TVM Relay IR to extract layer information for NPU code generation.
    
    This visitor traverses the Relay expression graph and extracts:
    - Operation types (conv2d, relu, add, etc.)
    - Operation attributes (kernel size, strides, padding, etc.)
    - Input/output shapes
    - Weight tensor names
    - Bias information for expand_dims->add pattern
    """
    
    def __init__(self):
        super().__init__()
        self.layers: List[LayerInfo] = []
        self.constants: Dict[int, np.ndarray] = {}  # id(const) -> numpy array
        self.var_shapes: Dict[str, List[int]] = {}
        self.layer_idx = 0
        self.weight_mapping: Dict[str, str] = {}  # relay var -> onnx weight name
        # Track expand_dims output for bias add pattern detection
        self.expand_dims_bias: Dict[int, Tuple[np.ndarray, List[int]]] = {}  # id(call) -> (bias_data, shape)
        # Track last conv output channels for bias association
        self.last_conv_channels: int = 0
        self.pending_bias: Optional[np.ndarray] = None
        self.pending_bias_shape: List[int] = []
        
    def _get_shape(self, expr) -> List[int]:
        """Extract shape from expression type."""
        try:
            if hasattr(expr, 'checked_type'):
                t = expr.checked_type
                if hasattr(t, 'shape'):
                    return [int(d) for d in t.shape]
        except:
            pass
        return []
    
    def _extract_attrs(self, call) -> Dict[str, Any]:
        """Extract attributes from Call node."""
        attrs = {}
        if call.attrs is not None:
            for key in dir(call.attrs):
                if key.startswith('_'):
                    continue
                try:
                    val = getattr(call.attrs, key)
                    if isinstance(val, (int, str, bool, float)):
                        attrs[key] = val
                    elif hasattr(val, 'value'):
                        attrs[key] = val.value
                    elif hasattr(val, '__iter__') and not isinstance(val, str):
                        attrs[key] = [int(x) if hasattr(x, '__int__') else x for x in val]
                except:
                    pass
        return attrs
    
    def visit_var(self, var):
        """Record variable shapes."""
        self.var_shapes[var.name_hint] = self._get_shape(var)
        
    def visit_constant(self, const):
        """Record constant tensors."""
        try:
            data = const.data.numpy()
            self.constants[id(const)] = data
        except:
            pass
    
    def visit_call(self, call):
        """Visit Call node and extract layer info."""
        # Visit arguments first (depth-first)
        for arg in call.args:
            self.visit(arg)
        
        # Get op name
        if hasattr(call.op, 'name'):
            op_name = call.op.name
        else:
            op_name = str(call.op)
        
        # Extract attributes
        attrs = self._extract_attrs(call)
        
        # Get shapes
        input_shape = self._get_shape(call.args[0]) if call.args else []
        output_shape = self._get_shape(call)
        
        # Create layer info
        layer = LayerInfo(
            idx=self.layer_idx,
            op_type=op_name,
            name=f"{op_name}_{self.layer_idx}",
            attrs=attrs,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Track weight names for conv/dense
        if "conv2d" in op_name or "dense" in op_name:
            if len(call.args) > 1:
                weight_arg = call.args[1]
                if isinstance(weight_arg, Var):
                    layer.weight_name = weight_arg.name_hint
            # Track last conv output channels for bias association
            if len(output_shape) > 1:
                self.last_conv_channels = output_shape[1]
        
        # Handle expand_dims: this typically reshapes bias [C] -> [C,1,1]
        if "expand_dims" in op_name:
            # Check if input is a constant (the bias tensor)
            if len(call.args) > 0:
                bias_arg = call.args[0]
                if isinstance(bias_arg, Constant):
                    try:
                        bias_data = bias_arg.data.numpy()
                        # Store for later use by the add operation
                        self.expand_dims_bias[id(call)] = (bias_data, output_shape)
                        layer.attrs['_bias_data_id'] = id(call)
                        layer.attrs['_bias_channels'] = len(bias_data) if bias_data.ndim == 1 else bias_data.shape[0]
                    except:
                        pass
        
        # Handle add: check if this is bias add (one input from expand_dims)
        if op_name == "add":
            for i, arg in enumerate(call.args):
                if id(arg) in self.expand_dims_bias:
                    bias_data, bias_shape = self.expand_dims_bias[id(arg)]
                    layer.attrs['_is_bias_add'] = True
                    layer.attrs['_bias_channels'] = len(bias_data) if bias_data.ndim == 1 else bias_data.shape[0]
                    # Mark which argument has the bias
                    layer.attrs['_bias_arg_idx'] = i
                    break
        
        self.layers.append(layer)
        self.layer_idx += 1
        
        return call


#############################################################################
# Code Generator
#############################################################################

class NPUCodeGenerator:
    """
    Generate C code for REMU NPU execution.
    
    Produces:
    - weights.bin: Binary blob of all quantized weights
    - weights.h: Header with weight offsets and scales
    - inference.c: Complete inference code with NPU API calls
    - layers.json: Layer metadata for debugging
    """
    
    def __init__(self, model_name: str, config: NPUConfig = NPU_CONFIG):
        self.model_name = model_name
        self.config = config
        self.weights: OrderedDict[str, QuantizedWeight] = OrderedDict()
        self.current_offset = 0
        self.layers: List[LayerInfo] = []
        self.bias_scales: Dict[str, float] = {}  # bias_name -> scale
        
    def add_weight(self, name: str, tensor: np.ndarray) -> QuantizedWeight:
        """Add a weight tensor with quantization."""
        # Quantize
        q_tensor, scale = quantize_symmetric(tensor.astype(np.float32))
        
        # Create weight info
        weight = QuantizedWeight(
            name=name,
            data=q_tensor,
            shape=tensor.shape,
            scale=scale,
            zero_point=0,
            original_dtype=str(tensor.dtype),
            offset=self.current_offset,
        )
        
        self.weights[name] = weight
        self.current_offset += q_tensor.nbytes
        # Align to 4 bytes
        self.current_offset = (self.current_offset + 3) & ~3
        
        return weight
    
    def add_bias(self, name: str, tensor: np.ndarray, input_scale: float, weight_scale: float) -> QuantizedWeight:
        """Add a bias tensor with proper scaling for int32 accumulator.
        
        Bias needs to be quantized to int32 to match the scale of the accumulator:
        accumulator_scale = input_scale * weight_scale
        bias_quantized = bias_float / accumulator_scale
        """
        acc_scale = input_scale * weight_scale
        # Quantize bias to int32
        q_tensor = np.round(tensor.astype(np.float32) / acc_scale).astype(np.int32)
        
        # Create weight info
        weight = QuantizedWeight(
            name=name,
            data=q_tensor,
            shape=tensor.shape,
            scale=acc_scale,
            zero_point=0,
            original_dtype=str(tensor.dtype),
            offset=self.current_offset,
        )
        
        self.weights[name] = weight
        self.bias_scales[name] = acc_scale
        self.current_offset += q_tensor.nbytes
        # Align to 4 bytes
        self.current_offset = (self.current_offset + 3) & ~3
        
        return weight
    
    def _safe_name(self, name: str) -> str:
        """Convert weight name to safe C identifier."""
        return name.replace(".", "_").replace("/", "_").replace(":", "_").replace("-", "_").upper()
    
    def generate_weights_binary(self, path: str):
        """Generate binary weights file."""
        with open(path, 'wb') as f:
            for name, weight in self.weights.items():
                f.write(weight.data.tobytes())
                # Padding for alignment
                padding = (4 - (weight.data.nbytes % 4)) % 4
                f.write(b'\x00' * padding)
        print(f"  Weights binary: {path} ({self.current_offset:,} bytes)")
        
    def generate_weights_header(self, path: str):
        """Generate weights header file."""
        lines = [
            "/**",
            f" * {self.model_name} weights header",
            " * Generated by TVM REMU NPU Compiler",
            " *",
            " * This file contains weight tensor offsets, sizes, and quantization scales",
            " */",
            "",
            f"#ifndef __{self.model_name.upper()}_WEIGHTS_H__",
            f"#define __{self.model_name.upper()}_WEIGHTS_H__",
            "",
            "#include <stdint.h>",
            "",
            "// Flash base address for weight storage",
            f"#define WEIGHTS_FLASH_BASE 0x{self.config.flash_base:08X}",
            f"#define WEIGHTS_TOTAL_SIZE {self.current_offset}",
            "",
            "// Weight tensor definitions",
        ]
        
        for name, weight in self.weights.items():
            safe_name = self._safe_name(name)
            shape_str = "x".join(str(d) for d in weight.shape)
            
            lines.append(f"")
            lines.append(f"// {name}: shape=[{', '.join(str(d) for d in weight.shape)}], scale={weight.scale:.6e}")
            lines.append(f"#define WEIGHT_{safe_name}_OFFSET {weight.offset}")
            lines.append(f"#define WEIGHT_{safe_name}_SIZE {weight.data.nbytes}")
            lines.append(f"#define WEIGHT_{safe_name}_SCALE {weight.scale:.6e}f")
            lines.append(f"#define WEIGHT_{safe_name}_SCALE_Q16 {int(weight.scale * 65536)}")
            lines.append(f"#define WEIGHT_{safe_name} ((const int8_t*)(WEIGHTS_FLASH_BASE + {weight.offset}))")
            
            # Add shape macros for weights
            for i, dim in enumerate(weight.shape):
                lines.append(f"#define WEIGHT_{safe_name}_DIM{i} {dim}")
        
        lines.extend([
            "",
            f"#endif // __{self.model_name.upper()}_WEIGHTS_H__",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Weights header: {path}")
    
    def _compute_output_size(self, in_h: int, in_w: int, kh: int, kw: int, 
                              pad: int, stride: int) -> Tuple[int, int]:
        """Compute output spatial dimensions for conv/pool."""
        out_h = (in_h + 2 * pad - kh) // stride + 1
        out_w = (in_w + 2 * pad - kw) // stride + 1
        return out_h, out_w
    
    def generate_inference_code(self, path: str, layers: List[LayerInfo], 
                                 input_shape: Tuple, onnx_weights: Dict,
                                 conv_bias_map: Dict[str, str] = None,
                                 dense_bias_map: Dict[str, str] = None):
        """Generate complete C inference code with NPU API calls.
        
        Args:
            path: Output file path
            layers: List of layer information from TVM analysis
            input_shape: Input tensor shape (N, C, H, W)
            onnx_weights: Dictionary of ONNX weights
            conv_bias_map: Mapping from conv weight name to bias name
            dense_bias_map: Mapping from dense weight name to bias name
        """
        if conv_bias_map is None:
            conv_bias_map = {}
        if dense_bias_map is None:
            dense_bias_map = {}
            
        N, C, H, W = input_shape
        
        # Find all conv layers and their weights
        conv_layers = []
        weight_info = {}
        
        for layer in layers:
            if "conv2d" in layer.op_type:
                conv_layers.append(layer)
            if layer.weight_name and layer.weight_name in onnx_weights:
                weight_info[layer.idx] = {
                    "weight_name": layer.weight_name,
                    "shape": onnx_weights[layer.weight_name].shape,
                }
        
        # Calculate actual maximum activation buffer size
        # Find the maximum C * H * W across all layer outputs
        max_act_size = 0
        for layer in layers:
            if len(layer.output_shape) >= 4:
                act_size = layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3]
                max_act_size = max(max_act_size, act_size)
        
        if max_act_size == 0:
            max_act_size = C * H * W  # Fallback to input size
        
        # Add some margin
        max_act_size = max(max_act_size, C * H * W)
        
        lines = [
            "/**",
            f" * {self.model_name} inference code for REMU NPU",
            " * Generated by TVM REMU NPU Compiler",
            " *",
            f" * Input shape: {input_shape}",
            f" * Total operations: {len(layers)}",
            f" * Conv2D layers: {len(conv_layers)}",
            " */",
            "",
            "#include <am.h>",
            "#include <klib.h>",
            '#include "npu.h"',
            '#include "npu_ops.h"',
            f'#include "{self.model_name}_weights.h"',
            "",
            f"// Input tensor dimensions",
            f"#define INPUT_N {N}",
            f"#define INPUT_C {C}",
            f"#define INPUT_H {H}",
            f"#define INPUT_W {W}",
            f"#define INPUT_SIZE ({C} * {H} * {W})",
            "",
            f"// Maximum buffer size for intermediate activations",
            f"// Calculated from max layer output: {max_act_size} elements",
            f"#define MAX_ACT_SIZE {max_act_size}",
            "",
            "// Double-buffered activation memory",
            "static int8_t act_buf_0[MAX_ACT_SIZE] __attribute__((aligned(4)));",
            "static int8_t act_buf_1[MAX_ACT_SIZE] __attribute__((aligned(4)));",
            "static int32_t acc_buf[MAX_ACT_SIZE] __attribute__((aligned(4)));",
            "",
            "// Residual buffer for skip connections",
            "static int8_t residual_buf[MAX_ACT_SIZE] __attribute__((aligned(4)));",
            "",
        ]
        
        # Generate layer shape table
        lines.append("// Layer output shapes (for debugging)")
        lines.append("typedef struct {")
        lines.append("    int n, c, h, w;")
        lines.append("} TensorShape;")
        lines.append("")
        lines.append(f"static const TensorShape layer_shapes[{len(layers)}] = {{")
        for layer in layers:
            shape = layer.output_shape + [1, 1, 1, 1]  # Pad to 4 dims
            lines.append(f"    {{ {shape[0]}, {shape[1]}, {shape[2]}, {shape[3]} }},  // {layer.name}")
        lines.append("};")
        lines.append("")
        
        # Generate inference function
        lines.extend([
            "/**",
            f" * Run {self.model_name} inference on input tensor",
            " *",
            " * @param input  Input tensor, int8_t[batch, channels, height, width]",
            " * @param output Output tensor, int32_t for raw scores or int8_t for quantized",
            " * @return 0 on success",
            " */",
            f"int {self.model_name}_inference(const int8_t* input, int32_t* output) {{",
            "    // Initialize NPU",
            "    npu_reset();",
            "",
            "    // Copy input to activation buffer 0",
            "    memcpy(act_buf_0, input, INPUT_SIZE);",
            "",
            "    // Track current activation buffer",
            "    int8_t* cur_input = act_buf_0;",
            "    int8_t* cur_output = act_buf_1;",
            "    int32_t* cur_acc = acc_buf;",
            "",
            "    // Current tensor dimensions",
            "    int cur_n = INPUT_N;",
            "    int cur_c = INPUT_C;",
            "    int cur_h = INPUT_H;",
            "    int cur_w = INPUT_W;",
            "",
        ])
        
        # Track state for layer generation
        residual_stack = []  # For skip connections
        cur_shape = list(input_shape)
        last_conv_channels = 0  # For tracking bias add
        
        # Generate layer-by-layer execution
        for i, layer in enumerate(layers):
            op = layer.op_type
            attrs = layer.attrs
            out_shape = layer.output_shape if layer.output_shape else cur_shape
            
            lines.append(f"    // === Layer {i}: {op} ===")
            
            if "nn.conv2d" in op:
                # Extract conv parameters
                kh = attrs.get("kernel_size", [3, 3])
                kw = kh[1] if isinstance(kh, list) else kh
                kh = kh[0] if isinstance(kh, list) else kh
                
                strides = attrs.get("strides", [1, 1])
                stride = strides[0] if isinstance(strides, list) else strides
                
                padding = attrs.get("padding", [0, 0, 0, 0])
                if isinstance(padding, list):
                    pad = padding[0] if len(padding) > 0 else 0
                else:
                    pad = padding
                
                groups = attrs.get("groups", 1)
                # Use TVM's output shape for channels
                channels = out_shape[1] if len(out_shape) > 1 else 32
                
                # Find weight tensor
                weight_name = layer.weight_name
                safe_weight = self._safe_name(weight_name) if weight_name else f"CONV{i}"
                
                # Check if this conv has a bias
                bias_name = conv_bias_map.get(weight_name) if weight_name else None
                safe_bias = self._safe_name(bias_name) if bias_name else None
                
                if groups > 1 and groups == cur_shape[1]:
                    # Depthwise convolution
                    lines.extend([
                        f"    // Depthwise Conv2D: {cur_shape[1]} channels, kernel {kh}x{kw}, stride {stride}, pad {pad}",
                        f"    npu_depthwise_conv2d(",
                        f"        cur_input,",
                        f"        (int8_t*)WEIGHT_{safe_weight},",
                        f"        cur_acc,",
                        f"        cur_n, cur_c, cur_h, cur_w,",
                        f"        {kh}, {kw}, {pad}, {stride},",
                        f"        NPU_ACT_NONE",
                        f"    );",
                    ])
                else:
                    # Standard convolution
                    lines.extend([
                        f"    // Conv2D: {cur_shape[1]} -> {channels} channels, kernel {kh}x{kw}, stride {stride}, pad {pad}",
                        f"    npu_conv2d(",
                        f"        cur_input,",
                        f"        (int8_t*)WEIGHT_{safe_weight},",
                        f"        cur_acc,",
                        f"        cur_n, cur_c, cur_h, cur_w,",
                        f"        {channels}, {kh}, {kw}, {pad}, {stride},",
                        f"        NPU_ACT_NONE",
                        f"    );",
                    ])
                
                # Use TVM's output shape directly (it handles padding correctly)
                cur_shape = list(out_shape) if len(out_shape) >= 4 else [1, channels, cur_shape[2], cur_shape[3]]
                last_conv_channels = channels
                
                lines.extend([
                    f"    // Output: {cur_shape}",
                    f"    cur_c = {cur_shape[1]};",
                    f"    cur_h = {cur_shape[2]};",
                    f"    cur_w = {cur_shape[3]};",
                ])
                
                # Add bias if present (fused into conv)
                if bias_name and safe_bias:
                    lines.extend([
                        f"    // Add bias (fused): {bias_name}",
                        f"    {{",
                        f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                        f"        int spatial_size = cur_h * cur_w;",
                        f"        for (int c = 0; c < cur_c; c++) {{",
                        f"            int32_t bias_val = bias[c];",
                        f"            int32_t* channel_ptr = cur_acc + c * spatial_size;",
                        f"            for (int s = 0; s < spatial_size; s++) {{",
                        f"                channel_ptr[s] += bias_val;",
                        f"            }}",
                        f"        }}",
                        f"    }}",
                    ])
                
                lines.append("")
                
            elif "nn.bias_add" in op:
                # Bias addition after conv - should be handled by conv above
                lines.extend([
                    f"    // Bias add (already fused into conv)",
                    "",
                ])
                
            elif "expand_dims" in op:
                # This is typically bias broadcast - skip it (handled by conv bias fusion)
                lines.extend([
                    f"    // expand_dims: Bias reshape (skipped - fused into conv)",
                    "",
                ])
                
            elif "nn.relu" in op:
                size = cur_shape[1] * cur_shape[2] * cur_shape[3]
                lines.extend([
                    f"    // ReLU activation",
                    f"    npu_relu_elementwise(cur_acc, cur_acc, {size}, 1);  // dtype=1 for int32",
                    f"    // Requantize int32 -> int8 for next layer",
                    f"    npu_requantize_shift(cur_acc, cur_output, {size}, 8);",
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                    "",
                ])
                
            elif "clip" in op:
                # ReLU6 or generic clip
                a_min = attrs.get("a_min", 0)
                a_max = attrs.get("a_max", 6)
                size = cur_shape[1] * cur_shape[2] * cur_shape[3]
                
                if a_min == 0 and a_max == 6:
                    lines.extend([
                        f"    // ReLU6 (Clip [0, 6]) in int32 domain",
                        f"    // Apply clip before requantize: max = 6 << 8 (shift=8)",
                        f"    npu_clip_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1, 0, {6 << 8});",
                    ])
                else:
                    lines.extend([
                        f"    // Clip [{a_min}, {a_max}]",
                        f"    npu_clip_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1, {int(a_min)}, {int(a_max)});",
                    ])
                
                # After activation, requantize back to int8 for next layer
                lines.extend([
                    f"    // Requantize int32 -> int8 for next layer",
                    f"    npu_requantize_shift(cur_acc, cur_output, cur_c * cur_h * cur_w, 8);",
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                    "",
                ])
                
            elif "add" in op and "bias" not in op:
                # Check if this is bias add (following expand_dims) - skip if we already fused it
                is_bias_add = attrs.get("_is_bias_add", False)
                
                # Check if this add follows a dense layer (dense bias add pattern from TVM Gemm decomposition)
                is_dense_bias_add = (i > 0 and "nn.dense" in layers[i-1].op_type)
                
                if is_bias_add:
                    # Bias add already handled in conv fusion - skip
                    lines.extend([
                        f"    // Bias add (skipped - already fused into conv)",
                        "",
                    ])
                elif is_dense_bias_add:
                    # Dense bias add - already fused into dense layer - skip
                    lines.extend([
                        f"    // Dense bias add (skipped - already fused into nn.dense)",
                        "",
                    ])
                elif len(out_shape) >= 4 and out_shape[2] > 1 and out_shape[3] > 1:
                    # Likely bias add from TVM decomposition - should be fused already
                    lines.extend([
                        f"    // Spatial add (skipped - bias fused into conv)",
                        "",
                    ])
                else:
                    # Element-wise add (residual connection)
                    lines.extend([
                        f"    // Element-wise add (residual connection)",
                        f"    npu_add_i32(cur_acc, (int32_t*)residual_buf, cur_acc, cur_c * cur_h * cur_w);",
                        "",
                    ])
                # Update shape from TVM
                if len(out_shape) >= 4:
                    cur_shape = list(out_shape)
                
            elif "nn.global_avg_pool2d" in op:
                lines.extend([
                    f"    // Global Average Pooling: [{cur_shape[1]}, {cur_shape[2]}, {cur_shape[3]}] -> [{cur_shape[1]}, 1, 1]",
                    f"    npu_global_avgpool2d(cur_input, cur_acc, cur_n, cur_c, cur_h, cur_w);",
                    f"    cur_h = 1;",
                    f"    cur_w = 1;",
                    f"    // Requantize for dense layer input",
                    f"    npu_requantize_shift(cur_acc, cur_output, cur_c, 8);",
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                    "",
                ])
                cur_shape[2] = 1
                cur_shape[3] = 1
                
            elif "nn.max_pool2d" in op:
                pool_size = attrs.get("pool_size", [2, 2])
                kh = pool_size[0] if isinstance(pool_size, list) else pool_size
                kw = pool_size[1] if isinstance(pool_size, list) else pool_size
                strides = attrs.get("strides", [2, 2])
                stride = strides[0] if isinstance(strides, list) else strides
                padding = attrs.get("padding", [0, 0, 0, 0])
                pad = padding[0] if isinstance(padding, list) else padding
                
                lines.extend([
                    f"    // Max Pooling {kh}x{kw}, stride {stride}",
                    f"    npu_maxpool2d(cur_input, cur_output, cur_n, cur_c, cur_h, cur_w,",
                    f"                  {kh}, {kw}, {stride}, {pad});",
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                ])
                
                # Use TVM's output shape
                if len(out_shape) >= 4:
                    cur_shape = list(out_shape)
                
                lines.extend([
                    f"    cur_h = {cur_shape[2]};",
                    f"    cur_w = {cur_shape[3]};",
                    "",
                ])
                
            elif "nn.avg_pool2d" in op:
                pool_size = attrs.get("pool_size", [2, 2])
                kh = pool_size[0] if isinstance(pool_size, list) else pool_size
                kw = pool_size[1] if isinstance(pool_size, list) else pool_size
                strides = attrs.get("strides", [2, 2])
                stride = strides[0] if isinstance(strides, list) else strides
                padding = attrs.get("padding", [0, 0, 0, 0])
                pad = padding[0] if isinstance(padding, list) else padding
                
                lines.extend([
                    f"    // Average Pooling {kh}x{kw}, stride {stride}",
                    f"    npu_avgpool2d(cur_input, cur_output, cur_n, cur_c, cur_h, cur_w,",
                    f"                  {kh}, {kw}, {stride}, {pad});",
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                ])
                
                # Use TVM's output shape
                if len(out_shape) >= 4:
                    cur_shape = list(out_shape)
                
                lines.extend([
                    f"    cur_h = {cur_shape[2]};",
                    f"    cur_w = {cur_shape[3]};",
                    "",
                ])
                
            elif "nn.dense" in op:
                units = attrs.get("units", layer.output_shape[-1] if layer.output_shape else 1000)
                weight_name = layer.weight_name
                safe_weight = self._safe_name(weight_name) if weight_name else f"DENSE{i}"
                
                # Check if this dense has a bias
                bias_name = dense_bias_map.get(weight_name) if weight_name else None
                safe_bias = self._safe_name(bias_name) if bias_name else None
                
                # After global avgpool, data is [N, C, 1, 1], flatten to [N, C]
                in_features = cur_shape[1] * cur_shape[2] * cur_shape[3]
                
                lines.extend([
                    f"    // Dense (Fully Connected): {in_features} -> {units}",
                    f"    // Input from global avgpool is already flat [1, {in_features}]",
                    f"    npu_matmul(",
                    f"        cur_input,",
                    f"        (int8_t*)WEIGHT_{safe_weight},",
                    f"        cur_acc,",
                    f"        1, {units}, {in_features}",
                    f"    );",
                    f"    cur_c = {units};",
                    f"    cur_h = 1;",
                    f"    cur_w = 1;",
                ])
                
                # Add bias if present
                if bias_name and safe_bias:
                    lines.extend([
                        f"    // Add bias (fused): {bias_name}",
                        f"    {{",
                        f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                        f"        for (int i = 0; i < {units}; i++) {{",
                        f"            cur_acc[i] += bias[i];",
                        f"        }}",
                        f"    }}",
                    ])
                
                lines.append("")
                cur_shape = [cur_shape[0], units, 1, 1]
                
            elif "nn.batch_flatten" in op or "reshape" in op or "squeeze" in op:
                # Reshape operations - preserve dimensions for classifier
                # If output shape is empty/invalid, keep current shape
                if len(out_shape) >= 2:
                    lines.extend([
                        f"    // Reshape/Flatten: output shape {out_shape}",
                        f"    // No computation needed, just dimension tracking",
                    ])
                    # Pad output shape to 4 dims for tracking
                    new_shape = list(out_shape) + [1, 1, 1, 1]
                    cur_shape = new_shape[:4]
                    lines.extend([
                        f"    cur_c = {cur_shape[1] if len(out_shape) > 1 else cur_shape[1]};",
                        f"    cur_h = {cur_shape[2] if len(out_shape) > 2 else 1};",
                        f"    cur_w = {cur_shape[3] if len(out_shape) > 3 else 1};",
                        "",
                    ])
                else:
                    # Shape invalid, keep current (common for dyn.reshape)
                    lines.extend([
                        f"    // Reshape/Flatten (shape unchanged for classifier)",
                        f"    // Keeping dimensions: [{cur_shape[1]}, {cur_shape[2]}, {cur_shape[3]}]",
                        "",
                    ])
                
            elif "concatenate" in op:
                # Concatenation - usually for shape inference, preserve dimensions
                if len(out_shape) >= 2:
                    lines.extend([
                        f"    // Concatenate: output shape {out_shape}",
                        f"    // Dimensions updated from TVM shape inference",
                    ])
                    new_shape = list(out_shape) + [1, 1, 1, 1]
                    cur_shape = new_shape[:4]
                    lines.extend([
                        f"    cur_c = {cur_shape[1]};",
                        f"    cur_h = {cur_shape[2] if len(out_shape) > 2 else 1};",
                        f"    cur_w = {cur_shape[3] if len(out_shape) > 3 else 1};",
                        "",
                    ])
                else:
                    # For classifier path, concatenate is shape manipulation, keep dims
                    lines.extend([
                        f"    // Concatenate (shape manipulation for classifier, dimensions unchanged)",
                        "",
                    ])
                    
            elif "multiply" in op:
                size = cur_shape[1] * cur_shape[2] * cur_shape[3]
                lines.extend([
                    f"    // Element-wise multiply",
                    f"    npu_mul(cur_input, cur_output, cur_input, {size});",
                    "",
                ])
                
            else:
                # Unknown op - add comment
                lines.append(f"    // TODO: {op} - not implemented")
                lines.append("")
        
        # Copy final output
        final_size = cur_shape[1] * cur_shape[2] * cur_shape[3]
        lines.extend([
            f"    // Copy final output ({final_size} elements)",
            f"    memcpy(output, cur_acc, {final_size} * sizeof(int32_t));",
            "",
            "    return 0;",
            "}",
            "",
        ])
        
        # Generate stats function
        lines.extend([
            "/**",
            " * Print NPU performance statistics",
            " */",
            f"void {self.model_name}_print_stats(void) {{",
            '    printf("=== NPU Performance Stats ===\\n");',
            '    printf("Cycles:        %u\\n", npu_get_cycles());',
            '    printf("Memory Traffic: %u bytes\\n", npu_get_mem_bytes());',
            '    printf("GEMM Ops:      %u\\n", npu_get_gemm_count());',
            '    printf("Activation Ops: %u\\n", npu_get_act_count());',
            '    printf("DMA Transfers:  %u\\n", npu_get_dma_count());',
            "}",
            "",
        ])
        
        # Generate validation function
        lines.extend([
            "/**",
            " * Validate model configuration",
            " */",
            f"int {self.model_name}_validate(void) {{",
            f'    printf("Model: {self.model_name}\\n");',
            f'    printf("Input:  [%d, %d, %d, %d]\\n", INPUT_N, INPUT_C, INPUT_H, INPUT_W);',
            f'    printf("Weights: %d bytes @ 0x%08x\\n", WEIGHTS_TOTAL_SIZE, WEIGHTS_FLASH_BASE);',
            f'    printf("Layers: {len(layers)}\\n");',
            "    return 0;",
            "}",
            "",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Inference code: {path}")


#############################################################################
# Test Data and Program Generation
#############################################################################

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


#############################################################################
# Main Compiler
#############################################################################

def compile_model(onnx_path: str, 
                  output_dir: str,
                  model_name: str = "model",
                  input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """
    Compile ONNX model for REMU NPU using TVM.
    
    Args:
        onnx_path: Path to ONNX model file
        output_dir: Directory for output files
        model_name: Name for generated files
        input_shape: Input tensor shape (N, C, H, W)
        
    Returns:
        Dictionary with compilation statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TVM REMU NPU Compiler")
    print("=" * 70)
    print(f"Model:       {onnx_path}")
    print(f"Output:      {output_dir}")
    print(f"Input shape: {input_shape}")
    print(f"TVM version: {tvm.__version__}")
    
    # 1. Load ONNX model
    print("\n[1/6] Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    print(f"  ONNX opset: {onnx_model.opset_import[0].version}")
    
    # Get input name from model
    input_name = onnx_model.graph.input[0].name
    print(f"  Input tensor: {input_name}")
    
    # 2. Extract weights directly from ONNX
    print("\n[2/6] Extracting weights from ONNX...")
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        tensor = numpy_helper.to_array(init)
        onnx_weights[init.name] = tensor
    print(f"  Found {len(onnx_weights)} weight tensors")
    
    # Extract Conv-Bias mapping from ONNX graph
    # In ONNX, Conv nodes have inputs: [input, weight, bias?]
    conv_bias_map = {}  # weight_name -> bias_name
    dense_bias_map = {}  # weight_name -> bias_name (for Gemm/MatMul)
    
    for node in onnx_model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 3:
            weight_name = node.input[1]
            bias_name = node.input[2]
            if bias_name in onnx_weights:
                conv_bias_map[weight_name] = bias_name
                print(f"  Conv bias mapping: {weight_name} -> {bias_name}")
        elif node.op_type == "Gemm" and len(node.input) >= 3:
            weight_name = node.input[1]
            bias_name = node.input[2]
            if bias_name in onnx_weights:
                dense_bias_map[weight_name] = bias_name
                print(f"  Dense bias mapping: {weight_name} -> {bias_name}")
    
    print(f"  Found {len(conv_bias_map)} Conv layers with bias")
    print(f"  Found {len(dense_bias_map)} Dense layers with bias")
    
    # Print weight summary
    total_params = sum(w.size for w in onnx_weights.values())
    total_bytes = sum(w.nbytes for w in onnx_weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size (float): {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    
    # 3. Convert to TVM Relay IR
    print("\n[3/6] Converting to TVM Relay IR...")
    shape_dict = {input_name: input_shape}
    
    mod, params = relay.frontend.from_onnx(
        onnx_model, 
        shape_dict,
        freeze_params=False
    )
    print(f"  Relay module created with {len(params)} parameters")
    
    # 4. Apply TVM optimization passes
    print("\n[4/6] Applying TVM optimization passes...")
    with tvm.transform.PassContext(opt_level=3):
        # Type inference
        mod = relay.transform.InferType()(mod)
        print("    - InferType")
        
        # Constant folding
        mod = relay.transform.FoldConstant()(mod)
        print("    - FoldConstant")
        
        # Simplify inference (remove training-only ops)
        mod = relay.transform.SimplifyInference()(mod)
        print("    - SimplifyInference")
        
        # Fold scale into weights
        mod = relay.transform.FoldScaleAxis()(mod)
        print("    - FoldScaleAxis")
        
        # Canonicalize operations
        mod = relay.transform.CanonicalizeOps()(mod)
        print("    - CanonicalizeOps")
        
        # Dead code elimination
        mod = relay.transform.DeadCodeElimination()(mod)
        print("    - DeadCodeElimination")
    
    # Analyze the optimized graph
    print("\n[5/6] Analyzing optimized graph...")
    analyzer = RelayAnalyzer()
    analyzer.visit(mod["main"])
    layers = analyzer.layers
    
    print(f"  Total operations: {len(layers)}")
    
    # Count operations
    op_counts = {}
    for layer in layers:
        op = layer.op_type
        op_counts[op] = op_counts.get(op, 0) + 1
    
    print("  Operation breakdown:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")
    
    # 5. Generate code
    print("\n[6/6] Generating NPU code...")
    codegen = NPUCodeGenerator(model_name)
    
    # Track weight scales for bias quantization
    weight_scales = {}  # weight_name -> scale
    
    # Add weights with quantization
    # First pass: add conv/fc weights and record their scales
    for name, tensor in onnx_weights.items():
        if tensor.dtype in [np.float32, np.float16, np.float64]:
            # Check if this is a bias tensor
            is_conv_bias = name in conv_bias_map.values()
            is_dense_bias = name in dense_bias_map.values()
            
            if is_conv_bias or is_dense_bias:
                # Skip bias for now, will add with proper scale later
                continue
            
            # Dense weights in ONNX are typically [out_features, in_features],
            # but npu_matmul expects B in [K, N] layout => transpose to [in, out].
            if name in dense_bias_map and tensor.ndim == 2:
                tensor = tensor.T
            w = codegen.add_weight(name, tensor)
            weight_scales[name] = w.scale
    
    # Second pass: add conv bias tensors with proper accumulator scale
    # Assume input scale ~= 1/127 for quantized input, or use tracked scale
    input_scale = 1.0 / 127.0  # Default input scale for int8
    
    for conv_weight, bias_name in conv_bias_map.items():
        if bias_name in onnx_weights:
            bias_tensor = onnx_weights[bias_name]
            weight_scale = weight_scales.get(conv_weight, 1.0 / 127.0)
            codegen.add_bias(bias_name, bias_tensor, input_scale, weight_scale)
            print(f"  Added conv bias: {bias_name} (scale={input_scale * weight_scale:.6e})")
    
    # Third pass: add dense bias tensors
    for dense_weight, bias_name in dense_bias_map.items():
        if bias_name in onnx_weights:
            bias_tensor = onnx_weights[bias_name]
            weight_scale = weight_scales.get(dense_weight, 1.0 / 127.0)
            codegen.add_bias(bias_name, bias_tensor, input_scale, weight_scale)
            print(f"  Added dense bias: {bias_name} (scale={input_scale * weight_scale:.6e})")
    
    # Also add any TVM params not in ONNX
    for name, param in params.items():
        if name not in onnx_weights:
            codegen.add_weight(name, param.numpy())
    
    print(f"  Quantized weights: {codegen.current_offset:,} bytes ({codegen.current_offset/1024/1024:.2f} MB)")
    
    # Generate output files
    weights_bin = os.path.join(output_dir, f"{model_name}_weights.bin")
    weights_h = os.path.join(output_dir, f"{model_name}_weights.h")
    inference_c = os.path.join(output_dir, f"{model_name}_inference.c")
    layers_json = os.path.join(output_dir, f"{model_name}_layers.json")
    
    codegen.generate_weights_binary(weights_bin)
    codegen.generate_weights_header(weights_h)
    codegen.generate_inference_code(inference_c, layers, input_shape, onnx_weights, conv_bias_map, dense_bias_map)
    
    # Save layer info for debugging
    layer_data = [layer.to_dict() for layer in layers]
    with open(layers_json, 'w') as f:
        json.dump(layer_data, f, indent=2, default=str)
    print(f"  Layer info: {layers_json}")
    
    # Generate test data and test program
    print("\n[7/7] Generating test data and test program...")
    test_dir = os.path.join(output_dir, "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate test data (use ONNX model for reference inference)
    generate_test_data(onnx_model, test_dir, input_shape, input_scale)
    
    # Generate test program
    test_c = os.path.join(output_dir, f"test_{model_name}.c")
    generate_test_program(test_c, model_name, input_shape)
    
    # Generate Makefile
    makefile_path = os.path.join(output_dir, "Makefile")
    generate_makefile(makefile_path, model_name)
    
    print("\n" + "=" * 70)
    print("Compilation Complete!")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  - {model_name}_weights.bin  ({codegen.current_offset:,} bytes)")
    print(f"  - {model_name}_weights.h")
    print(f"  - {model_name}_inference.c")
    print(f"  - {model_name}_layers.json")
    print(f"  - test_{model_name}.c")
    print(f"  - Makefile")
    print(f"  - test_data/test_data.h (with embedded test input)")
    print(f"\nTo build and run on REMU:")
    print(f"  cd {output_dir} && make ARCH=riscv32-remu run")
    
    return {
        "weights_size": codegen.current_offset,
        "num_layers": len(layers),
        "op_counts": op_counts,
        "total_params": total_params,
    }


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
    input_shape = tuple(map(int, sys.argv[4].split(","))) if len(sys.argv) > 4 else (1, 3, 224, 224)
    
    result = compile_model(model_path, output_dir, model_name, input_shape)
    
    print("\nSummary:")
    print(f"  Parameters: {result['total_params']:,}")
    print(f"  Weights: {result['weights_size']:,} bytes")
    print(f"  Layers: {result['num_layers']}")
