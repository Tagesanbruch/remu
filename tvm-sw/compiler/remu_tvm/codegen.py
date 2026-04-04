import numpy as np
import os
from typing import List, Dict, OrderedDict, Tuple
from collections import OrderedDict
from .config import NPUConfig, NPU_CONFIG
from .quantization import QuantizedWeight, quantize_symmetric
from .analyzer import LayerInfo

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
        # For already-quantized integer tensors (e.g., qnn conv weights), keep raw data.
        if np.issubdtype(tensor.dtype, np.signedinteger) and tensor.dtype.itemsize == 1:
            q_tensor = tensor.astype(np.int8, copy=False)
            scale = 1.0
        elif np.issubdtype(tensor.dtype, np.floating):
            q_tensor, scale = quantize_symmetric(tensor.astype(np.float32))
        elif np.issubdtype(tensor.dtype, np.integer):
            # Keep non-int8 integers as-is; these are typically metadata tensors.
            q_tensor = tensor.astype(tensor.dtype, copy=False)
            scale = 1.0
        else:
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
                # Pad to alignment
                padding = weight.offset - f.tell()
                if padding > 0:
                    f.write(b'\x00' * padding)
                f.write(weight.data.tobytes())
                
        print(f"  Weights binary: {path} ({self.current_offset:,} bytes)")
        
    def generate_weights_c(self, path: str):
        """Generate weights C file for embedding."""
        lines = [
            f'#include "{self.model_name}_weights.h"',
            "",
            "// 16-byte alignment for DMA",
            "__attribute__((aligned(16)))",
            "const int8_t weights_data[] = {",
        ]
        
        all_data = bytearray()
        current_off = 0
        
        for name, weight in self.weights.items():
            # Pad to alignment relative to buffer start
            padding = weight.offset - current_off
            if padding > 0:
                all_data.extend(b'\x00' * padding)
                current_off += padding
            
            w_bytes = weight.data.tobytes()
            all_data.extend(w_bytes)
            current_off += len(w_bytes)
            
        # Convert to hex strings
        hex_data = []
        for b in all_data:
            hex_data.append(f"0x{b:02x}")
            
        # Chunk into lines to avoid huge lines
        chunk_size = 12 
        for i in range(0, len(hex_data), chunk_size):
            chunk = hex_data[i:i+chunk_size]
            lines.append("    " + ", ".join(chunk) + ",")
            
        lines.append("};")
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Weights C source: {path}")
        
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
            "// Weights embedded in C",
            "extern const int8_t weights_data[];",
            f"#define WEIGHTS_TOTAL_SIZE {self.current_offset}",
            "",
            "// Weight tensor definitions",
        ]
        
        for name, weight in self.weights.items():
            safe_name = self._safe_name(name)
            
            lines.append(f"")
            lines.append(f"// {name}: shape=[{', '.join(str(d) for d in weight.shape)}], scale={weight.scale:.6e}")
            lines.append(f"#define WEIGHT_{safe_name}_OFFSET {weight.offset}")
            lines.append(f"#define WEIGHT_{safe_name}_SIZE {weight.data.nbytes}")
            lines.append(f"#define WEIGHT_{safe_name}_SCALE {weight.scale:.6e}f")
            lines.append(f"#define WEIGHT_{safe_name}_SCALE_Q16 {int(weight.scale * 65536)}")
            # Point to embedded array
            lines.append(f"#define WEIGHT_{safe_name} (weights_data + {weight.offset})")
            
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
                                 dense_bias_map: Dict[str, str] = None,
                                 onnx_model=None,
                                 input_scale_ref: float = 1.0):
        """Generate complete C inference code with NPU API calls.
        
        Args:
            path: Output file path
            layers: List of layer information from TVM analysis
            input_shape: Input tensor shape (N, C, H, W)
            onnx_weights: Dictionary of ONNX weights
            conv_bias_map: Mapping from conv weight name to bias name
            dense_bias_map: Mapping from dense weight name to bias name
            onnx_model: Optional ONNX model proto for preserving conv node attrs
            input_scale_ref: Reference input scale used for non-qnn bias quantization
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
        
        # Capture original ONNX Conv/QLinearConv attrs by weight name so we can
        # preserve padding/stride semantics when Relay canonicalization differs.
        onnx_conv_attrs = {}
        if onnx_model is not None:
            try:
                for node in onnx_model.graph.node:
                    if node.op_type not in ("QLinearConv", "Conv"):
                        continue
                    if node.op_type == "QLinearConv":
                        if len(node.input) < 4:
                            continue
                        weight_name = node.input[3]
                    else:
                        if len(node.input) < 2:
                            continue
                        weight_name = node.input[1]
                    pads = [0, 0, 0, 0]
                    strides = [1, 1]
                    for attr in node.attribute:
                        if attr.name == "pads" and attr.ints:
                            pads = [int(x) for x in attr.ints]
                        elif attr.name == "strides" and attr.ints:
                            strides = [int(x) for x in attr.ints]
                    onnx_conv_attrs[weight_name] = {
                        "pads": pads,
                        "strides": strides,
                    }
            except Exception:
                onnx_conv_attrs = {}

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
            "#ifdef HOST_NATIVE",
            '#include "native_compat/am.h"',
            '#include "native_compat/klib.h"',
            '#include "native_compat/npu.h"',
            '#include "native_compat/npu_ops.h"',
            "#else",
            "#include <am.h>",
            "#include <klib.h>",
            '#include "npu.h"',
            '#include "npu_ops.h"',
            "#endif",
            f'#include "{self.model_name}_weights.h"',
            "",
            "#ifndef NPU_PROFILE_LAYERS",
            "#define NPU_PROFILE_LAYERS 0",
            "#endif",
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
            "static inline int32_t calc_auto_requant_divisor(const int32_t* input, int len) {",
            "    int32_t max_abs = 0;",
            "    for (int i = 0; i < len; i++) {",
            "        int32_t v = input[i];",
            "        if (v == INT32_MIN) v = INT32_MAX;",
            "        if (v < 0) v = -v;",
            "        if (v > max_abs) max_abs = v;",
            "    }",
            "    if (max_abs <= 127) return 1;",
            "    return (max_abs + 126) / 127;",
            "}",
            "",
            "// Q16 fixed-point rounding helper (round to nearest, ties to even)",
            "static inline int32_t round_shift_q16_even(int64_t v) {",
            "    int sign = (v < 0) ? -1 : 1;",
            "    uint64_t a = (v < 0) ? (uint64_t)(-v) : (uint64_t)v;",
            "    uint64_t q = a >> 16;",
            "    uint64_t rem = a & 0xFFFFu;",
            "    if (rem > 0x8000u || (rem == 0x8000u && (q & 1u))) {",
            "        q++;",
            "    }",
            "    return (int32_t)((sign > 0) ? (int64_t)q : -(int64_t)q);",
            "}",
            "",
            "static inline int32_t round_shift_q20_even(int64_t v) {",
            "    int sign = (v < 0) ? -1 : 1;",
            "    uint64_t a = (v < 0) ? (uint64_t)(-v) : (uint64_t)v;",
            "    uint64_t q = a >> 20;",
            "    uint64_t rem = a & 0xFFFFFu;",
            "    if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) {",
            "        q++;",
            "    }",
            "    return (int32_t)((sign > 0) ? (int64_t)q : -(int64_t)q);",
            "}",
            "",
            "static inline int32_t round_shift_q31_even(int64_t v) {",
            "    int sign = (v < 0) ? -1 : 1;",
            "    uint64_t a = (v < 0) ? (uint64_t)(-v) : (uint64_t)v;",
            "    uint64_t q = a >> 31;",
            "    uint64_t rem = a & 0x7FFFFFFFu;",
            "    if (rem > 0x40000000u || (rem == 0x40000000u && (q & 1u))) {",
            "        q++;",
            "    }",
            "    return (int32_t)((sign > 0) ? (int64_t)q : -(int64_t)q);",
            "}",
            "",
        ]
        
        # Generate layer shape table
        lines.append("// Layer output shapes (for debugging)")
        lines.append("typedef struct {")
        lines.append("    int n, c, h, w;")
        lines.append("} TensorShape;")
        lines.append("")
        lines.append(f"static const TensorShape layer_shapes[{len(layers)}] __attribute__((unused)) = {{")
        for layer in layers:
            # Flatten empty or weird shapes cautiously
            if not layer.output_shape:
                shape = [1, 1, 1, 1]
            elif len(layer.output_shape) == 1:
                shape = [1, layer.output_shape[0], 1, 1]
            elif len(layer.output_shape) == 2:
                shape = [1, layer.output_shape[1], 1, 1] # Assumes (N, C)
            else:
                shape = list(layer.output_shape) + [1] * (4 - len(layer.output_shape))
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
            f"    const int64_t input_scale_q20 = {int(round(float(input_scale_ref) * (1 << 20)))};",
            "    int64_t cur_scale_q20 = input_scale_q20;",
            "    int64_t cur_acc_scale_q20 = input_scale_q20;",
            "    int64_t residual_scale_q20 = input_scale_q20;",
            "    int32_t rq_div = 1;",
            "",
        ])
        
        # Track state for layer generation
        residual_stack = []  # For skip connections
        cur_shape = list(input_shape)
        last_conv_channels = 0  # For tracking bias add
        current_is_acc = False  # False: cur_input(int8) holds current tensor; True: cur_acc(int32)
        fused_bias_names = set()
        global_avg_sum_spatial = None  # Track sum-domain global avg path before qnn.quantize
        is_qnn_model = any(layer.op_type.startswith("qnn.") for layer in layers)

        requant_mode = os.environ.get("REQUANT_MODE", "auto").strip().lower()
        try:
            requant_shift = int(os.environ.get("REQUANT_SHIFT", "8"))
        except ValueError:
            requant_shift = 8

        qnn_add_explicit = os.environ.get("QNN_ADD_EXPLICIT", "1").strip().lower() in ("1", "true", "yes", "on")
        qnn_conv_zp_corr = os.environ.get("QNN_CONV_ZP_CORR", "1").strip().lower() in ("1", "true", "yes", "on")
        qnn_dense_zp_corr = os.environ.get("QNN_DENSE_ZP_CORR", "1").strip().lower() in ("1", "true", "yes", "on")

        qnn_conv_zp_corr_skip = set()
        qnn_conv_skip_env = os.environ.get("QNN_CONV_ZP_CORR_SKIP_LAYERS", "").strip()
        if qnn_conv_skip_env:
            for tok in qnn_conv_skip_env.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    qnn_conv_zp_corr_skip.add(int(tok))
                except ValueError:
                    pass

        debug_layers = set()
        debug_layers_env = os.environ.get("NPU_DEBUG_STATS_LAYERS", "").strip()
        if debug_layers_env:
            for tok in debug_layers_env.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    debug_layers.add(int(tok))
                except ValueError:
                    pass

        def requant_lines(length_expr: str, update_scale: bool = True) -> List[str]:
            if requant_mode == "shift":
                out = [f"npu_requantize_shift(cur_acc, cur_output, {length_expr}, {requant_shift});"]
                if update_scale:
                    shift_div = int(1 << max(requant_shift, 0))
                    out.append(f"cur_scale_q20 = cur_acc_scale_q20 * {shift_div};")
                    out.append("if (cur_scale_q20 < 1) cur_scale_q20 = 1;")
                return out
            out = [
                f"rq_div = calc_auto_requant_divisor(cur_acc, {length_expr});",
                f"npu_requantize_auto(cur_acc, cur_output, {length_expr});",
            ]
            if update_scale:
                out.append("cur_scale_q20 = cur_acc_scale_q20 * (int64_t)rq_div;")
                out.append("if (cur_scale_q20 < 1) cur_scale_q20 = 1;")
            return out

        def pad_tuple(padding_attr):
            """Normalize Relay padding attr to (top, left, bottom, right)."""
            if isinstance(padding_attr, (int, float)):
                p = int(padding_attr)
                return (p, p, p, p)
            if hasattr(padding_attr, "__iter__"):
                vals = [int(x) for x in padding_attr]
                if len(vals) == 0:
                    return (0, 0, 0, 0)
                if len(vals) == 1:
                    return (vals[0], vals[0], vals[0], vals[0])
                if len(vals) == 2:
                    return (vals[0], vals[1], vals[0], vals[1])
                return (vals[0], vals[1], vals[2], vals[3])
            return (0, 0, 0, 0)

        def scalar_pad(padding_attr):
            """Convert Relay padding attr to a scalar pad used by current NPU APIs."""
            pt, pl, pb, pr = pad_tuple(padding_attr)
            return max(pt, pl, pb, pr)

        def packed_conv_pad(padding_attr):
            """Pack asymmetric padding into a 32-bit value for conv/depthwise kernels."""
            pt, pl, pb, pr = pad_tuple(padding_attr)
            if pt == pl == pb == pr:
                return pt
            return ((pt & 0xFF) |
                    ((pl & 0xFF) << 8) |
                    ((pb & 0xFF) << 16) |
                    ((pr & 0xFF) << 24))

        def next_effective_op(start_idx: int):
            """Return next meaningful op after start_idx, skipping bias-only scaffolding nodes."""
            for k in range(start_idx + 1, len(layers)):
                opk = layers[k].op_type
                if "expand_dims" in opk:
                    continue
                if opk == "add" and layers[k].attrs.get("_is_bias_add", False):
                    continue
                return opk
            return None

        def resolve_scalar_attr(attrs: Dict[str, object], name_key: str, value_key: str):
            if value_key in attrs:
                try:
                    return float(attrs[value_key])
                except Exception:
                    pass

            tensor_name = attrs.get(name_key)
            if isinstance(tensor_name, str) and tensor_name in onnx_weights:
                try:
                    arr = np.asarray(onnx_weights[tensor_name])
                    if arr.size > 0:
                        return float(arr.reshape(-1)[0])
                except Exception:
                    pass
            return None

        def emit_static_i32_array(name: str, values: List[int], indent: str = "    ", chunk: int = 16):
            lines.append(f"{indent}static const int32_t {name}[{len(values)}] = {{")
            for j in range(0, len(values), chunk):
                part = values[j:j + chunk]
                lines.append(f"{indent}    " + ", ".join(str(v) for v in part) + ",")
            lines.append(f"{indent}}};")

        # Precompute where to capture residual tensors for later non-bias add.
        residual_source_for_add = {}
        residual_capture_layers = set()
        for add_idx, add_layer in enumerate(layers):
            if add_layer.op_type != "add":
                continue
            if add_layer.attrs.get("_is_bias_add", False):
                continue
            if add_idx > 0 and "nn.dense" in layers[add_idx - 1].op_type:
                continue

            input_layers = add_layer.attrs.get("_input_layers", [])
            if not isinstance(input_layers, (list, tuple)) or len(input_layers) < 2:
                continue

            prev_idx = add_idx - 1
            valid_inputs = []
            for x in input_layers:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if xi >= 0:
                    valid_inputs.append(xi)

            if not valid_inputs:
                continue

            # Main branch is usually the immediately previous op. Prefer the other input as skip source.
            skip_candidates = [xi for xi in valid_inputs if xi != prev_idx]
            skip_idx = skip_candidates[0] if skip_candidates else valid_inputs[0]

            residual_source_for_add[add_idx] = skip_idx
            residual_capture_layers.add(skip_idx)
        
        # Generate layer-by-layer execution
        for i, layer in enumerate(layers):
            op = layer.op_type
            attrs = layer.attrs
            out_shape = layer.output_shape if layer.output_shape else cur_shape
            
            lines.append(f"    // === Layer {i}: {op} ===")
            lines.extend([
                f"#if NPU_PROFILE_LAYERS",
                f"    uint32_t __lp_cyc_beg_{i} = npu_get_cycles();",
                f"    uint32_t __lp_mem_beg_{i} = npu_get_mem_bytes();",
                f"    uint32_t __lp_gemm_beg_{i} = npu_get_gemm_count();",
                f"    uint32_t __lp_act_beg_{i} = npu_get_act_count();",
                f"    uint32_t __lp_dma_beg_{i} = npu_get_dma_count();",
                f"#endif",
            ])
            
            if "nn.conv2d" in op or "qnn.conv2d" in op:
                # Extract conv parameters
                kh = attrs.get("kernel_size", [3, 3])
                kw = kh[1] if isinstance(kh, list) else kh
                kh = kh[0] if isinstance(kh, list) else kh
                
                strides = attrs.get("strides", [1, 1])
                stride = strides[0] if isinstance(strides, list) else strides
                
                padding = attrs.get("padding", [0, 0, 0, 0])
                weight_name = layer.weight_name
                if weight_name in onnx_conv_attrs:
                    onnx_pad = onnx_conv_attrs[weight_name].get("pads", [0, 0, 0, 0])
                    if len(onnx_pad) == 4:
                        padding = onnx_pad
                    onnx_stride = onnx_conv_attrs[weight_name].get("strides", [stride, stride])
                    if isinstance(onnx_stride, list) and len(onnx_stride) >= 1:
                        stride = int(onnx_stride[0])
                pad_vals = pad_tuple(padding)
                pad = packed_conv_pad(padding)
                pad_desc = str(list(pad_vals)) if len(set(pad_vals)) != 1 else str(pad_vals[0])
                
                groups = attrs.get("groups", 1)
                # Use TVM's output shape for channels
                channels = out_shape[1] if len(out_shape) > 1 else 32
                
                # Find weight tensor
                safe_weight = self._safe_name(weight_name) if weight_name else f"CONV{i}"
                
                # Check if this conv has a bias
                bias_name = conv_bias_map.get(weight_name) if weight_name else None
                safe_bias = self._safe_name(bias_name) if bias_name else None

                qnn_in_zp = None
                qnn_w_zp = None
                if "qnn.conv2d" in op:
                    qnn_in_zp = resolve_scalar_attr(attrs, '_qnn_in_zp_name', '_qnn_in_zp_val')
                    qnn_w_zp = resolve_scalar_attr(attrs, '_qnn_w_zp_name', '_qnn_w_zp_val')

                in_zp_i = int(round(qnn_in_zp)) if qnn_in_zp not in (None, 0.0) else 0
                conv_zp_corr_enabled = qnn_conv_zp_corr and (i not in qnn_conv_zp_corr_skip)
                pad_fill = in_zp_i if ("qnn.conv2d" in op and conv_zp_corr_enabled and in_zp_i != 0) else 0
                lines.append(f"    npu_set_input_pad_value({pad_fill});")
                
                if groups > 1 and groups == cur_shape[1]:
                    # Depthwise convolution
                    lines.extend([
                        f"    // Depthwise Conv2D: {cur_shape[1]} channels, kernel {kh}x{kw}, stride {stride}, pad {pad_desc}",
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
                        f"    // Conv2D: {cur_shape[1]} -> {channels} channels, kernel {kh}x{kw}, stride {stride}, pad {pad_desc}",
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

                if not is_qnn_model and "qnn.conv2d" not in op:
                    conv_weight_scale = 1.0
                    if weight_name in self.weights:
                        conv_weight_scale = float(self.weights[weight_name].scale)
                    conv_weight_scale_q20 = int(round(conv_weight_scale * (1 << 20)))
                    if conv_weight_scale_q20 < 1:
                        conv_weight_scale_q20 = 1
                    lines.extend([
                        f"    cur_acc_scale_q20 = (cur_scale_q20 * {conv_weight_scale_q20}LL) >> 20;",
                        f"    if (cur_acc_scale_q20 < 1) cur_acc_scale_q20 = 1;",
                    ])

                if qnn_w_zp not in (None, 0.0):
                    lines.extend([
                        f"    // WARNING: kernel zero-point {int(round(qnn_w_zp))} is not handled in current lowering",
                    ])

                if (
                    conv_zp_corr_enabled
                    and
                    qnn_in_zp not in (None, 0.0)
                    and weight_name
                    and weight_name in onnx_weights
                ):
                    try:
                        w_arr = np.asarray(onnx_weights[weight_name], dtype=np.int32)
                        if w_arr.ndim >= 3:
                            wsum = w_arr.reshape(w_arr.shape[0], -1).sum(axis=1)
                            wsum_vals = [int(x) for x in wsum.tolist()]
                        else:
                            wsum_vals = []
                    except Exception:
                        wsum_vals = []

                    if wsum_vals:
                        in_zp_i = int(round(qnn_in_zp))
                        wsum_name = f"qnn_wsum_{i}"
                        lines.append("    // qnn.conv2d input zero-point correction")
                        lines.append("    {")
                        emit_static_i32_array(wsum_name, wsum_vals, indent="        ")
                        lines.extend([
                            f"        int spatial_size = cur_h * cur_w;",
                            f"        for (int c = 0; c < cur_c; c++) {{",
                            f"            int32_t corr = {in_zp_i} * {wsum_name}[c];",
                            f"            int32_t* channel_ptr = cur_acc + c * spatial_size;",
                            f"            for (int s = 0; s < spatial_size; s++) {{",
                            f"                channel_ptr[s] -= corr;",
                            f"            }}",
                            f"        }}",
                            f"    }}",
                        ])

                lines.append("    npu_set_input_pad_value(0);")
                
                # Add bias if present (fused into conv)
                if bias_name and safe_bias:
                    if not is_qnn_model and "qnn.conv2d" not in op:
                        lines.extend([
                            f"    // Add bias (fused): {bias_name}",
                            f"    {{",
                            f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                            f"        int spatial_size = cur_h * cur_w;",
                            f"        int64_t bias_ratio_q20 = (cur_scale_q20 > 0) ? ((input_scale_q20 << 20) / cur_scale_q20) : (1LL << 20);",
                            f"        for (int c = 0; c < cur_c; c++) {{",
                            f"            int32_t bias_val = round_shift_q20_even((int64_t)bias[c] * bias_ratio_q20);",
                            f"            int32_t* channel_ptr = cur_acc + c * spatial_size;",
                            f"            for (int s = 0; s < spatial_size; s++) {{",
                            f"                channel_ptr[s] += bias_val;",
                            f"            }}",
                            f"        }}",
                            f"    }}",
                        ])
                    else:
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
                    fused_bias_names.add(bias_name)
                
                lines.append("")
                current_is_acc = True
                
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
                ])
                lines.extend([f"    {x}" for x in requant_lines(str(size), update_scale=not is_qnn_model)])
                lines.extend([
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                    "",
                ])
                current_is_acc = False
                
            elif "clip" in op:
                # ReLU6 or generic clip
                a_min = attrs.get("a_min", 0)
                a_max = attrs.get("a_max", 6)
                size = cur_shape[1] * cur_shape[2] * cur_shape[3]

                if a_min == 0 and a_max == 6:
                    if not is_qnn_model:
                        lines.extend([
                            f"    // ReLU6 in Relay graph (non-qnn): clip in accumulator domain using tracked scale",
                            f"    {{",
                            f"        int32_t relu6_clip = (cur_acc_scale_q20 > 0) ? (int32_t)(((6LL << 20) + (cur_acc_scale_q20 / 2)) / cur_acc_scale_q20) : 1;",
                            f"        if (relu6_clip < 1) relu6_clip = 1;",
                            f"        npu_clip_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1, 0, relu6_clip);",
                            f"    }}",
                        ])
                    else:
                        lines.extend([
                            f"    // ReLU6 in Relay graph",
                            f"    npu_relu_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1);",
                        ])
                else:
                    if not is_qnn_model:
                        clip_min_q20 = int(round(float(a_min) * (1 << 20)))
                        clip_max_q20 = int(round(float(a_max) * (1 << 20)))
                        lines.extend([
                            f"    // Clip [{a_min}, {a_max}] with tracked accumulator scale",
                            f"    {{",
                            f"        int64_t clip_min_q20 = {clip_min_q20};",
                            f"        int64_t clip_max_q20 = {clip_max_q20};",
                            f"        int32_t clip_min = (cur_acc_scale_q20 > 0) ? (int32_t)((clip_min_q20 >= 0) ? ((clip_min_q20 + (cur_acc_scale_q20 / 2)) / cur_acc_scale_q20) : -(((-clip_min_q20) + (cur_acc_scale_q20 / 2)) / cur_acc_scale_q20)) : {int(a_min)};",
                            f"        int32_t clip_max = (cur_acc_scale_q20 > 0) ? (int32_t)((clip_max_q20 >= 0) ? ((clip_max_q20 + (cur_acc_scale_q20 / 2)) / cur_acc_scale_q20) : -(((-clip_max_q20) + (cur_acc_scale_q20 / 2)) / cur_acc_scale_q20)) : {int(a_max)};",
                            f"        npu_clip_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1, clip_min, clip_max);",
                            f"    }}",
                        ])
                    else:
                        lines.extend([
                            f"    // Clip [{a_min}, {a_max}]",
                            f"    npu_clip_elementwise(cur_acc, cur_acc, cur_c * cur_h * cur_w, 1, {int(a_min)}, {int(a_max)});",
                        ])

                # After activation, requantize back to int8 for next layer
                lines.extend([
                    f"    // Requantize int32 -> int8 for next layer",
                ])
                lines.extend([f"    {x}" for x in requant_lines('cur_c * cur_h * cur_w', update_scale=not is_qnn_model)])
                lines.extend([
                    f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                    "",
                ])
                current_is_acc = False
                
            elif "add" in op and "bias" not in op:
                # Check if this is bias add (following expand_dims) - skip if we already fused it
                is_bias_add = attrs.get("_is_bias_add", False)
                add_result_is_acc = current_is_acc
                if not is_bias_add and i > 0 and "expand_dims" in layers[i - 1].op_type:
                    # Relay often lowers bias add as `expand_dims(const_bias)` + add.
                    # That pattern is already fused when generating conv/dense blocks.
                    is_bias_add = True
                
                # Check if this add follows a dense layer (dense bias add pattern from TVM Gemm decomposition)
                is_dense_bias_add = (
                    i > 0 and
                    (
                        "nn.dense" in layers[i - 1].op_type or
                        "qnn.dense" in layers[i - 1].op_type
                    )
                )
                
                if is_bias_add:
                    bias_var_name = None
                    input_layers = attrs.get("_input_layers", [])
                    bias_arg_idx = attrs.get("_bias_arg_idx", None)
                    if isinstance(input_layers, (list, tuple)) and isinstance(bias_arg_idx, int):
                        if 0 <= bias_arg_idx < len(input_layers):
                            try:
                                bias_src_idx = int(input_layers[bias_arg_idx])
                            except Exception:
                                bias_src_idx = -1
                            if 0 <= bias_src_idx < len(layers):
                                bias_src_layer = layers[bias_src_idx]
                                if "expand_dims" in bias_src_layer.op_type:
                                    bias_var_name = bias_src_layer.attrs.get("_bias_var_name")

                    can_apply_bias = (
                        isinstance(bias_var_name, str)
                        and bias_var_name in self.weights
                        and bias_var_name not in fused_bias_names
                    )

                    if can_apply_bias:
                        safe_bias = self._safe_name(bias_var_name)
                        if not is_qnn_model:
                            lines.extend([
                                f"    // Bias add from expand_dims source: {bias_var_name}",
                                f"    {{",
                                f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                                f"        int spatial_size = cur_h * cur_w;",
                                f"        int64_t bias_ratio_q20 = (cur_scale_q20 > 0) ? ((input_scale_q20 << 20) / cur_scale_q20) : (1LL << 20);",
                                f"        for (int c = 0; c < cur_c; c++) {{",
                                f"            int32_t bias_val = round_shift_q20_even((int64_t)bias[c] * bias_ratio_q20);",
                                f"            int32_t* channel_ptr = cur_acc + c * spatial_size;",
                                f"            for (int s = 0; s < spatial_size; s++) {{",
                                f"                channel_ptr[s] += bias_val;",
                                f"            }}",
                                f"        }}",
                                f"    }}",
                            ])
                        else:
                            lines.extend([
                                f"    // Bias add from expand_dims source: {bias_var_name}",
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
                        add_result_is_acc = True
                    else:
                        lines.extend([
                            f"    // Bias add (skipped - already fused or unavailable)",
                        ])

                    # If next op is not an activation, we still need int32->int8 handoff.
                    next_op = next_effective_op(i)
                    need_requant = (
                        next_op is not None
                        and "clip" not in next_op
                        and "nn.relu" not in next_op
                        and "add" not in next_op
                        and "qnn.requantize" not in next_op
                        and "qnn.quantize" not in next_op
                    )
                    if need_requant:
                        lines.extend([
                            f"    // Requantize after fused bias add for next op: {next_op}",
                        ])
                        lines.extend([f"    {x}" for x in requant_lines('cur_c * cur_h * cur_w', update_scale=not is_qnn_model)])
                        lines.extend([
                            f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                        ])
                        add_result_is_acc = False
                    lines.append("")
                elif is_dense_bias_add:
                    # Dense bias add - already fused into dense layer - skip
                    lines.extend([
                        f"    // Dense bias add (skipped - already fused into nn.dense)",
                        "",
                    ])
                else:
                    quantized_residual_add_done = False

                    # Handle qnn.dequantize + add + qnn.quantize with explicit scale/zp math.
                    q_layer = layers[i + 1] if (i + 1 < len(layers) and "qnn.quantize" in layers[i + 1].op_type) else None
                    if qnn_add_explicit and q_layer is not None and i in residual_source_for_add:
                        raw_inputs = attrs.get("_input_layers", [])
                        valid_inputs = []
                        if isinstance(raw_inputs, (list, tuple)):
                            for x in raw_inputs:
                                try:
                                    xi = int(x)
                                except Exception:
                                    continue
                                if xi >= 0:
                                    valid_inputs.append(xi)

                        if len(valid_inputs) >= 2:
                            prev_idx = i - 1
                            main_idx = prev_idx if prev_idx in valid_inputs else valid_inputs[0]
                            skip_idx = residual_source_for_add.get(i)
                            if skip_idx not in valid_inputs:
                                skip_candidates = [xi for xi in valid_inputs if xi != main_idx]
                                skip_idx = skip_candidates[0] if skip_candidates else None

                            if (
                                skip_idx is not None
                                and 0 <= main_idx < len(layers)
                                and 0 <= skip_idx < len(layers)
                            ):
                                main_layer = layers[main_idx]
                                skip_layer = layers[skip_idx]

                                if "qnn.dequantize" in main_layer.op_type and "qnn.dequantize" in skip_layer.op_type:
                                    main_scale = resolve_scalar_attr(main_layer.attrs, '_dq_in_scale_name', '_dq_in_scale_val')
                                    main_zp = resolve_scalar_attr(main_layer.attrs, '_dq_in_zp_name', '_dq_in_zp_val')
                                    skip_scale = resolve_scalar_attr(skip_layer.attrs, '_dq_in_scale_name', '_dq_in_scale_val')
                                    skip_zp = resolve_scalar_attr(skip_layer.attrs, '_dq_in_zp_name', '_dq_in_zp_val')
                                    out_scale = resolve_scalar_attr(q_layer.attrs, '_q_out_scale_name', '_q_out_scale_val')
                                    out_zp = resolve_scalar_attr(q_layer.attrs, '_q_out_zp_name', '_q_out_zp_val')

                                    if (
                                        main_scale is not None and main_zp is not None
                                        and skip_scale is not None and skip_zp is not None
                                        and out_scale not in (None, 0.0) and out_zp is not None
                                    ):
                                        main_zp_i = int(round(main_zp))
                                        skip_zp_i = int(round(skip_zp))
                                        out_zp_i = int(round(out_zp))
                                        main_mul_q20 = int(round((main_scale / out_scale) * (1 << 20)))
                                        skip_mul_q20 = int(round((skip_scale / out_scale) * (1 << 20)))
                                        lines.extend([
                                            f"    // Quantized residual add via dequantize/quantize params",
                                            f"    for (int ri = 0; ri < cur_c * cur_h * cur_w; ri++) {{",
                                            f"        int32_t main_delta = (int32_t)cur_input[ri] - {main_zp_i};",
                                            f"        int32_t skip_delta = (int32_t)residual_buf[ri] - {skip_zp_i};",
                                            f"        int64_t sum_q20 = (int64_t)main_delta * {main_mul_q20} + (int64_t)skip_delta * {skip_mul_q20};",
                                            f"        int qi = round_shift_q20_even(sum_q20) + {out_zp_i};",
                                            f"        if (qi > 127) qi = 127;",
                                            f"        if (qi < -128) qi = -128;",
                                            f"        cur_output[ri] = (int8_t)qi;",
                                            f"    }}",
                                            f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                                        ])
                                        quantized_residual_add_done = True
                                        add_result_is_acc = False

                    if not quantized_residual_add_done:
                        add_result_is_acc = True
                        # Element-wise add (residual connection)
                        if i in residual_source_for_add:
                            if not is_qnn_model:
                                lines.extend([
                                    f"    // Element-wise add (residual connection) with scale alignment",
                                    f"    for (int ri = 0; ri < cur_c * cur_h * cur_w; ri++) {{",
                                    f"        cur_acc[ri] += round_shift_q20_even((int64_t)residual_buf[ri] * ((cur_acc_scale_q20 > 0) ? ((residual_scale_q20 << 20) / cur_acc_scale_q20) : (1LL << 20)));",
                                    f"    }}",
                                ])
                            else:
                                lines.extend([
                                    f"    // Element-wise add (residual connection)",
                                    f"    for (int ri = 0; ri < cur_c * cur_h * cur_w; ri++) {{",
                                    f"        cur_acc[ri] += (int32_t)residual_buf[ri];",
                                    f"    }}",
                                ])
                        else:
                            lines.extend([
                                f"    // Residual add source unresolved; keeping main branch only",
                            ])

                        next_op = next_effective_op(i)
                        need_requant = (
                            next_op is not None
                            and "clip" not in next_op
                            and "nn.relu" not in next_op
                            and "add" not in next_op
                            and "qnn.requantize" not in next_op
                            and "qnn.quantize" not in next_op
                        )
                        if need_requant:
                            lines.extend([
                                f"    // Requantize after residual add for next op: {next_op}",
                            ])
                            lines.extend([f"    {x}" for x in requant_lines('cur_c * cur_h * cur_w', update_scale=not is_qnn_model)])
                            lines.extend([
                                f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                            ])
                            add_result_is_acc = False
                    lines.append("")

                current_is_acc = add_result_is_acc
                # Update shape from TVM
                if len(out_shape) >= 4:
                    cur_shape = list(out_shape)
                
            elif "qnn.requantize" in op:
                in_scale = resolve_scalar_attr(attrs, '_rq_in_scale_name', '_rq_in_scale_val')
                out_scale = resolve_scalar_attr(attrs, '_rq_out_scale_name', '_rq_out_scale_val')
                out_zp = resolve_scalar_attr(attrs, '_rq_out_zp_name', '_rq_out_zp_val')

                if in_scale is not None and out_scale not in (None, 0.0) and out_zp is not None:
                    multiplier = in_scale / out_scale
                    scale_q31 = int(round(multiplier * (1 << 31)))
                    scale_q31 = max(min(scale_q31, 2147483647), -2147483648)
                    zero_point = int(round(out_zp))
                    lines.extend([
                        f"    // qnn.requantize (scale={in_scale:.6e}/{out_scale:.6e}, zp={zero_point})",
                        f"    npu_requantize_q31(cur_acc, cur_output, cur_c * cur_h * cur_w, {scale_q31}, {zero_point});",
                        f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                        "",
                    ])
                else:
                    lines.extend([
                        f"    // qnn.requantize (fallback path)",
                        f"    npu_requantize_auto(cur_acc, cur_output, cur_c * cur_h * cur_w);",
                        f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                        "",
                    ])
                current_is_acc = False

            elif "qnn.quantize" in op:
                q_out_scale = resolve_scalar_attr(attrs, '_q_out_scale_name', '_q_out_scale_val')
                q_out_zp = resolve_scalar_attr(attrs, '_q_out_zp_name', '_q_out_zp_val')
                quantized = False

                # Pattern: qnn.dequantize -> nn.global_avg_pool2d -> qnn.quantize
                # cur_acc currently holds mean(x_q) from global avg pool in quant domain.
                # Restore dequantize zero-point/scale semantics before requantizing.
                if (
                    current_is_acc
                    and q_out_scale not in (None, 0.0)
                    and q_out_zp is not None
                    and i > 1
                    and "nn.global_avg_pool2d" in layers[i - 1].op_type
                    and "qnn.dequantize" in layers[i - 2].op_type
                ):
                    dq_scale = resolve_scalar_attr(layers[i - 2].attrs, '_dq_in_scale_name', '_dq_in_scale_val')
                    dq_zp = resolve_scalar_attr(layers[i - 2].attrs, '_dq_in_zp_name', '_dq_in_zp_val')
                    if dq_scale not in (None, 0.0) and dq_zp is not None:
                        in_zp_i = int(round(dq_zp))
                        out_zp_i = int(round(q_out_zp))
                        if global_avg_sum_spatial is not None and global_avg_sum_spatial > 0:
                            # Use sum-domain quantization to avoid double rounding:
                            # round(((sum(x_q) - zp*N) * s_in) / (N * s_out)) + zp_out
                            mul_q31 = int(round((dq_scale / (q_out_scale * global_avg_sum_spatial)) * (1 << 31)))
                            mul_q31 = max(min(mul_q31, 2147483647), -2147483648)
                            lines.extend([
                                f"    // qnn.quantize after dequantized global_avg_pool2d (sum-domain)",
                                f"    for (int qi = 0; qi < cur_c * cur_h * cur_w; qi++) {{",
                                f"        int32_t delta = cur_acc[qi] - ({in_zp_i} * {global_avg_sum_spatial});",
                                f"        int64_t tmp = (int64_t)delta * {mul_q31};",
                                f"        int qv = round_shift_q31_even(tmp) + {out_zp_i};",
                                f"        if (qv > 127) qv = 127;",
                                f"        if (qv < -128) qv = -128;",
                                f"        cur_output[qi] = (int8_t)qv;",
                                f"    }}",
                                f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                                "",
                            ])
                        else:
                            mul_q31 = int(round((dq_scale / q_out_scale) * (1 << 31)))
                            mul_q31 = max(min(mul_q31, 2147483647), -2147483648)
                            lines.extend([
                                f"    // qnn.quantize after dequantized global_avg_pool2d",
                                f"    for (int qi = 0; qi < cur_c * cur_h * cur_w; qi++) {{",
                                f"        int32_t delta = cur_acc[qi] - {in_zp_i};",
                                f"        int64_t tmp = (int64_t)delta * {mul_q31};",
                                f"        int qv = round_shift_q31_even(tmp) + {out_zp_i};",
                                f"        if (qv > 127) qv = 127;",
                                f"        if (qv < -128) qv = -128;",
                                f"        cur_output[qi] = (int8_t)qv;",
                                f"    }}",
                                f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                                "",
                            ])
                        current_is_acc = False
                        quantized = True
                        global_avg_sum_spatial = None

                if (not quantized) and current_is_acc and q_out_scale not in (None, 0.0) and q_out_zp is not None and i > 0 and "qnn.dequantize" in layers[i - 1].op_type:
                    prev_dq_scale = resolve_scalar_attr(layers[i - 1].attrs, '_dq_in_scale_name', '_dq_in_scale_val')
                    if prev_dq_scale not in (None, 0.0):
                        multiplier = prev_dq_scale / q_out_scale
                        scale_q31 = int(round(multiplier * (1 << 31)))
                        scale_q31 = max(min(scale_q31, 2147483647), -2147483648)
                        zero_point = int(round(q_out_zp))
                        lines.extend([
                            f"    // qnn.quantize from int32 domain via preceding dequantize scale",
                            f"    npu_requantize_q31(cur_acc, cur_output, cur_c * cur_h * cur_w, {scale_q31}, {zero_point});",
                            f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                            "",
                        ])
                        current_is_acc = False
                        quantized = True
                        global_avg_sum_spatial = None

                if not quantized:
                    lines.extend([
                        f"    // qnn.quantize (treated as no-op in current execution flow)",
                        "",
                    ])
                    global_avg_sum_spatial = None

            elif "qnn.dequantize" in op or op == "cast":
                lines.extend([
                    f"    // {op} (treated as no-op in current execution flow)",
                    "",
                ])

            elif "nn.global_avg_pool2d" in op:
                next_op = next_effective_op(i)
                prev_h = cur_shape[2]
                prev_w = cur_shape[3]

                if next_op is not None and "qnn.quantize" in next_op:
                    spatial = prev_h * prev_w
                    lines.extend([
                        f"    // Global Average Pooling: [{cur_shape[1]}, {prev_h}, {prev_w}] -> [{cur_shape[1]}, 1, 1]",
                        f"    // Keep sum-domain accumulator here to avoid double-rounding before qnn.quantize",
                        f"    {{",
                        f"        int spatial = {spatial};",
                        f"        for (int c = 0; c < cur_c; c++) {{",
                        f"            int32_t sum = 0;",
                        f"            int8_t* ptr = cur_input + c * spatial;",
                        f"            for (int s = 0; s < spatial; s++) {{",
                        f"                sum += (int32_t)ptr[s];",
                        f"            }}",
                        f"            cur_acc[c] = sum;",
                        f"        }}",
                        f"    }}",
                        f"    cur_h = 1;",
                        f"    cur_w = 1;",
                        "",
                    ])
                    current_is_acc = True
                    global_avg_sum_spatial = spatial
                else:
                    lines.extend([
                        f"    // Global Average Pooling: [{cur_shape[1]}, {prev_h}, {prev_w}] -> [{cur_shape[1]}, 1, 1]",
                        f"    npu_global_avgpool2d(cur_input, cur_acc, cur_n, cur_c, cur_h, cur_w);",
                        f"    cur_h = 1;",
                        f"    cur_w = 1;",
                    ])
                    if not is_qnn_model:
                        lines.append("    cur_acc_scale_q20 = cur_scale_q20;")
                    global_avg_sum_spatial = None

                if next_op is not None and "qnn.quantize" in next_op:
                    lines.extend([
                        f"    // Next op is qnn.quantize; keep int32 accumulator for scale/zp-aware quantization",
                        "",
                    ])
                else:
                    lines.extend([
                        f"    // Requantize for dense layer input",
                    ])
                    lines.extend([f"    {x}" for x in requant_lines('cur_c', update_scale=not is_qnn_model)])
                    lines.extend([
                        f"    {{ int8_t* tmp = cur_input; cur_input = cur_output; cur_output = tmp; }}",
                        "",
                    ])
                    current_is_acc = False
                cur_shape[2] = 1
                cur_shape[3] = 1
                
            elif "nn.max_pool2d" in op:
                pool_size = attrs.get("pool_size", [2, 2])
                kh = pool_size[0] if isinstance(pool_size, list) else pool_size
                kw = pool_size[1] if isinstance(pool_size, list) else pool_size
                strides = attrs.get("strides", [2, 2])
                stride = strides[0] if isinstance(strides, list) else strides
                padding = attrs.get("padding", [0, 0, 0, 0])
                pad = scalar_pad(padding)
                
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
                current_is_acc = False
                
            elif "nn.avg_pool2d" in op:
                pool_size = attrs.get("pool_size", [2, 2])
                kh = pool_size[0] if isinstance(pool_size, list) else pool_size
                kw = pool_size[1] if isinstance(pool_size, list) else pool_size
                strides = attrs.get("strides", [2, 2])
                stride = strides[0] if isinstance(strides, list) else strides
                padding = attrs.get("padding", [0, 0, 0, 0])
                pad = scalar_pad(padding)
                
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
                current_is_acc = False
                
            elif "nn.dense" in op or "qnn.dense" in op:
                units = attrs.get("units", layer.output_shape[-1] if layer.output_shape else 1000)
                weight_name = layer.weight_name
                safe_weight = self._safe_name(weight_name) if weight_name else f"DENSE{i}"

                qnn_in_zp = None
                qnn_w_zp = None
                if "qnn.dense" in op:
                    qnn_in_zp = resolve_scalar_attr(attrs, '_qnn_in_zp_name', '_qnn_in_zp_val')
                    qnn_w_zp = resolve_scalar_attr(attrs, '_qnn_w_zp_name', '_qnn_w_zp_val')
                
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

                if not is_qnn_model and "qnn.dense" not in op:
                    dense_weight_scale = 1.0
                    if weight_name in self.weights:
                        dense_weight_scale = float(self.weights[weight_name].scale)
                    dense_weight_scale_q20 = int(round(dense_weight_scale * (1 << 20)))
                    if dense_weight_scale_q20 < 1:
                        dense_weight_scale_q20 = 1
                    lines.extend([
                        f"    cur_acc_scale_q20 = (cur_scale_q20 * {dense_weight_scale_q20}LL) >> 20;",
                        f"    if (cur_acc_scale_q20 < 1) cur_acc_scale_q20 = 1;",
                    ])

                if qnn_w_zp not in (None, 0.0):
                    lines.extend([
                        f"    // WARNING: qnn.dense weight zero-point {int(round(qnn_w_zp))} is not handled",
                    ])

                if (
                    qnn_dense_zp_corr
                    and
                    qnn_in_zp not in (None, 0.0)
                    and weight_name
                    and weight_name in onnx_weights
                ):
                    try:
                        w_arr = np.asarray(onnx_weights[weight_name], dtype=np.int32)
                        if w_arr.ndim >= 2:
                            row_sum = w_arr.reshape(w_arr.shape[0], -1).sum(axis=1)
                            row_sum_vals = [int(x) for x in row_sum.tolist()]
                        else:
                            row_sum_vals = []
                    except Exception:
                        row_sum_vals = []

                    if row_sum_vals:
                        in_zp_i = int(round(qnn_in_zp))
                        dense_wsum_name = f"qnn_dense_wsum_{i}"
                        lines.append("    // qnn.dense input zero-point correction")
                        lines.append("    {")
                        emit_static_i32_array(dense_wsum_name, row_sum_vals, indent="        ")
                        lines.extend([
                            f"        for (int i = 0; i < {units}; i++) {{",
                            f"            cur_acc[i] -= {in_zp_i} * {dense_wsum_name}[i];",
                            f"        }}",
                            f"    }}",
                        ])
                
                # Add bias if present
                if bias_name and safe_bias:
                    if not is_qnn_model and "qnn.dense" not in op:
                        lines.extend([
                            f"    // Add bias (fused): {bias_name}",
                            f"    {{",
                            f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                            f"        int64_t bias_ratio_q20 = (cur_scale_q20 > 0) ? ((input_scale_q20 << 20) / cur_scale_q20) : (1LL << 20);",
                            f"        for (int i = 0; i < {units}; i++) {{",
                            f"            cur_acc[i] += round_shift_q20_even((int64_t)bias[i] * bias_ratio_q20);",
                            f"        }}",
                            f"    }}",
                        ])
                    else:
                        lines.extend([
                            f"    // Add bias (fused): {bias_name}",
                            f"    {{",
                            f"        const int32_t* bias = (const int32_t*)WEIGHT_{safe_bias};",
                            f"        for (int i = 0; i < {units}; i++) {{",
                            f"            cur_acc[i] += bias[i];",
                            f"        }}",
                            f"    }}",
                        ])
                    fused_bias_names.add(bias_name)
                
                lines.append("")
                cur_shape = [cur_shape[0], units, 1, 1]
                current_is_acc = True
                
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
                current_is_acc = False
                
            else:
                # Unknown op - add comment
                lines.append(f"    // TODO: {op} - not implemented")
                lines.append("")

            if i in residual_capture_layers:
                lines.extend([
                    f"    // Save residual source for later element-wise add",
                    f"    memcpy(residual_buf, cur_input, cur_c * cur_h * cur_w);",
                ])
                if not is_qnn_model:
                    lines.append("    residual_scale_q20 = cur_scale_q20;")
                lines.append("")

            if i in debug_layers:
                if current_is_acc:
                    lines.extend([
                        f"    // Debug stats for layer {i} (int32 domain)",
                        f"    {{",
                        f"        int dbg_len = cur_c * cur_h * cur_w;",
                        f"        int32_t dbg_sum = 0;",
                        f"        int32_t dbg_min = (dbg_len > 0) ? cur_acc[0] : 0;",
                        f"        int32_t dbg_max = (dbg_len > 0) ? cur_acc[0] : 0;",
                        f"        for (int di = 0; di < dbg_len; di++) {{",
                        f"            int32_t v = cur_acc[di];",
                        f"            dbg_sum += v;",
                        f"            if (v < dbg_min) dbg_min = v;",
                        f"            if (v > dbg_max) dbg_max = v;",
                        f"        }}",
                        f"        int32_t v0 = (dbg_len > 0) ? cur_acc[0] : 0;",
                        f"        int32_t v1 = (dbg_len > 1) ? cur_acc[1] : 0;",
                        f"        int32_t v2 = (dbg_len > 2) ? cur_acc[2] : 0;",
                        f"        int32_t v3 = (dbg_len > 3) ? cur_acc[3] : 0;",
                        f"        printf(\"DBG_L{i}_I32 sum=%d min=%d max=%d v0=%d v1=%d v2=%d v3=%d\\n\",",
                        f"               dbg_sum, dbg_min, dbg_max, v0, v1, v2, v3);",
                        f"    }}",
                        "",
                    ])
                else:
                    lines.extend([
                        f"    // Debug stats for layer {i} (int8 domain)",
                        f"    {{",
                        f"        int dbg_len = cur_c * cur_h * cur_w;",
                        f"        int32_t dbg_sum = 0;",
                        f"        int dbg_min = (dbg_len > 0) ? (int)cur_input[0] : 0;",
                        f"        int dbg_max = (dbg_len > 0) ? (int)cur_input[0] : 0;",
                        f"        for (int di = 0; di < dbg_len; di++) {{",
                        f"            int v = (int)cur_input[di];",
                        f"            dbg_sum += v;",
                        f"            if (v < dbg_min) dbg_min = v;",
                        f"            if (v > dbg_max) dbg_max = v;",
                        f"        }}",
                        f"        int v0 = (dbg_len > 0) ? (int)cur_input[0] : 0;",
                        f"        int v1 = (dbg_len > 1) ? (int)cur_input[1] : 0;",
                        f"        int v2 = (dbg_len > 2) ? (int)cur_input[2] : 0;",
                        f"        int v3 = (dbg_len > 3) ? (int)cur_input[3] : 0;",
                        f"        printf(\"DBG_L{i}_I8 sum=%d min=%d max=%d v0=%d v1=%d v2=%d v3=%d\\n\",",
                        f"               dbg_sum, dbg_min, dbg_max, v0, v1, v2, v3);",
                        f"    }}",
                        "",
                    ])

            lines.extend([
                f"#if NPU_PROFILE_LAYERS",
                f"    printf(\"LPROF,{i},{op},%u,%u,%u,%u,%u\\n\",",
                f"           (unsigned)(npu_get_cycles() - __lp_cyc_beg_{i}),",
                f"           (unsigned)(npu_get_mem_bytes() - __lp_mem_beg_{i}),",
                f"           (unsigned)(npu_get_gemm_count() - __lp_gemm_beg_{i}),",
                f"           (unsigned)(npu_get_act_count() - __lp_act_beg_{i}),",
                f"           (unsigned)(npu_get_dma_count() - __lp_dma_beg_{i}));",
                f"#endif",
                "",
            ])
        
        # Copy final output
        final_size = cur_shape[1] * cur_shape[2] * cur_shape[3]
        if current_is_acc:
            lines.extend([
                f"    // Copy final output ({final_size} elements) from int32 accumulator",
                f"    memcpy(output, cur_acc, {final_size} * sizeof(int32_t));",
            ])
        else:
            lines.extend([
                f"    // Copy final output ({final_size} elements) from int8 tensor",
                f"    for (int i = 0; i < {final_size}; i++) {{",
                f"        output[i] = (int32_t)cur_input[i];",
                f"    }}",
            ])

        lines.extend([
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
            f'    printf("Weights: %d bytes @ %p\\n", WEIGHTS_TOTAL_SIZE, weights_data);',
            f'    printf("Layers: {len(layers)}\\n");',
            "    return 0;",
            "}",
            "",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Inference code: {path}")
