import numpy as np
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
            f'    printf("Weights: %d bytes @ %p\\n", WEIGHTS_TOTAL_SIZE, weights_data);',
            f'    printf("Layers: {len(layers)}\\n");',
            "    return 0;",
            "}",
            "",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Inference code: {path}")
