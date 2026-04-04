from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tvm.relay.expr import Constant, Var
from tvm.relay.expr_functor import ExprVisitor

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
        self.expr_to_layer_idx: Dict[int, int] = {}
        self.var_to_layer_idx: Dict[str, int] = {}

    def _expr_key(self, expr) -> int:
        """Return a stable key for TVM Expr objects across Python wrapper instances."""
        try:
            return int(expr.handle.value)
        except Exception:
            return id(expr)
        
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
                        # Handle array of IntImm
                        def _to_py(v):
                            if hasattr(v, 'value'): return v.value
                            try: return int(v)
                            except: return v
                        attrs[key] = [_to_py(x) for x in val]
                except:
                    pass
                    
        # Explicitly check for common spatial attributes that might be missed by dir()
        common_attrs = ['strides', 'padding', 'kernel_size', 'pool_size', 'dilation', 'groups', 'channels']
        if call.attrs is not None:
            for key in common_attrs:
                if key not in attrs and hasattr(call.attrs, key):
                    try:
                        val = getattr(call.attrs, key)
                        if val is None: continue
                        
                        if isinstance(val, (int, str, bool, float)):
                            attrs[key] = val
                        elif hasattr(val, '__iter__') and not isinstance(val, str):
                            def _to_py(v):
                                if hasattr(v, 'value'): return v.value
                                try: return int(v)
                                except: return v
                            attrs[key] = [_to_py(x) for x in val]
                    except:
                        pass
        return attrs
    
    def visit_var(self, var):
        """Record variable shapes."""
        self.var_shapes[var.name_hint] = self._get_shape(var)

    def visit_let(self, let):
        """Track producer layer index for Let-bound variables."""
        self.visit(let.value)
        if hasattr(let.var, 'name_hint'):
            src_idx = self.expr_to_layer_idx.get(self._expr_key(let.value), -1)
            if src_idx >= 0:
                self.var_to_layer_idx[let.var.name_hint] = src_idx
        self.visit(let.body)
        
    def visit_constant(self, const):
        """Record constant tensors."""
        try:
            data = const.data.numpy()
            self.constants[self._expr_key(const)] = data
        except:
            pass
    
    def _infer_padding(self, attrs: Dict[str, Any], op_name: str, input_shape: List[int], output_shape: List[int]) -> None:
        """
        Infer padding if explicit padding is 0 but shapes imply otherwise.
        This handles cases where TVM/ONNX uses implicit padding (like "SAME") 
        which resolves to 0 in attributes but requires padding in the explicit NPU kernel.
        """
        # Only relevant for ops with spatial dimensions
        if not output_shape or len(output_shape) < 4 or not input_shape or len(input_shape) < 4:
            return

        # Check if padding is explicitly 0 (or default)
        padding = attrs.get("padding", [0, 0, 0, 0])
        explicit_pad = padding[0] if isinstance(padding, list) and len(padding) > 0 else 0
        if explicit_pad != 0:
            return  # Trust explicit non-zero padding

        # Get kernel and stride
        kernel = None
        if "conv2d" in op_name:
            kernel = attrs.get("kernel_size")
        elif "pool2d" in op_name:
            kernel = attrs.get("pool_size")
            
        if not kernel: 
            return

        kh = kernel[0] if isinstance(kernel, list) else kernel
        
        # Get strides
        strides = attrs.get("strides", [1, 1])
        stride = strides[0] if isinstance(strides, list) else strides
        
        # Dimensions (N, C, H, W)
        in_h = input_shape[2]
        out_h = output_shape[2]
        
        # Calculate expected output with pad=0
        # Formula: floor((in - k)/stride) + 1
        expected_out_h = (in_h - kh) // stride + 1
        
        if expected_out_h < out_h:
            # We need padding to match the output shape
            # Formula: out = (in + 2P - k)/s + 1
            # (out - 1) * s = in + 2P - k
            # 2P = (out - 1) * s + k - in
            needed_2p = (out_h - 1) * stride + kh - in_h
            
            if needed_2p > 0:
                # Calculate symmetric pad (ceiling of half)
                pad = (needed_2p + 1) // 2
                attrs["padding"] = [pad, pad, pad, pad]
                # print(f"DEBUG: Inferred padding {pad} for {op_name} ({in_h} -> {out_h}, k={kh}, s={stride})")

    def _reconcile_shapes(self, attrs: Dict[str, Any], op_name: str, input_shape: List[int],
                          output_shape: List[int], call=None) -> List[int]:
        """
        Keep Relay output shape as the source of truth, and only补全 codegen 所需空间属性。
        """
        if not output_shape:
            return output_shape

        if "conv2d" in op_name:
            # Infer kernel_size from weight tensor shape when Relay attrs omit it.
            if "kernel_size" not in attrs and call is not None and len(call.args) > 1:
                w_shape = self._get_shape(call.args[1])
                if len(w_shape) >= 4:
                    attrs["kernel_size"] = [int(w_shape[2]), int(w_shape[3])]

            # Fill missing strides from shape ratio.
            if "strides" not in attrs:
                if len(input_shape) >= 4 and len(output_shape) >= 4 and output_shape[2] > 0 and output_shape[3] > 0:
                    sh = max(1, int(round(input_shape[2] / output_shape[2])))
                    sw = max(1, int(round(input_shape[3] / output_shape[3])))
                    attrs["strides"] = [sh, sw]
                else:
                    attrs["strides"] = [1, 1]

            # Validate/repair padding against Relay output shape.
            kh, kw = attrs.get("kernel_size", [1, 1])
            sh, sw = attrs.get("strides", [1, 1])
            pad_attr = attrs.get("padding", [0, 0, 0, 0])
            if isinstance(pad_attr, (int, float)):
                pad_attr = [int(pad_attr), int(pad_attr), int(pad_attr), int(pad_attr)]
            elif hasattr(pad_attr, "__iter__"):
                pad_attr = [int(x) for x in pad_attr]
                if len(pad_attr) == 2:
                    pad_attr = [pad_attr[0], pad_attr[1], pad_attr[0], pad_attr[1]]
                elif len(pad_attr) < 4:
                    pad_attr = [0, 0, 0, 0]
            else:
                pad_attr = [0, 0, 0, 0]

            if len(input_shape) >= 4 and len(output_shape) >= 4:
                in_h, in_w = input_shape[2], input_shape[3]
                out_h, out_w = output_shape[2], output_shape[3]

                calc_h = (in_h + pad_attr[0] + pad_attr[2] - kh) // sh + 1
                calc_w = (in_w + pad_attr[1] + pad_attr[3] - kw) // sw + 1

                if calc_h != out_h or calc_w != out_w:
                    pad_h_total = max(0, (out_h - 1) * sh + kh - in_h)
                    pad_w_total = max(0, (out_w - 1) * sw + kw - in_w)
                    pad_top = pad_h_total // 2
                    pad_bottom = pad_h_total - pad_top
                    pad_left = pad_w_total // 2
                    pad_right = pad_w_total - pad_left
                    attrs["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
                else:
                    attrs["padding"] = pad_attr
            else:
                attrs["padding"] = pad_attr

        return output_shape

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

        # Track producer layer index of each input expression (or -1 for Var/Const/input)
        input_layer_idxs = []
        for arg in call.args:
            src_idx = self.expr_to_layer_idx.get(self._expr_key(arg), -1)
            if src_idx < 0 and isinstance(arg, Var):
                src_idx = self.var_to_layer_idx.get(arg.name_hint, -1)
            input_layer_idxs.append(src_idx)
        attrs['_input_layers'] = input_layer_idxs
        
        # Get shapes
        input_shape = self._get_shape(call.args[0]) if call.args else []
        output_shape = self._get_shape(call)
        
        # Reconcile shapes with attributes to ensure codegen consistency
        output_shape = self._reconcile_shapes(attrs, op_name, input_shape, output_shape, call)

        
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
                        key = self._expr_key(call)
                        self.expand_dims_bias[key] = (bias_data, output_shape)
                        layer.attrs['_bias_data_id'] = key
                        layer.attrs['_bias_channels'] = len(bias_data) if bias_data.ndim == 1 else bias_data.shape[0]
                    except:
                        pass
                elif isinstance(bias_arg, Var):
                    # In freeze_params=False mode, bias often appears as Var instead of Constant.
                    bias_shape = self._get_shape(bias_arg)
                    if bias_shape:
                        channels = int(bias_shape[0])
                    elif len(output_shape) > 0:
                        channels = int(output_shape[0])
                    else:
                        channels = 0
                    self.expand_dims_bias[self._expr_key(call)] = (None, output_shape)
                    layer.attrs['_bias_var_name'] = bias_arg.name_hint
                    layer.attrs['_bias_channels'] = channels
        
        # Handle add: check if this is bias add (one input from expand_dims)
        if op_name == "add":
            for i, arg in enumerate(call.args):
                key = self._expr_key(arg)
                if key in self.expand_dims_bias:
                    bias_data, bias_shape = self.expand_dims_bias[key]
                    layer.attrs['_is_bias_add'] = True
                    if bias_data is not None:
                        layer.attrs['_bias_channels'] = len(bias_data) if bias_data.ndim == 1 else bias_data.shape[0]
                    elif bias_shape:
                        layer.attrs['_bias_channels'] = int(bias_shape[0])
                    else:
                        layer.attrs['_bias_channels'] = 0
                    # Mark which argument has the bias
                    layer.attrs['_bias_arg_idx'] = i
                    break

        def _capture_scalar(arg, name_key: str, value_key: str):
            if isinstance(arg, Var):
                layer.attrs[name_key] = arg.name_hint
            elif isinstance(arg, Constant):
                try:
                    arr = arg.data.numpy()
                    if arr.size > 0:
                        layer.attrs[value_key] = float(arr.reshape(-1)[0])
                except Exception:
                    pass

        # Handle qnn.requantize scalar parameters.
        if "qnn.requantize" in op_name and len(call.args) >= 5:
            _capture_scalar(call.args[1], '_rq_in_scale_name', '_rq_in_scale_val')
            _capture_scalar(call.args[2], '_rq_in_zp_name', '_rq_in_zp_val')
            _capture_scalar(call.args[3], '_rq_out_scale_name', '_rq_out_scale_val')
            _capture_scalar(call.args[4], '_rq_out_zp_name', '_rq_out_zp_val')

        # Handle qnn.dequantize scalar parameters.
        if "qnn.dequantize" in op_name and len(call.args) >= 3:
            _capture_scalar(call.args[1], '_dq_in_scale_name', '_dq_in_scale_val')
            _capture_scalar(call.args[2], '_dq_in_zp_name', '_dq_in_zp_val')

        # Handle qnn.quantize scalar parameters.
        if "qnn.quantize" in op_name and len(call.args) >= 3:
            _capture_scalar(call.args[1], '_q_out_scale_name', '_q_out_scale_val')
            _capture_scalar(call.args[2], '_q_out_zp_name', '_q_out_zp_val')

        # Handle qnn.conv2d scalar parameters.
        if "qnn.conv2d" in op_name and len(call.args) >= 6:
            _capture_scalar(call.args[2], '_qnn_in_zp_name', '_qnn_in_zp_val')
            _capture_scalar(call.args[3], '_qnn_w_zp_name', '_qnn_w_zp_val')
            _capture_scalar(call.args[4], '_qnn_in_scale_name', '_qnn_in_scale_val')
            _capture_scalar(call.args[5], '_qnn_w_scale_name', '_qnn_w_scale_val')

        # Handle qnn.dense scalar parameters.
        if "qnn.dense" in op_name and len(call.args) >= 6:
            _capture_scalar(call.args[2], '_qnn_in_zp_name', '_qnn_in_zp_val')
            _capture_scalar(call.args[3], '_qnn_w_zp_name', '_qnn_w_zp_val')
            _capture_scalar(call.args[4], '_qnn_in_scale_name', '_qnn_in_scale_val')
            _capture_scalar(call.args[5], '_qnn_w_scale_name', '_qnn_w_scale_val')
        
        self.layers.append(layer)
        self.expr_to_layer_idx[self._expr_key(call)] = layer.idx
        self.layer_idx += 1
        
        return call
