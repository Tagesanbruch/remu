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
