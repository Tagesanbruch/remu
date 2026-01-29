import os
import json
import numpy as np
from typing import Tuple, Dict
import tvm
from tvm import relay
import onnx
from onnx import numpy_helper

from .analyzer import RelayAnalyzer
from .codegen import NPUCodeGenerator
from .test_gen import generate_test_data, generate_test_program, generate_makefile

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
def load_and_optimize_model(onnx_path: str, input_shape: Tuple[int, ...]):
    """
    Load ONNX model and apply TVM optimizations.
    Returns: (mod, params, input_shape, onnx_weights, input_name)
    """
    print(f"Model:       {onnx_path}")
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
        
        # Constant folding
        mod = relay.transform.FoldConstant()(mod)
        
        # Simplify inference (remove training-only ops)
        mod = relay.transform.SimplifyInference()(mod)
        
        # Fold scale into weights
        mod = relay.transform.FoldScaleAxis()(mod)
        
        # Canonicalize operations
        mod = relay.transform.CanonicalizeOps()(mod)
        
        # Dead code elimination
        mod = relay.transform.DeadCodeElimination()(mod)
        
    return mod, params, input_shape, onnx_weights, input_name, onnx_model

def compile_model(onnx_path: str, 
                  output_dir: str,
                  model_name: str = "model",
                  input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """
    Compile ONNX model for REMU NPU using TVM.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TVM REMU NPU Compiler")
    print("=" * 70)
    
    # Load and optimize
    mod, params, input_shape, onnx_weights, input_name, onnx_model = load_and_optimize_model(onnx_path, input_shape)
    
    # Extract mappings
    conv_bias_map = {}
    dense_bias_map = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 3:
            weight_name = node.input[1]
            bias_name = node.input[2]
            if bias_name in onnx_weights:
                conv_bias_map[weight_name] = bias_name
        elif node.op_type == "Gemm" and len(node.input) >= 3:
            weight_name = node.input[1]
            bias_name = node.input[2]
            if bias_name in onnx_weights:
                dense_bias_map[weight_name] = bias_name

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
    # Note: We reuse the ONNX model loaded for compilation
    generate_test_data(onnx_model, test_dir, input_shape, 1.0/127.0)
    
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
    
    total_params = sum(w.size for w in onnx_weights.values())
    return {
        "weights_size": codegen.current_offset,
        "num_layers": len(layers),
        "op_counts": op_counts,
        "total_params": total_params,
    }
