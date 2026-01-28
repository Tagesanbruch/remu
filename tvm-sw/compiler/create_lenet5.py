#!/usr/bin/env python3
"""
Create a standard LeNet-5 ONNX model with smaller weights for testing.
"""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    import os
    os.system("uv add onnx")
    import onnx
    from onnx import helper, TensorProto, numpy_helper


def create_lenet5_onnx():
    """Create a standard LeNet-5 model."""
    
    # Input: 1x28x28 (MNIST)
    # Conv1: 6 filters, 5x5 -> 6x24x24
    # Pool1: 2x2 -> 6x12x12  
    # Conv2: 16 filters, 5x5 -> 16x8x8
    # Pool2: 2x2 -> 16x4x4
    # FC1: 256 -> 120
    # FC2: 120 -> 84
    # FC3: 84 -> 10
    
    np.random.seed(42)
    
    # Initialize weights with small random values
    conv1_w = (np.random.randn(6, 1, 5, 5) * 0.1).astype(np.float32)
    conv1_b = np.zeros(6, dtype=np.float32)
    conv2_w = (np.random.randn(16, 6, 5, 5) * 0.1).astype(np.float32)
    conv2_b = np.zeros(16, dtype=np.float32)
    fc1_w = (np.random.randn(120, 256) * 0.1).astype(np.float32)
    fc1_b = np.zeros(120, dtype=np.float32)
    fc2_w = (np.random.randn(84, 120) * 0.1).astype(np.float32)
    fc2_b = np.zeros(84, dtype=np.float32)
    fc3_w = (np.random.randn(10, 84) * 0.1).astype(np.float32)
    fc3_b = np.zeros(10, dtype=np.float32)
    
    # Create initializers
    initializers = [
        numpy_helper.from_array(conv1_w, "conv1_weight"),
        numpy_helper.from_array(conv1_b, "conv1_bias"),
        numpy_helper.from_array(conv2_w, "conv2_weight"),
        numpy_helper.from_array(conv2_b, "conv2_bias"),
        numpy_helper.from_array(fc1_w, "fc1_weight"),
        numpy_helper.from_array(fc1_b, "fc1_bias"),
        numpy_helper.from_array(fc2_w, "fc2_weight"),
        numpy_helper.from_array(fc2_b, "fc2_bias"),
        numpy_helper.from_array(fc3_w, "fc3_weight"),
        numpy_helper.from_array(fc3_b, "fc3_bias"),
    ]
    
    # Create nodes
    nodes = [
        # Conv1 + ReLU
        helper.make_node("Conv", ["input", "conv1_weight", "conv1_bias"], ["conv1_out"], 
                        kernel_shape=[5, 5], pads=[0, 0, 0, 0]),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"]),
        helper.make_node("MaxPool", ["relu1_out"], ["pool1_out"], 
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Conv2 + ReLU
        helper.make_node("Conv", ["pool1_out", "conv2_weight", "conv2_bias"], ["conv2_out"],
                        kernel_shape=[5, 5], pads=[0, 0, 0, 0]),
        helper.make_node("Relu", ["conv2_out"], ["relu2_out"]),
        helper.make_node("MaxPool", ["relu2_out"], ["pool2_out"],
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Flatten
        helper.make_node("Flatten", ["pool2_out"], ["flat_out"], axis=1),
        
        # FC1 + ReLU
        helper.make_node("Gemm", ["flat_out", "fc1_weight", "fc1_bias"], ["fc1_out"],
                        transB=1),
        helper.make_node("Relu", ["fc1_out"], ["relu3_out"]),
        
        # FC2 + ReLU
        helper.make_node("Gemm", ["relu3_out", "fc2_weight", "fc2_bias"], ["fc2_out"],
                        transB=1),
        helper.make_node("Relu", ["fc2_out"], ["relu4_out"]),
        
        # FC3 (output)
        helper.make_node("Gemm", ["relu4_out", "fc3_weight", "fc3_bias"], ["output"],
                        transB=1),
    ]
    
    # Input/Output
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    ]
    
    # Create graph
    graph = helper.make_graph(nodes, "lenet5", inputs, outputs, initializers)
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    
    # Validate
    onnx.checker.check_model(model)
    
    return model


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "../onnx/lenet5_simple.onnx")
    
    print("Creating LeNet-5 ONNX model...")
    model = create_lenet5_onnx()
    
    # Save
    onnx.save(model, output_path)
    print(f"Saved: {output_path}")
    
    # Print summary
    total_params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total_params += arr.size
        print(f"  {init.name}: {arr.shape}")
    
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    main()
