#!/usr/bin/env python3
"""
TVM Model Verification Script

This script verifies that TVM-compiled models produce correct results by:
1. Loading the original ONNX model
2. Converting to TVM Relay IR and applying optimizations
3. Running inference on both ONNX Runtime and TVM
4. Comparing outputs to ensure correctness

Usage:
    python verify_tvm_model.py --model mobilenetv2-7.onnx --image test.jpg
    python verify_tvm_model.py --model resnet50-v2-7.onnx --image_dir ./test_images/
"""

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image

# Patch ml_dtypes for compatibility
def _patch_ml_dtypes():
    try:
        import ml_dtypes
        if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
            ml_dtypes.float8_e4m3fnuz = ml_dtypes.float8_e4m3fn
        if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
            ml_dtypes.float8_e5m2fnuz = ml_dtypes.float8_e5m2
        if not hasattr(ml_dtypes, 'int4'):
            ml_dtypes.int4 = np.int8
        if not hasattr(ml_dtypes, 'uint4'):
            ml_dtypes.uint4 = np.uint8
        if not hasattr(ml_dtypes, 'float4_e2m1fn'):
            ml_dtypes.float4_e2m1fn = ml_dtypes.bfloat16
    except:
        pass
_patch_ml_dtypes()

import onnx
import onnxruntime
import tvm
from tvm import relay
from tvm.contrib import graph_executor


# ImageNet class labels (top 10 common ones for quick reference)
IMAGENET_LABELS = {
    0: "tench",
    1: "goldfish", 
    2: "great white shark",
    3: "tiger shark",
    4: "hammerhead",
    5: "electric ray",
    6: "stingray",
    7: "cock",
    8: "hen",
    9: "ostrich",
    # ... more labels can be loaded from file
}


def load_labels(labels_file=None):
    """Load ImageNet labels from file."""
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return {i: label for i, label in enumerate(labels)}
    return IMAGENET_LABELS


def preprocess_image(image_path, height=224, width=224, normalize=True):
    """
    Preprocess image for model inference.
    
    Args:
        image_path: Path to image file
        height: Target height
        width: Target width
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Preprocessed numpy array in NCHW format
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height), Image.BILINEAR)
    
    # Convert to numpy array
    img_data = np.array(img, dtype=np.float32)
    
    if normalize:
        # ImageNet normalization
        mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img_data = img_data - mean
    
    # NHWC -> NCHW
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data.transpose(0, 3, 1, 2)
    
    return img_data


def run_onnx_inference(model_path, input_data):
    """Run inference using ONNX Runtime."""
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    start = time.perf_counter()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed = (time.perf_counter() - start) * 1000
    
    return outputs[0], elapsed


def run_tvm_inference(model_path, input_data, opt_level=3):
    """
    Run inference using TVM with optimization passes.
    
    This demonstrates TVM's optimization capabilities.
    """
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    
    # Get input info
    input_name = onnx_model.graph.input[0].name
    input_shape = input_data.shape
    shape_dict = {input_name: input_shape}
    
    # Convert to Relay IR
    print(f"  Converting to TVM Relay IR...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    # Apply optimization passes
    print(f"  Applying TVM optimizations (opt_level={opt_level})...")
    with tvm.transform.PassContext(opt_level=opt_level):
        # These are the same passes used in our NPU compiler
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        mod = relay.transform.FoldScaleAxis()(mod)
        mod = relay.transform.CanonicalizeOps()(mod)
        mod = relay.transform.DeadCodeElimination()(mod)
        
        # Build for CPU (for verification)
        target = "llvm"
        lib = relay.build(mod, target=target, params=params)
    
    # Create runtime
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    
    # Set input
    module.set_input(input_name, input_data)
    
    # Run inference
    start = time.perf_counter()
    module.run()
    elapsed = (time.perf_counter() - start) * 1000
    
    # Get output
    output = module.get_output(0).numpy()
    
    return output, elapsed


def compare_outputs(onnx_output, tvm_output, rtol=1e-3, atol=1e-5):
    """Compare ONNX and TVM outputs."""
    # Check shapes match
    if onnx_output.shape != tvm_output.shape:
        print(f"  Shape mismatch: ONNX {onnx_output.shape} vs TVM {tvm_output.shape}")
        return False
    
    # Check values are close
    if not np.allclose(onnx_output, tvm_output, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(onnx_output - tvm_output))
        mean_diff = np.mean(np.abs(onnx_output - tvm_output))
        print(f"  Value mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        # Check if top-k predictions match
        onnx_topk = np.argsort(onnx_output.flatten())[-5:][::-1]
        tvm_topk = np.argsort(tvm_output.flatten())[-5:][::-1]
        
        if np.array_equal(onnx_topk, tvm_topk):
            print(f"  Top-5 predictions match despite numerical differences")
            return True
        else:
            print(f"  Top-5 ONNX: {onnx_topk}")
            print(f"  Top-5 TVM:  {tvm_topk}")
            return False
    
    return True


def get_top_predictions(output, labels, k=5):
    """Get top-k predictions with labels."""
    indices = np.argsort(output.flatten())[-k:][::-1]
    probs = np.exp(output.flatten()[indices]) / np.sum(np.exp(output.flatten()))
    
    results = []
    for i, idx in enumerate(indices):
        label = labels.get(idx, f"class_{idx}")
        results.append((idx, label, probs[i]))
    
    return results


def benchmark_model(model_path, input_shape=(1, 3, 224, 224), runs=10):
    """Benchmark both ONNX and TVM inference."""
    print(f"\nBenchmarking {model_path}...")
    print(f"Input shape: {input_shape}")
    print(f"Runs: {runs}")
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # ONNX Runtime benchmark
    print("\nONNX Runtime:")
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    _ = session.run(None, {input_name: input_data})
    
    onnx_times = []
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_data})
        elapsed = (time.perf_counter() - start) * 1000
        onnx_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}ms")
    
    print(f"  Average: {np.mean(onnx_times):.2f}ms")
    
    # TVM benchmark
    print("\nTVM (opt_level=3):")
    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: input_shape}
    
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        lib = relay.build(mod, target="llvm", params=params)
    
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, input_data)
    
    # Warmup
    module.run()
    
    tvm_times = []
    for i in range(runs):
        start = time.perf_counter()
        module.run()
        elapsed = (time.perf_counter() - start) * 1000
        tvm_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}ms")
    
    print(f"  Average: {np.mean(tvm_times):.2f}ms")
    print(f"\nSpeedup: {np.mean(onnx_times) / np.mean(tvm_times):.2f}x")


def verify_single_image(model_path, image_path, labels):
    """Verify model on a single image."""
    print(f"\n{'='*60}")
    print(f"Verifying: {model_path}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    print(f"Input shape: {input_data.shape}")
    
    # Run ONNX inference
    print("\n[ONNX Runtime]")
    onnx_output, onnx_time = run_onnx_inference(model_path, input_data)
    print(f"  Inference time: {onnx_time:.2f}ms")
    print(f"  Output shape: {onnx_output.shape}")
    
    onnx_preds = get_top_predictions(onnx_output, labels)
    print("  Top-5 predictions:")
    for idx, label, prob in onnx_preds:
        print(f"    {idx:4d}: {label} ({prob*100:.1f}%)")
    
    # Run TVM inference
    print("\n[TVM Optimized]")
    tvm_output, tvm_time = run_tvm_inference(model_path, input_data)
    print(f"  Inference time: {tvm_time:.2f}ms")
    print(f"  Output shape: {tvm_output.shape}")
    
    tvm_preds = get_top_predictions(tvm_output, labels)
    print("  Top-5 predictions:")
    for idx, label, prob in tvm_preds:
        print(f"    {idx:4d}: {label} ({prob*100:.1f}%)")
    
    # Compare outputs
    print("\n[Comparison]")
    match = compare_outputs(onnx_output, tvm_output)
    if match:
        print("  ✓ ONNX and TVM outputs match!")
    else:
        print("  ✗ Outputs differ (may still be acceptable)")
    
    print(f"  Speedup: {onnx_time/tvm_time:.2f}x")
    
    return match


def main():
    parser = argparse.ArgumentParser(description="Verify TVM model compilation")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--image_dir", help="Directory of test images")
    parser.add_argument("--labels", help="Path to labels file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs")
    args = parser.parse_args()
    
    # Load labels
    labels = load_labels(args.labels)
    
    if args.benchmark:
        benchmark_model(args.model, runs=args.runs)
        return
    
    if args.image:
        verify_single_image(args.model, args.image, labels)
    elif args.image_dir:
        # Process all images in directory
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"No images found in {args.image_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        all_match = True
        for img_file in image_files:
            img_path = os.path.join(args.image_dir, img_file)
            match = verify_single_image(args.model, img_path, labels)
            all_match = all_match and match
        
        print(f"\n{'='*60}")
        if all_match:
            print("All images verified successfully!")
        else:
            print("Some images had mismatches (check details above)")
    else:
        # No image provided, just benchmark
        print("No image provided. Running benchmark with random data...")
        benchmark_model(args.model, runs=args.runs)


if __name__ == "__main__":
    main()
