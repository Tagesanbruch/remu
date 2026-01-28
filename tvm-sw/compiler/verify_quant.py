#!/usr/bin/env python3
"""
Verify quantized LeNet-5 inference in Python before running on REMU.
This simulates the exact same computation as the C code.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantize_weight(w, name):
    """Quantize float32 -> int8"""
    abs_max = max(abs(w.min()), abs(w.max()))
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / 127.0
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    print(f"  {name}: range=[{w.min():.4f}, {w.max():.4f}], scale={scale:.6f}")
    return q, scale


def quantize_input(x):
    """Quantize normalized input to int8"""
    # Input is already normalized to ~[-0.5, 2.8] for MNIST
    # Map to int8 range
    scale = max(abs(x.min()), abs(x.max())) / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def int8_matmul(a, b, m, n, k):
    """Simulate NPU GEMM: A[m,k] @ B[k,n] -> C[m,n] (int32)"""
    c = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            acc = 0
            for kk in range(k):
                acc += int(a[i, kk]) * int(b[kk, j])
            c[i, j] = acc
    return c


def int8_conv2d_im2col(inp, weight, in_c, in_h, in_w, out_c, kh, kw, pad, stride):
    """Simulate NPU conv2d using im2col + GEMM"""
    out_h = (in_h + 2 * pad - kh) // stride + 1
    out_w = (in_w + 2 * pad - kw) // stride + 1
    
    M = out_h * out_w
    N = out_c
    K = in_c * kh * kw
    
    # Im2col
    im2col = np.zeros((M, K), dtype=np.int8)
    for oh in range(out_h):
        for ow in range(out_w):
            m = oh * out_w + ow
            for ic in range(in_c):
                for ky in range(kh):
                    for kx in range(kw):
                        ih = oh * stride - pad + ky
                        iw = ow * stride - pad + kx
                        k_idx = ic * kh * kw + ky * kw + kx
                        
                        val = 0
                        if 0 <= ih < in_h and 0 <= iw < in_w:
                            val = inp[ic, ih, iw]
                        im2col[m, k_idx] = val
    
    # Weight: [out_c, in_c, kh, kw] -> [out_c, K] -> transpose to [K, out_c]
    weight_reshaped = weight.reshape(out_c, K)
    weight_t = weight_reshaped.T  # [K, N]
    
    # GEMM
    output = int8_matmul(im2col, weight_t, M, N, K)
    
    # Reshape to [out_c, out_h, out_w]
    return output.reshape(out_h, out_w, out_c).transpose(2, 0, 1)


def maxpool2x2(x, scale_shift):
    """2x2 max pooling with quantization"""
    c, h, w = x.shape
    oh, ow = h // 2, w // 2
    out = np.zeros((c, oh, ow), dtype=np.int8)
    
    for ch in range(c):
        for y in range(oh):
            for xx in range(ow):
                v = max(x[ch, y*2, xx*2], x[ch, y*2, xx*2+1],
                       x[ch, y*2+1, xx*2], x[ch, y*2+1, xx*2+1])
                q = v >> scale_shift
                out[ch, y, xx] = np.clip(q, -128, 127)
    
    return out


def quantize_i32_to_i8(x, scale_shift):
    """Quantize int32 -> int8"""
    q = x >> scale_shift
    return np.clip(q, -128, 127).astype(np.int8)


def simulate_inference(model, x_float):
    """Simulate quantized inference"""
    print("\n=== Simulating Quantized Inference ===")
    
    # Get weights
    conv1_w = model.conv1.weight.detach().numpy()  # [6, 1, 5, 5]
    conv1_b = model.conv1.bias.detach().numpy()
    conv2_w = model.conv2.weight.detach().numpy()  # [16, 6, 5, 5]
    conv2_b = model.conv2.bias.detach().numpy()
    fc1_w = model.fc1.weight.detach().numpy()      # [120, 400]
    fc1_b = model.fc1.bias.detach().numpy()
    fc2_w = model.fc2.weight.detach().numpy()      # [84, 120]
    fc2_b = model.fc2.bias.detach().numpy()
    fc3_w = model.fc3.weight.detach().numpy()      # [10, 84]
    fc3_b = model.fc3.bias.detach().numpy()
    
    # Quantize weights
    print("\nQuantizing weights:")
    conv1_wq, s1 = quantize_weight(conv1_w, "conv1.weight")
    conv1_bq, sb1 = quantize_weight(conv1_b, "conv1.bias")
    conv2_wq, s2 = quantize_weight(conv2_w, "conv2.weight")
    conv2_bq, sb2 = quantize_weight(conv2_b, "conv2.bias")
    fc1_wq, sf1 = quantize_weight(fc1_w, "fc1.weight")
    fc1_bq, sfb1 = quantize_weight(fc1_b, "fc1.bias")
    fc2_wq, sf2 = quantize_weight(fc2_w, "fc2.weight")
    fc2_bq, sfb2 = quantize_weight(fc2_b, "fc2.bias")
    fc3_wq, sf3 = quantize_weight(fc3_w, "fc3.weight")
    fc3_bq, sfb3 = quantize_weight(fc3_b, "fc3.bias")
    
    # Quantize input
    x_np = x_float.squeeze().numpy()  # [28, 28]
    x_q, sx = quantize_input(x_np)
    x_q = x_q.reshape(1, 28, 28)  # [1, 28, 28]
    print(f"\nInput quantized: range=[{x_q.min()}, {x_q.max()}], scale={sx:.6f}")
    
    # Conv1: 1x28x28 -> 6x28x28
    print("\nConv1...")
    conv1_out = int8_conv2d_im2col(x_q, conv1_wq, 1, 28, 28, 6, 5, 5, 2, 1)
    print(f"  Conv1 out shape: {conv1_out.shape}, range: [{conv1_out.min()}, {conv1_out.max()}]")
    
    # Add bias (scaled)
    for c in range(6):
        conv1_out[c] += int(conv1_bq[c]) << 8
    
    # ReLU
    conv1_out = np.maximum(conv1_out, 0)
    
    # Pool: 6x28x28 -> 6x14x14
    pool1_out = maxpool2x2(conv1_out, 8)
    print(f"  Pool1 out shape: {pool1_out.shape}, range: [{pool1_out.min()}, {pool1_out.max()}]")
    
    # Conv2: 6x14x14 -> 16x10x10
    print("\nConv2...")
    conv2_out = int8_conv2d_im2col(pool1_out, conv2_wq, 6, 14, 14, 16, 5, 5, 0, 1)
    print(f"  Conv2 out shape: {conv2_out.shape}, range: [{conv2_out.min()}, {conv2_out.max()}]")
    
    # Add bias
    for c in range(16):
        conv2_out[c] += int(conv2_bq[c]) << 8
    
    # ReLU
    conv2_out = np.maximum(conv2_out, 0)
    
    # Pool: 16x10x10 -> 16x5x5
    pool2_out = maxpool2x2(conv2_out, 8)
    print(f"  Pool2 out shape: {pool2_out.shape}, range: [{pool2_out.min()}, {pool2_out.max()}]")
    
    # Flatten
    flatten = pool2_out.flatten()
    print(f"  Flatten shape: {flatten.shape}")
    
    # FC1: 400 -> 120
    print("\nFC1...")
    # fc1_w shape is [120, 400], we need [400, 120] for matmul
    fc1_out = int8_matmul(flatten.reshape(1, 400), fc1_wq.T, 1, 120, 400)
    for i in range(120):
        fc1_out[0, i] += int(fc1_bq[i]) << 8
        if fc1_out[0, i] < 0:
            fc1_out[0, i] = 0
    fc1_out_q = quantize_i32_to_i8(fc1_out, 8).flatten()
    print(f"  FC1 out range: [{fc1_out.min()}, {fc1_out.max()}]")
    
    # FC2: 120 -> 84
    print("\nFC2...")
    fc2_out = int8_matmul(fc1_out_q.reshape(1, 120), fc2_wq.T, 1, 84, 120)
    for i in range(84):
        fc2_out[0, i] += int(fc2_bq[i]) << 8
        if fc2_out[0, i] < 0:
            fc2_out[0, i] = 0
    fc2_out_q = quantize_i32_to_i8(fc2_out, 8).flatten()
    print(f"  FC2 out range: [{fc2_out.min()}, {fc2_out.max()}]")
    
    # FC3: 84 -> 10
    print("\nFC3...")
    fc3_out = int8_matmul(fc2_out_q.reshape(1, 84), fc3_wq.T, 1, 10, 84)
    for i in range(10):
        fc3_out[0, i] += int(fc3_bq[i]) << 8
    
    print(f"  FC3 output: {fc3_out.flatten()}")
    
    pred = np.argmax(fc3_out)
    print(f"\n  Predicted: {pred}")
    
    return pred


def main():
    # Load model
    model = LeNet5()
    model_path = os.path.join(OUTPUT_DIR, 'lenet5.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Get test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    
    # Find one sample of each digit
    correct = 0
    total = 10
    
    for digit in range(10):
        # Find first sample of this digit
        for i, (x, y) in enumerate(test_dataset):
            if y == digit:
                print(f"\n{'='*50}")
                print(f"Testing digit {digit} (sample {i})")
                print(f"{'='*50}")
                
                # Float inference for reference
                with torch.no_grad():
                    float_out = model(x.unsqueeze(0))
                    float_pred = float_out.argmax().item()
                print(f"Float inference: {float_out.numpy().flatten()}")
                print(f"Float predicted: {float_pred}")
                
                # Quantized inference
                q_pred = simulate_inference(model, x)
                
                if q_pred == digit:
                    correct += 1
                    print(f"\n>>> PASS")
                else:
                    print(f"\n>>> FAIL")
                
                break
    
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} = {correct*100//total}%")


if __name__ == "__main__":
    main()
