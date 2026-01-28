#!/usr/bin/env python3
"""
PTQ (Post-Training Quantization) for LeNet-5.
Implements correct per-layer quantization with scale propagation.

Quantization scheme:
- Input: int8 with input_scale
- Weights: int8 with weight_scale (per-layer)
- Bias: int32 (scaled to match accumulator scale)
- Output accumulator: int32
- Output: int8 with output_scale (after requantize)

Scale relationship:
  output_scale = input_scale * weight_scale
  bias_scale = input_scale * weight_scale (to match accumulator)
  
After GEMM: acc[i32] = sum(a[i8] * w[i8])
Requantize: out[i8] = (acc[i32] * output_multiplier) >> output_shift
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


def calibrate_scale(data, num_bits=8):
    """Calculate symmetric quantization scale for data."""
    abs_max = max(abs(data.min()), abs(data.max()))
    if abs_max == 0:
        abs_max = 1e-6
    qmax = (1 << (num_bits - 1)) - 1  # 127 for int8
    return abs_max / qmax


def quantize_symmetric(data, scale):
    """Quantize float to int8 using symmetric quantization."""
    q = np.round(data / scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def compute_requant_params(acc_scale, out_scale):
    """
    Compute requantization parameters.
    acc_scale = input_scale * weight_scale
    out_scale = desired output scale
    
    We need: out = acc * (acc_scale / out_scale)
    Decompose as: out = (acc * multiplier) >> shift
    """
    real_mult = acc_scale / out_scale
    
    # Find shift such that multiplier fits in int32
    shift = 0
    while real_mult < 0.5 and shift < 31:
        real_mult *= 2
        shift += 1
    
    # Multiplier as fixed-point (Q31)
    multiplier = int(round(real_mult * (1 << 31)))
    
    # Total shift (31 for Q31 + additional)
    total_shift = 31 - shift
    
    return multiplier, total_shift


def requantize(acc, multiplier, shift):
    """Requantize int32 accumulator to int8."""
    # Use int64 for multiplication
    result = (acc.astype(np.int64) * multiplier) >> shift
    return np.clip(result, -128, 127).astype(np.int8)


class QuantizedLeNet:
    """Quantized LeNet-5 with proper scale tracking."""
    
    def __init__(self, model, calibration_data):
        """
        Initialize quantized model by calibrating scales on sample data.
        """
        self.model = model
        model.eval()
        
        print("=== Calibrating scales ===")
        
        # Run calibration samples to get activation ranges
        with torch.no_grad():
            activations = self._collect_activations(calibration_data)
        
        # Calculate scales for each layer
        self.input_scale = calibrate_scale(activations['input'])
        print(f"Input scale: {self.input_scale:.6f}")
        
        # Quantize weights and compute layer parameters
        self._quantize_conv1(activations)
        self._quantize_conv2(activations)
        self._quantize_fc1(activations)
        self._quantize_fc2(activations)
        self._quantize_fc3(activations)
        
        print("\nQuantization complete!")
    
    def _collect_activations(self, data):
        """Run forward pass and collect activation ranges."""
        activations = {
            'input': [], 'conv1_out': [], 'pool1_out': [],
            'conv2_out': [], 'pool2_out': [], 'fc1_out': [],
            'fc2_out': [], 'fc3_out': []
        }
        
        for x, _ in data:
            x = x.unsqueeze(0)
            activations['input'].append(x.numpy())
            
            # Conv1 -> ReLU -> Pool
            x = self.model.conv1(x)
            x = self.model.relu(x)
            activations['conv1_out'].append(x.numpy())
            x = self.model.pool(x)
            activations['pool1_out'].append(x.numpy())
            
            # Conv2 -> ReLU -> Pool
            x = self.model.conv2(x)
            x = self.model.relu(x)
            activations['conv2_out'].append(x.numpy())
            x = self.model.pool(x)
            activations['pool2_out'].append(x.numpy())
            
            # Flatten
            x = x.view(-1, 16 * 5 * 5)
            
            # FC1 -> ReLU
            x = self.model.fc1(x)
            x = self.model.relu(x)
            activations['fc1_out'].append(x.numpy())
            
            # FC2 -> ReLU
            x = self.model.fc2(x)
            x = self.model.relu(x)
            activations['fc2_out'].append(x.numpy())
            
            # FC3
            x = self.model.fc3(x)
            activations['fc3_out'].append(x.numpy())
        
        # Concatenate all samples
        for key in activations:
            activations[key] = np.concatenate(activations[key])
        
        return activations
    
    def _quantize_conv1(self, activations):
        """Quantize conv1 layer."""
        w = self.model.conv1.weight.detach().numpy()
        b = self.model.conv1.bias.detach().numpy()
        
        self.conv1_weight_scale = calibrate_scale(w)
        self.conv1_output_scale = calibrate_scale(activations['pool1_out'])
        
        self.conv1_weight = quantize_symmetric(w, self.conv1_weight_scale)
        
        # Bias scale = input_scale * weight_scale
        bias_scale = self.input_scale * self.conv1_weight_scale
        self.conv1_bias = np.round(b / bias_scale).astype(np.int32)
        
        # Requant params
        acc_scale = self.input_scale * self.conv1_weight_scale
        self.conv1_mult, self.conv1_shift = compute_requant_params(acc_scale, self.conv1_output_scale)
        
        print(f"Conv1: weight_scale={self.conv1_weight_scale:.6f}, "
              f"output_scale={self.conv1_output_scale:.6f}, "
              f"mult={self.conv1_mult}, shift={self.conv1_shift}")
    
    def _quantize_conv2(self, activations):
        """Quantize conv2 layer."""
        w = self.model.conv2.weight.detach().numpy()
        b = self.model.conv2.bias.detach().numpy()
        
        self.conv2_weight_scale = calibrate_scale(w)
        self.conv2_output_scale = calibrate_scale(activations['pool2_out'])
        
        self.conv2_weight = quantize_symmetric(w, self.conv2_weight_scale)
        
        # Bias scale = input_scale (pool1) * weight_scale
        bias_scale = self.conv1_output_scale * self.conv2_weight_scale
        self.conv2_bias = np.round(b / bias_scale).astype(np.int32)
        
        acc_scale = self.conv1_output_scale * self.conv2_weight_scale
        self.conv2_mult, self.conv2_shift = compute_requant_params(acc_scale, self.conv2_output_scale)
        
        print(f"Conv2: weight_scale={self.conv2_weight_scale:.6f}, "
              f"output_scale={self.conv2_output_scale:.6f}")
    
    def _quantize_fc1(self, activations):
        """Quantize fc1 layer."""
        w = self.model.fc1.weight.detach().numpy()
        b = self.model.fc1.bias.detach().numpy()
        
        self.fc1_weight_scale = calibrate_scale(w)
        self.fc1_output_scale = calibrate_scale(activations['fc1_out'])
        
        self.fc1_weight = quantize_symmetric(w, self.fc1_weight_scale)
        
        bias_scale = self.conv2_output_scale * self.fc1_weight_scale
        self.fc1_bias = np.round(b / bias_scale).astype(np.int32)
        
        acc_scale = self.conv2_output_scale * self.fc1_weight_scale
        self.fc1_mult, self.fc1_shift = compute_requant_params(acc_scale, self.fc1_output_scale)
        
        print(f"FC1: weight_scale={self.fc1_weight_scale:.6f}, "
              f"output_scale={self.fc1_output_scale:.6f}")
    
    def _quantize_fc2(self, activations):
        """Quantize fc2 layer."""
        w = self.model.fc2.weight.detach().numpy()
        b = self.model.fc2.bias.detach().numpy()
        
        self.fc2_weight_scale = calibrate_scale(w)
        self.fc2_output_scale = calibrate_scale(activations['fc2_out'])
        
        self.fc2_weight = quantize_symmetric(w, self.fc2_weight_scale)
        
        bias_scale = self.fc1_output_scale * self.fc2_weight_scale
        self.fc2_bias = np.round(b / bias_scale).astype(np.int32)
        
        acc_scale = self.fc1_output_scale * self.fc2_weight_scale
        self.fc2_mult, self.fc2_shift = compute_requant_params(acc_scale, self.fc2_output_scale)
        
        print(f"FC2: weight_scale={self.fc2_weight_scale:.6f}, "
              f"output_scale={self.fc2_output_scale:.6f}")
    
    def _quantize_fc3(self, activations):
        """Quantize fc3 layer (output layer - keep int32 for argmax)."""
        w = self.model.fc3.weight.detach().numpy()
        b = self.model.fc3.bias.detach().numpy()
        
        self.fc3_weight_scale = calibrate_scale(w)
        self.fc3_output_scale = calibrate_scale(activations['fc3_out'])
        
        self.fc3_weight = quantize_symmetric(w, self.fc3_weight_scale)
        
        bias_scale = self.fc2_output_scale * self.fc3_weight_scale
        self.fc3_bias = np.round(b / bias_scale).astype(np.int32)
        
        # For FC3, we don't requantize - keep as int32 for argmax
        print(f"FC3: weight_scale={self.fc3_weight_scale:.6f}, "
              f"output_scale={self.fc3_output_scale:.6f}")
    
    def infer(self, x_float):
        """Run quantized inference."""
        # Quantize input
        x_q = quantize_symmetric(x_float.numpy(), self.input_scale)
        x_q = x_q.reshape(1, 28, 28)
        
        # Conv1
        conv1_out = self._conv2d(x_q, self.conv1_weight, self.conv1_bias,
                                  1, 28, 28, 6, 5, 5, 2, 1)
        # ReLU
        conv1_out = np.maximum(conv1_out, 0)
        # Requantize + Pool
        pool1_out = self._maxpool_requant(conv1_out, 6, 28, 28, 
                                           self.conv1_mult, self.conv1_shift)
        
        # Conv2
        conv2_out = self._conv2d(pool1_out, self.conv2_weight, self.conv2_bias,
                                  6, 14, 14, 16, 5, 5, 0, 1)
        conv2_out = np.maximum(conv2_out, 0)
        pool2_out = self._maxpool_requant(conv2_out, 16, 10, 10,
                                           self.conv2_mult, self.conv2_shift)
        
        # Flatten
        flatten = pool2_out.flatten()
        
        # FC1
        fc1_out = self._matmul(flatten, self.fc1_weight, self.fc1_bias, 1, 120, 400)
        fc1_out = np.maximum(fc1_out, 0)  # ReLU
        fc1_out_q = requantize(fc1_out, self.fc1_mult, self.fc1_shift).flatten()
        
        # FC2
        fc2_out = self._matmul(fc1_out_q, self.fc2_weight, self.fc2_bias, 1, 84, 120)
        fc2_out = np.maximum(fc2_out, 0)  # ReLU
        fc2_out_q = requantize(fc2_out, self.fc2_mult, self.fc2_shift).flatten()
        
        # FC3 (no ReLU, keep int32)
        fc3_out = self._matmul(fc2_out_q, self.fc3_weight, self.fc3_bias, 1, 10, 84)
        
        return fc3_out.flatten(), np.argmax(fc3_out)
    
    def _conv2d(self, inp, weight, bias, in_c, in_h, in_w, out_c, kh, kw, pad, stride):
        """Conv2D using im2col + matmul."""
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
                            
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                im2col[m, k_idx] = inp[ic, ih, iw]
        
        # Weight transpose
        weight_t = weight.reshape(out_c, K).T  # [K, N]
        
        # Matmul
        acc = np.zeros((M, N), dtype=np.int32)
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    acc[m, n] += int(im2col[m, k]) * int(weight_t[k, n])
                acc[m, n] += bias[n]
        
        return acc.reshape(out_h, out_w, out_c).transpose(2, 0, 1)
    
    def _maxpool_requant(self, x, c, h, w, mult, shift):
        """Max pooling 2x2 with requantization."""
        oh, ow = h // 2, w // 2
        out = np.zeros((c, oh, ow), dtype=np.int8)
        
        for ch in range(c):
            for y in range(oh):
                for xx in range(ow):
                    v = max(x[ch, y*2, xx*2], x[ch, y*2, xx*2+1],
                           x[ch, y*2+1, xx*2], x[ch, y*2+1, xx*2+1])
                    # Requantize
                    q = (int(v) * mult) >> shift
                    out[ch, y, xx] = np.clip(q, -128, 127)
        
        return out
    
    def _matmul(self, a, weight, bias, m, n, k):
        """Matrix multiplication A[m,k] @ W[n,k].T -> [m,n]"""
        # weight is [n, k], we need W.T = [k, n]
        acc = np.zeros((m, n), dtype=np.int32)
        a = a.flatten()
        weight_t = weight.T  # [k, n]
        
        for i in range(m):
            for j in range(n):
                for kk in range(k):
                    acc[i, j] += int(a[i * k + kk]) * int(weight_t[kk, j])
                acc[i, j] += bias[j]
        
        return acc
    
    def export_weights(self, output_path):
        """Export quantized weights to C header file."""
        with open(output_path, 'w') as f:
            f.write("// Auto-generated quantized LeNet-5 weights\n")
            f.write("// Generated with proper PTQ scale tracking\n")
            f.write("#ifndef __QUANT_WEIGHTS_H__\n")
            f.write("#define __QUANT_WEIGHTS_H__\n\n")
            f.write("#include <stdint.h>\n\n")
            
            # Export scales
            f.write("// Quantization parameters\n")
            f.write(f"#define INPUT_SCALE {self.input_scale}f\n")
            f.write(f"#define CONV1_MULT {self.conv1_mult}\n")
            f.write(f"#define CONV1_SHIFT {self.conv1_shift}\n")
            f.write(f"#define CONV2_MULT {self.conv2_mult}\n")
            f.write(f"#define CONV2_SHIFT {self.conv2_shift}\n")
            f.write(f"#define FC1_MULT {self.fc1_mult}\n")
            f.write(f"#define FC1_SHIFT {self.fc1_shift}\n")
            f.write(f"#define FC2_MULT {self.fc2_mult}\n")
            f.write(f"#define FC2_SHIFT {self.fc2_shift}\n\n")
            
            # Export weights
            self._write_array(f, "conv1_weight", self.conv1_weight, "int8_t")
            self._write_array(f, "conv1_bias", self.conv1_bias, "int32_t")
            self._write_array(f, "conv2_weight", self.conv2_weight, "int8_t")
            self._write_array(f, "conv2_bias", self.conv2_bias, "int32_t")
            self._write_array(f, "fc1_weight", self.fc1_weight, "int8_t")
            self._write_array(f, "fc1_bias", self.fc1_bias, "int32_t")
            self._write_array(f, "fc2_weight", self.fc2_weight, "int8_t")
            self._write_array(f, "fc2_bias", self.fc2_bias, "int32_t")
            self._write_array(f, "fc3_weight", self.fc3_weight, "int8_t")
            self._write_array(f, "fc3_bias", self.fc3_bias, "int32_t")
            
            f.write("#endif // __QUANT_WEIGHTS_H__\n")
        
        print(f"Exported weights to {output_path}")
    
    def _write_array(self, f, name, arr, dtype):
        """Write array to C header."""
        flat = arr.flatten()
        f.write(f"// {name}: shape={list(arr.shape)}\n")
        f.write(f"static const {dtype} q_{name}[{len(flat)}] = {{\n")
        
        for i in range(0, len(flat), 16):
            row = flat[i:i+16]
            if dtype == "int8_t":
                row_str = ", ".join(f"{v:4d}" for v in row)
            else:
                row_str = ", ".join(f"{v}" for v in row)
            f.write(f"    {row_str},\n")
        
        f.write("};\n\n")


def main():
    # Load model
    model = LeNet5()
    model_path = os.path.join(OUTPUT_DIR, 'lenet5.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load calibration data (use first 100 test samples)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    calibration_data = [test_dataset[i] for i in range(100)]
    
    # Create quantized model
    qmodel = QuantizedLeNet(model, calibration_data)
    
    # Export weights
    qmodel.export_weights(os.path.join(OUTPUT_DIR, 'quant_weights.h'))
    
    # Verify on test samples
    print("\n=== Verification ===")
    correct = 0
    total = 10
    
    for digit in range(10):
        for i, (x, y) in enumerate(test_dataset):
            if y == digit:
                # Float reference
                with torch.no_grad():
                    float_out = model(x.unsqueeze(0))
                    float_pred = float_out.argmax().item()
                
                # Quantized inference
                q_out, q_pred = qmodel.infer(x.squeeze())
                
                status = "PASS" if q_pred == digit else "FAIL"
                print(f"Digit {digit}: float_pred={float_pred}, quant_pred={q_pred} -> {status}")
                
                if q_pred == digit:
                    correct += 1
                break
    
    print(f"\nAccuracy: {correct}/{total} = {correct*100//total}%")


if __name__ == "__main__":
    main()
