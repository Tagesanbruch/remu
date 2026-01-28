#!/usr/bin/env python3
"""
Simple PTQ for LeNet-5 - using per-layer scale with simple shift-based requantization.

Key insight: The int8*int8 -> int16/int32 accumulator needs to be properly scaled
back to int8 range for the next layer.

Approach:
1. Quantize weights to int8
2. Use float scales for simplicity 
3. At each layer output, scale by (1/scale) then clip to int8
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


def get_calibration_stats(model, data):
    """Get min/max of activations for calibration."""
    model.eval()
    stats = {}
    
    def make_hook(name):
        def hook(module, inp, out):
            arr = out.detach().numpy()
            if name not in stats:
                stats[name] = {'min': arr.min(), 'max': arr.max()}
            else:
                stats[name]['min'] = min(stats[name]['min'], arr.min())
                stats[name]['max'] = max(stats[name]['max'], arr.max())
        return hook
    
    handles = []
    handles.append(model.conv1.register_forward_hook(make_hook('conv1')))
    handles.append(model.conv2.register_forward_hook(make_hook('conv2')))
    handles.append(model.fc1.register_forward_hook(make_hook('fc1')))
    handles.append(model.fc2.register_forward_hook(make_hook('fc2')))
    handles.append(model.fc3.register_forward_hook(make_hook('fc3')))
    
    with torch.no_grad():
        for x, _ in data:
            x = x.unsqueeze(0)
            stats['input'] = {'min': x.numpy().min(), 'max': x.numpy().max()}
            _ = model(x)
    
    for h in handles:
        h.remove()
    
    return stats


def quantize_to_int8(x, scale):
    """Quantize float to int8 with given scale."""
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)


def dequantize_from_int8(x, scale):
    """Dequantize int8 to float."""
    return x.astype(np.float32) * scale


class SimpleQuantModel:
    """Simple quantized model using fake quantization approach."""
    
    def __init__(self, model, stats):
        self.model = model
        
        # Calculate scales from calibration stats
        def calc_scale(s):
            m = max(abs(s['min']), abs(s['max']))
            return m / 127.0 if m > 0 else 1.0
        
        self.input_scale = calc_scale(stats['input'])
        self.conv1_out_scale = calc_scale(stats['conv1'])  # After ReLU
        self.conv2_out_scale = calc_scale(stats['conv2'])
        self.fc1_out_scale = calc_scale(stats['fc1'])
        self.fc2_out_scale = calc_scale(stats['fc2'])
        self.fc3_out_scale = calc_scale(stats['fc3'])
        
        # Quantize weights
        w = model.conv1.weight.detach().numpy()
        self.conv1_weight_scale = max(abs(w.min()), abs(w.max())) / 127.0
        self.conv1_weight = quantize_to_int8(w, self.conv1_weight_scale)
        self.conv1_bias = model.conv1.bias.detach().numpy()
        
        w = model.conv2.weight.detach().numpy()
        self.conv2_weight_scale = max(abs(w.min()), abs(w.max())) / 127.0
        self.conv2_weight = quantize_to_int8(w, self.conv2_weight_scale)
        self.conv2_bias = model.conv2.bias.detach().numpy()
        
        w = model.fc1.weight.detach().numpy()
        self.fc1_weight_scale = max(abs(w.min()), abs(w.max())) / 127.0
        self.fc1_weight = quantize_to_int8(w, self.fc1_weight_scale)
        self.fc1_bias = model.fc1.bias.detach().numpy()
        
        w = model.fc2.weight.detach().numpy()
        self.fc2_weight_scale = max(abs(w.min()), abs(w.max())) / 127.0
        self.fc2_weight = quantize_to_int8(w, self.fc2_weight_scale)
        self.fc2_bias = model.fc2.bias.detach().numpy()
        
        w = model.fc3.weight.detach().numpy()
        self.fc3_weight_scale = max(abs(w.min()), abs(w.max())) / 127.0
        self.fc3_weight = quantize_to_int8(w, self.fc3_weight_scale)
        self.fc3_bias = model.fc3.bias.detach().numpy()
        
        print(f"Scales:")
        print(f"  input: {self.input_scale:.6f}")
        print(f"  conv1: w={self.conv1_weight_scale:.6f}, out={self.conv1_out_scale:.6f}")
        print(f"  conv2: w={self.conv2_weight_scale:.6f}, out={self.conv2_out_scale:.6f}")
        print(f"  fc1:   w={self.fc1_weight_scale:.6f}, out={self.fc1_out_scale:.6f}")
        print(f"  fc2:   w={self.fc2_weight_scale:.6f}, out={self.fc2_out_scale:.6f}")
        print(f"  fc3:   w={self.fc3_weight_scale:.6f}, out={self.fc3_out_scale:.6f}")
    
    def infer(self, x_float, verbose=False):
        """
        Run inference using integer arithmetic where possible.
        
        For each layer:
        1. Input is int8 with known scale
        2. Weight is int8 with known scale  
        3. Compute in int32: sum(a*w)
        4. Convert to float: multiply by (input_scale * weight_scale)
        5. Add float bias
        6. Apply activation
        7. Requantize to int8 with output scale
        """
        # Quantize input
        x_q = quantize_to_int8(x_float.numpy(), self.input_scale)
        x_q = x_q.reshape(1, 28, 28)
        
        if verbose:
            print(f"Input: min={x_q.min()}, max={x_q.max()}")
        
        # Conv1: int8 input, int8 weight -> int32 acc -> float -> int8
        conv1_acc = self._conv2d_int8(x_q, self.conv1_weight, 1, 28, 28, 6, 5, 5, 2, 1)
        # Convert to float and add bias
        conv1_float = conv1_acc.astype(np.float32) * (self.input_scale * self.conv1_weight_scale)
        for c in range(6):
            conv1_float[c] += self.conv1_bias[c]
        # ReLU
        conv1_float = np.maximum(conv1_float, 0)
        # Pool (on float)
        pool1_float = self._maxpool2x2_float(conv1_float, 6, 28, 28)
        # Quantize
        pool1_q = quantize_to_int8(pool1_float, self.conv1_out_scale)
        
        if verbose:
            print(f"Pool1: min={pool1_q.min()}, max={pool1_q.max()}, "
                  f"float_range=[{pool1_float.min():.3f}, {pool1_float.max():.3f}]")
        
        # Conv2
        conv2_acc = self._conv2d_int8(pool1_q, self.conv2_weight, 6, 14, 14, 16, 5, 5, 0, 1)
        conv2_float = conv2_acc.astype(np.float32) * (self.conv1_out_scale * self.conv2_weight_scale)
        for c in range(16):
            conv2_float[c] += self.conv2_bias[c]
        conv2_float = np.maximum(conv2_float, 0)
        pool2_float = self._maxpool2x2_float(conv2_float, 16, 10, 10)
        pool2_q = quantize_to_int8(pool2_float, self.conv2_out_scale)
        
        if verbose:
            print(f"Pool2: min={pool2_q.min()}, max={pool2_q.max()}")
        
        # Flatten
        flat_q = pool2_q.flatten()
        
        # FC1
        fc1_acc = self._matmul_int8(flat_q, self.fc1_weight, 1, 120, 400)
        fc1_float = fc1_acc.astype(np.float32) * (self.conv2_out_scale * self.fc1_weight_scale)
        fc1_float = fc1_float.flatten() + self.fc1_bias
        fc1_float = np.maximum(fc1_float, 0)
        fc1_q = quantize_to_int8(fc1_float, self.fc1_out_scale)
        
        if verbose:
            print(f"FC1: min={fc1_q.min()}, max={fc1_q.max()}")
        
        # FC2
        fc2_acc = self._matmul_int8(fc1_q, self.fc2_weight, 1, 84, 120)
        fc2_float = fc2_acc.astype(np.float32) * (self.fc1_out_scale * self.fc2_weight_scale)
        fc2_float = fc2_float.flatten() + self.fc2_bias
        fc2_float = np.maximum(fc2_float, 0)
        fc2_q = quantize_to_int8(fc2_float, self.fc2_out_scale)
        
        if verbose:
            print(f"FC2: min={fc2_q.min()}, max={fc2_q.max()}")
        
        # FC3 (no ReLU)
        fc3_acc = self._matmul_int8(fc2_q, self.fc3_weight, 1, 10, 84)
        fc3_float = fc3_acc.astype(np.float32) * (self.fc2_out_scale * self.fc3_weight_scale)
        fc3_float = fc3_float.flatten() + self.fc3_bias
        
        if verbose:
            print(f"FC3 output: {fc3_float}")
        
        return fc3_float, np.argmax(fc3_float)
    
    def _conv2d_int8(self, inp, weight, in_c, in_h, in_w, out_c, kh, kw, pad, stride):
        """Conv2D with int8 inputs, int32 accumulator."""
        out_h = (in_h + 2 * pad - kh) // stride + 1
        out_w = (in_w + 2 * pad - kw) // stride + 1
        
        out = np.zeros((out_c, out_h, out_w), dtype=np.int32)
        
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0
                    for ic in range(in_c):
                        for ky in range(kh):
                            for kx in range(kw):
                                ih = oh * stride - pad + ky
                                iw = ow * stride - pad + kx
                                
                                if 0 <= ih < in_h and 0 <= iw < in_w:
                                    acc += int(inp[ic, ih, iw]) * int(weight[oc, ic, ky, kx])
                    out[oc, oh, ow] = acc
        
        return out
    
    def _maxpool2x2_float(self, x, c, h, w):
        """Max pooling on float."""
        oh, ow = h // 2, w // 2
        out = np.zeros((c, oh, ow), dtype=np.float32)
        
        for ch in range(c):
            for y in range(oh):
                for xx in range(ow):
                    out[ch, y, xx] = max(
                        x[ch, y*2, xx*2], x[ch, y*2, xx*2+1],
                        x[ch, y*2+1, xx*2], x[ch, y*2+1, xx*2+1]
                    )
        
        return out
    
    def _matmul_int8(self, a, weight, m, n, k):
        """Matrix mul with int8, int32 accumulator. a[k], weight[n,k] -> out[n]"""
        out = np.zeros((m, n), dtype=np.int32)
        a = a.flatten()
        
        for i in range(m):
            for j in range(n):
                acc = 0
                for kk in range(k):
                    acc += int(a[i * k + kk]) * int(weight[j, kk])
                out[i, j] = acc
        
        return out

    def export_for_c(self, output_path):
        """Export weights in C format."""
        with open(output_path, 'w') as f:
            f.write("// LeNet-5 quantized weights (simple PTQ)\n")
            f.write("#ifndef __SIMPLE_QUANT_H__\n")
            f.write("#define __SIMPLE_QUANT_H__\n\n")
            f.write("#include <stdint.h>\n\n")
            
            # Scales as float (will be converted to fixed point in C)
            f.write("// Scales (for reference - actual C code uses fixed point)\n")
            f.write(f"#define INPUT_SCALE {self.input_scale}f\n")
            f.write(f"#define CONV1_W_SCALE {self.conv1_weight_scale}f\n")
            f.write(f"#define CONV1_OUT_SCALE {self.conv1_out_scale}f\n")
            f.write(f"#define CONV2_W_SCALE {self.conv2_weight_scale}f\n")
            f.write(f"#define CONV2_OUT_SCALE {self.conv2_out_scale}f\n")
            f.write(f"#define FC1_W_SCALE {self.fc1_weight_scale}f\n")
            f.write(f"#define FC1_OUT_SCALE {self.fc1_out_scale}f\n")
            f.write(f"#define FC2_W_SCALE {self.fc2_weight_scale}f\n")
            f.write(f"#define FC2_OUT_SCALE {self.fc2_out_scale}f\n")
            f.write(f"#define FC3_W_SCALE {self.fc3_weight_scale}f\n")
            f.write(f"#define FC3_OUT_SCALE {self.fc3_out_scale}f\n\n")
            
            # Compute scale ratios for requantization
            # output = (acc * in_scale * w_scale) / out_scale
            # = acc * (in_scale * w_scale / out_scale)
            # = acc * ratio
            
            conv1_ratio = (self.input_scale * self.conv1_weight_scale) / self.conv1_out_scale
            conv2_ratio = (self.conv1_out_scale * self.conv2_weight_scale) / self.conv2_out_scale
            fc1_ratio = (self.conv2_out_scale * self.fc1_weight_scale) / self.fc1_out_scale
            fc2_ratio = (self.fc1_out_scale * self.fc2_weight_scale) / self.fc2_out_scale
            
            # Convert to fixed point: ratio * 2^16
            f.write("// Requantization multipliers (Q16 fixed point)\n")
            f.write(f"#define CONV1_REQUANT_MULT {int(conv1_ratio * 65536)}\n")
            f.write(f"#define CONV2_REQUANT_MULT {int(conv2_ratio * 65536)}\n")
            f.write(f"#define FC1_REQUANT_MULT {int(fc1_ratio * 65536)}\n")
            f.write(f"#define FC2_REQUANT_MULT {int(fc2_ratio * 65536)}\n")
            f.write("#define REQUANT_SHIFT 16\n\n")
            
            # Quantized biases (scale to match accumulator)
            # bias_q = bias / (in_scale * w_scale)
            conv1_bias_q = np.round(self.conv1_bias / (self.input_scale * self.conv1_weight_scale)).astype(np.int32)
            conv2_bias_q = np.round(self.conv2_bias / (self.conv1_out_scale * self.conv2_weight_scale)).astype(np.int32)
            fc1_bias_q = np.round(self.fc1_bias / (self.conv2_out_scale * self.fc1_weight_scale)).astype(np.int32)
            fc2_bias_q = np.round(self.fc2_bias / (self.fc1_out_scale * self.fc2_weight_scale)).astype(np.int32)
            fc3_bias_q = np.round(self.fc3_bias / (self.fc2_out_scale * self.fc3_weight_scale)).astype(np.int32)
            
            # Write weights
            self._write_i8_array(f, "conv1_weight", self.conv1_weight)
            self._write_i32_array(f, "conv1_bias", conv1_bias_q)
            self._write_i8_array(f, "conv2_weight", self.conv2_weight)
            self._write_i32_array(f, "conv2_bias", conv2_bias_q)
            
            # FC weights need to be transposed for NPU matmul
            # PyTorch: [out_features, in_features] = [n, k]
            # NPU: expects [k, n] for B matrix in A[m,k] @ B[k,n]
            self._write_i8_array(f, "fc1_weight", self.fc1_weight.T)  # [400, 120]
            self._write_i32_array(f, "fc1_bias", fc1_bias_q)
            self._write_i8_array(f, "fc2_weight", self.fc2_weight.T)  # [120, 84]
            self._write_i32_array(f, "fc2_bias", fc2_bias_q)
            self._write_i8_array(f, "fc3_weight", self.fc3_weight.T)  # [84, 10]
            self._write_i32_array(f, "fc3_bias", fc3_bias_q)
            
            f.write("#endif // __SIMPLE_QUANT_H__\n")
        
        print(f"Exported to {output_path}")
    
    def _write_i8_array(self, f, name, arr):
        flat = arr.flatten()
        f.write(f"// {name}: shape={list(arr.shape)}\n")
        f.write(f"static const int8_t sq_{name}[{len(flat)}] = {{\n")
        for i in range(0, len(flat), 16):
            row = flat[i:i+16]
            f.write("    " + ", ".join(f"{v:4d}" for v in row) + ",\n")
        f.write("};\n\n")
    
    def _write_i32_array(self, f, name, arr):
        flat = arr.flatten()
        f.write(f"// {name}: shape={list(arr.shape)}\n")
        f.write(f"static const int32_t sq_{name}[{len(flat)}] = {{\n")
        for i in range(0, len(flat), 8):
            row = flat[i:i+8]
            f.write("    " + ", ".join(f"{v}" for v in row) + ",\n")
        f.write("};\n\n")


def main():
    # Load model
    model = LeNet5()
    model_path = os.path.join(OUTPUT_DIR, 'lenet5.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load calibration data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    calibration_data = [test_dataset[i] for i in range(100)]
    
    # Get stats
    print("\nCalibrating...")
    stats = get_calibration_stats(model, calibration_data)
    for name, s in stats.items():
        print(f"  {name}: [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Create quantized model
    print("\nCreating quantized model...")
    qmodel = SimpleQuantModel(model, stats)
    
    # Export
    qmodel.export_for_c(os.path.join(OUTPUT_DIR, 'simple_quant.h'))
    
    # Verify
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
                q_out, q_pred = qmodel.infer(x.squeeze(), verbose=(digit == 0))
                
                status = "PASS" if q_pred == digit else "FAIL"
                print(f"Digit {digit}: float={float_pred}, quant={q_pred} -> {status}")
                
                if q_pred == digit:
                    correct += 1
                break
    
    print(f"\nAccuracy: {correct}/{total} = {correct*100//total}%")


if __name__ == "__main__":
    main()
