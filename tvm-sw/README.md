# TVM-SW: AI Compiler Stack for REMU NPU

This directory contains the TVM-based software stack for deploying AI models on the REMU NPU simulator.

## Directory Structure

```
tvm-sw/
├── onnx/           # Pre-trained ONNX models
├── compiler/       # TVM compilation scripts
├── runtime/        # Lightweight C runtime for AM
├── tests/          # Test programs
└── Makefile        # Build system
```

## Requirements

- Python 3.8+ with uv package manager
- TVM (Apache TVM)
- ONNX Runtime (for model loading)

## Setup

```bash
cd tvm-sw
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### 1. Compile ONNX Model

```bash
python compiler/compile_model.py --model onnx/mobilenetv2-7.onnx --output build/
```

### 2. Generate C Code for AM

The compiler generates:
- `model.c` - Main inference code
- `model_params.bin` - Quantized weights (to be loaded from Flash)
- `model.h` - Header with inference API

### 3. Build AM Application

```bash
make ARCH=riscv32-remu run
```

## Models

| Model | Size | Notes |
|-------|------|-------|
| mobilenetv2-7.onnx | ~14MB | Good for testing |
| tinyyolov2-8.onnx | ~45MB | Object detection |
| resnet50-v2-7.onnx | ~98MB | Too large for basic tests |

## Notes

- Models need quantization (INT8) for NPU acceleration
- Large models require Flash storage support in REMU
