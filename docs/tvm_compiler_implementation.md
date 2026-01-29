# TVM REMU NPU 编译器文档

## 概述

本项目实现了基于 **Apache TVM 0.12.0** 的 REMU NPU 编译器后端，能够将 ONNX 模型编译为 NPU 可执行的 C 代码。

## 编译流程

```
┌──────────────────────────────────────────────────────────────┐
│                      ONNX Model (.onnx)                       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│            TVM Relay Frontend (relay.frontend.from_onnx)      │
│   - 解析 ONNX 格式                                            │
│   - 转换为 Relay IR                                           │
│   - 提取模型参数                                              │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    TVM Optimization Passes                    │
│   - InferType: 类型推导                                       │
│   - FoldConstant: 常量折叠                                    │
│   - SimplifyInference: 简化推理图                             │
│   - FoldScaleAxis: 折叠 scale 轴                              │
│   - CanonicalizeOps: 规范化算子                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   Relay IR Analysis (ExprVisitor)             │
│   - 遍历计算图                                                │
│   - 提取算子信息 (Conv2D, ReLU, Pool, Dense, etc.)           │
│   - 记录算子属性 (kernel_size, stride, padding, etc.)        │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   Weight Quantization                         │
│   - 对称 INT8 量化: scale = abs_max / 127                    │
│   - 生成二进制权重文件 (.bin)                                │
│   - 生成权重头文件 (.h) with offset macros                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   NPU Code Generation                         │
│   - 生成 C 推理代码框架                                       │
│   - 记录层信息到 JSON                                         │
└──────────────────────────────────────────────────────────────┘
```

## 使用方法

### 1. Docker 环境设置

```bash
cd tvm-sw

# 构建 Docker 镜像 (首次)
make docker-build

# 验证 TVM 安装
make verify-tvm
```

### 2. 编译模型

```bash
# 编译 MobileNetV2
make compile-mobilenet

# 编译 ResNet50
make compile-resnet

# 编译自定义模型
make compile MODEL=path/to/model.onnx NAME=model_name
```

### 3. Docker 交互 Shell

```bash
make docker-shell
```

## 文件结构

```
tvm-sw/
├── docker/
│   ├── Dockerfile              # TVM Docker 环境定义
│   └── docker-compose.yml      # Docker Compose 配置
├── compiler/
│   ├── tvm_compiler.py         # 主 TVM 编译器 (当前版本)
│   ├── analyze_onnx.py         # ONNX 模型分析工具
│   ├── onnx_compiler.py        # 旧版解释器 (已弃用)
│   └── output/
│       ├── mobilenet_tvm/      # MobileNetV2 编译输出
│       │   ├── mobilenet_weights.bin
│       │   ├── mobilenet_weights.h
│       │   ├── mobilenet_inference.c
│       │   └── mobilenet_layers.json
│       └── resnet_tvm/         # ResNet50 编译输出
│           ├── resnet_weights.bin
│           ├── resnet_weights.h
│           ├── resnet_inference.c
│           └── resnet_layers.json
├── onnx/
│   └── image_classification/
│       ├── mobilenetv2-7.onnx
│       └── resnet50-v2-7.onnx
└── Makefile                    # 构建系统
```

## 编译结果

### MobileNetV2

| 项目 | 值 |
|------|-----|
| 权重大小 | 3,487,816 bytes (3.3 MB) |
| Conv2D 层 | 52 |
| Clip (ReLU6) | 35 |
| Add (残差) | 11 |
| GlobalAvgPool | 1 |
| Dense | 1 |

### ResNet50-v2

| 项目 | 值 |
|------|-----|
| 权重大小 | 25,595,064 bytes (24.4 MB) |
| Conv2D 层 | 53 |
| ReLU | 50 |
| Add (残差) | 170 |
| MaxPool | 1 |
| GlobalAvgPool | 1 |

## Docker 环境

### 基础镜像
- `python:3.11-slim-bookworm` (ARM64)

### 安装的包

| 包 | 版本 |
|----|------|
| apache-tvm | 0.12.0 |
| numpy | >=1.23, <2 |
| onnx | 1.15.0 |
| onnxruntime | >=1.16, <1.18 |
| scipy | >=1.10, <1.14 |
| ml_dtypes | >=0.3, <0.4 |

### 版本兼容性说明

- `apache-tvm 0.12.0` 要求 `numpy < 2`
- `onnx 1.15.0` 使用 `ml_dtypes 0.3.x`（无 float4/uint4）
- `onnx 1.16+` 需要 `ml_dtypes 0.4+`（含新数据类型）

## NPU 硬件配置

```python
@dataclass
class NPUConfig:
    feature_sram_size: int = 16 * 1024  # 16KB
    weight_sram_size: int = 16 * 1024   # 16KB
    output_sram_size: int = 16 * 1024   # 16KB
    gemm_m_max: int = 256
    gemm_n_max: int = 256
    gemm_k_max: int = 256
    flash_base: int = 0x30000000        # 权重存储基址
    mmio_base: int = 0x21000000         # NPU MMIO 基址
```

## 后续工作

### 已完成
- [x] TVM Relay 前端集成
- [x] TVM 优化 pass 应用
- [x] 权重提取和 INT8 量化
- [x] 层信息分析和 JSON 导出
- [x] Docker 环境配置

### 待实现
- [ ] 完整的 NPU API 调用生成
- [ ] 自动内存规划 (buffer allocation)
- [ ] 算子融合 (Conv+BN+ReLU)
- [ ] 分片策略自动生成
- [ ] TVM AutoScheduler 集成

---

*文档更新时间: 2026-01-29*
*TVM 版本: 0.12.0*
