# ONNX 解释器实现文档

## 概述

**重要说明**: 之前的实现（v1）是一个 **ONNX 模型解释器**，直接解析 ONNX 模型并生成 C 推理代码，而非真正的 TVM 编译器。

**当前实现（v2）**：现已实现基于 **TVM 0.12.0** 的真正编译器后端，通过 Docker 环境运行。

## 版本对比

### v1: ONNX 解释器 (已弃用)

位置: `tvm-sw/compiler/onnx_compiler.py`

```
ONNX Model
    ↓
onnx.load() (Python ONNX库)
    ↓
手动遍历 ONNX Graph
    ↓
按算子类型生成模板 C 代码
    ↓
C 推理代码 + 量化权重
```

**问题**:
- 不使用 TVM，只是解析 ONNX
- 无算子优化 pass
- 无自动调度
- 手动实现量化

### v2: TVM 编译器 (当前版本)

位置: `tvm-sw/compiler/tvm_compiler.py`

```
ONNX Model
    ↓
TVM Relay Frontend (relay.frontend.from_onnx)
    ↓
Relay IR (高级中间表示)
    ↓
TVM Optimization Passes:
  - InferType
  - FoldConstant
  - SimplifyInference
  - FoldScaleAxis
  - CanonicalizeOps
    ↓
Relay IR Analysis (ExprVisitor)
    ↓
Weight Quantization (INT8)
    ↓
NPU Code Generation
    ↓
C 推理代码 + 量化权重二进制
```

**特点**:
- 真正使用 TVM Relay 前端
- 应用 TVM 优化 pass
- 从 Relay IR 分析图结构
- 生成 NPU 硬件兼容代码
