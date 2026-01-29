# ONNX 解释器实现文档

## 概述

**重要说明**: 之前的实现并非真正的 TVM 编译器，而是一个 **ONNX 模型解释器**，它直接解析 ONNX 模型并生成 C 推理代码。本文档详细说明该解释器的实现方式。

## 架构对比

### 真正的 TVM 编译流程

```
ONNX Model
    ↓
TVM Frontend (onnx.from_onnx)
    ↓
Relay IR (高级中间表示)
    ↓
TVM Lowering (算子调度、内存规划)
    ↓
TIR (低级中间表示)
    ↓
Codegen (REMU NPU Backend)
    ↓
C/汇编代码 + 权重
```

### 实际实现的解释器流程

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

## 解释器实现细节

### 1. 模型加载与解析 (`onnx_compiler.py`)

使用 Python `onnx` 库加载模型，而非 TVM：

```python
import onnx
from onnx import numpy_helper

def load_onnx_model(path: str):
    model = onnx.load(path)
    graph = model.graph
    
    # 提取权重
    initializers = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        initializers[init.name] = arr
    
    # 遍历节点
    for node in graph.node:
        # 手动解析算子属性
        ...
```

### 2. 权重量化

实现了简单的对称量化，而非 TVM 的量化 pass：

```python
def quantize_weight(weight: np.ndarray, bits: int = 8):
    """手动实现的 INT8 对称量化"""
    abs_max = max(abs(weight.min()), abs(weight.max()))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    quantized = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
    return quantized, scale
```

### 3. C 代码生成

通过模板字符串手动生成 C 代码，而非 TVM Codegen：

```python
def generate_conv2d_code(layer, in_shape, out_shape):
    """手动生成 Conv2D 调用代码"""
    code = f"""
    // {layer.name}: Conv2D {in_shape} -> {out_shape}
    npu_conv2d_tiled(
        act_{layer.inputs[0]},
        weight_{layer.weight_name},
        act_{layer.outputs[0]},
        {batch}, {in_c}, {in_h}, {in_w},
        {out_c}, {kh}, {kw}, {pad}, {stride},
        NPU_ACT_RELU
    );
    """
    return code
```

### 4. 算子映射

手动将 ONNX 算子映射到 NPU API 调用：

| ONNX 算子 | 生成的 NPU 调用 |
|-----------|-----------------|
| Conv (group=1) | `npu_conv2d_tiled()` |
| Conv (group=C) | `npu_depthwise_conv2d()` |
| MaxPool | `npu_maxpool2d()` |
| GlobalAveragePool | `npu_global_avgpool2d()` |
| Gemm/MatMul | `npu_matmul_tiled()` |
| Relu | `npu_relu_tiled()` |
| Add | `npu_add()` |

## 文件清单

### 编译器文件

| 文件 | 说明 |
|------|------|
| `tvm-sw/compiler/onnx_compiler.py` | 主要的 ONNX 解释器 (673 行) |
| `tvm-sw/compiler/compile_resnet.py` | ResNet50 特化编译脚本 |
| `tvm-sw/compiler/compile_mobilenet.py` | MobileNetV2 特化编译脚本 |
| `tvm-sw/compiler/compile_lenet.py` | LeNet5 编译脚本 |

### 生成的输出

| 目录 | 内容 |
|------|------|
| `tvm-sw/compiler/output/resnet50_e2e/` | ResNet50 权重 (24MB) + 推理代码 |
| `tvm-sw/compiler/output/mobilenet_e2e/` | MobileNetV2 权重 (3.3MB) + 推理代码 |

### 测试文件

| 文件 | 说明 |
|------|------|
| `am-kernels/tests/npu-tests/tests/test_resnet_e2e.c` | ResNet 端到端测试 |
| `am-kernels/tests/npu-tests/tests/test_mobilenet_e2e.c` | MobileNet 端到端测试 |

## 局限性

1. **无优化 Pass**: 没有算子融合、内存优化
2. **无自动调度**: 没有 TVM AutoTVM/AutoScheduler 的性能调优
3. **固定量化**: 简单的对称量化，无校准数据支持
4. **无 Graph 优化**: 没有常量折叠、死代码消除
5. **手动分片**: Tiling 策略硬编码，非自动生成

## 测试结果

尽管是简化实现，端到端测试可以运行：

```
MobileNetV2:
- Initial Conv: 98 GEMM tiles
- Block 1: 49 GEMM tiles  
- NPU Profile: 91868 cycles, 4.78MB traffic

ResNet50:
- Initial Conv (7x7): 58 GEMM tiles
- NPU Profile: 373984 cycles, 1.5MB traffic
```

## 下一步：实现真正的 TVM 后端

需要实现的 TVM 组件：

1. **REMU NPU Target**: 定义 target 设备
2. **Relay to NPU**: 实现算子调度策略
3. **NPU Codegen**: 从 TIR 生成 C 代码
4. **Runtime**: TVM runtime 适配

---

*文档创建时间: 2026-01-29*
*状态: 解释器实现已弃用，需要真正的 TVM 后端*
