# NPU 待实现算子与 RTL 下沉评估与拆分计划

本计划基于当前 `abstract-machine/am/src/platform/remu/ioe/npu.c` 的实现能力，结合 ONNX 模型分析结果，评估**适合在 NPU RTL 端实现的算子**与**软件栈保留的算子**，并给出 `npu.c` 的结构拆分建议。

> 目标：优先将高计算密度、可流水化、固定模式的算子下沉到 RTL；保持底层接口稳定，软件栈负责调度与复杂控制。

## 1. 当前 NPU 侧已具备能力
- **GEMM/MatMul**（int8×int8→int32）
- **Conv2D**（im2col + GEMM + tiling）
- **ReLU**
- **DMA**（feature/weight/output）
- **性能计数器**（cycles/bytes/gemm/act/dma）

## 2. 适合 RTL 下沉的算子（优先级）

### P0：必须下沉（显著收益）
- **Conv2D / GEMM / MatMul**
  - 计算密度高，访存模式固定，可阵列化。
  - 已有软件版本可作为 RTL 参考。

### P1：强烈建议下沉
- **Depthwise Conv2D**（MobileNetV2 必需）
  - 规律性强，可复用 GEMM 或专用引擎。
- **Pooling（MaxPool / AvgPool / GlobalAveragePool）**
  - 访存可流式化；极大减少 CPU 参与。
- **Activation（ReLU / LeakyReLU / Clip）**
  - 简单逐元素，适合在输出侧流水化。

### P2：视需求下沉
- **BatchNormalization（推理模式）**
  - 可融合为逐元素 `y = x*scale + bias`，本质为逐元素算子。
- **Quantize / Requantize / Dequantize**
  - 乘加 + shift + clamp，适合硬化（尤其 int8 pipeline）。
- **Elementwise Add / Mul**
  - 残差网络高频使用；可融合到 Conv/GEMM 输出路径。

### P3：暂不建议 RTL 下沉
- **Softmax / LayerNorm / GELU**
  - 涉及 exp/div/sqrt，多尺度非线性，复杂度高。
- **Reshape / Transpose / Concat / Slice / Squeeze**
  - 更适合软件调度与内存视图管理。
- **ArgMax / ZipMap / ArrayFeatureExtractor**
  - 后处理为主，软件端即可。

## 3. 按模型的 RTL 价值点

- **ResNet/MobileNet**
  - Conv/BN/ReLU/Pool/Add 是主路径，RTL 化收益最大。
  - MobileNet 需补齐 **Depthwise Conv**。
- **TinyYOLOv2**
  - Conv/BN/LeakyReLU/Pool 关键路径；LeakyReLU 建议下沉。
- **LeNet**
  - Conv/Pool/ReLU/FC 规模小，但可验证 RTL 基础。
- **BERT/BART**
  - MatMul 密集，若做 NLP 推理，建议优先增强 GEMM 与量化路径。

## 4. `npu.c` 拆分建议（结构清晰）

当前 `npu.c` 同时包含：MMIO 访问、DMA/寄存器控制、算法级调度（im2col/tiling），建议拆分为多层结构。

### 4.1 建议目录结构
```
abstract-machine/
  npu/
    platform/
      remu/
        include/
          npu_hw.h          # 低层寄存器/常量定义
          npu_ops.h         # 高层 API 声明 (conv/matmul/pool...)
        src/
          npu_hw.c          # MMIO/DMA/基本控制
          npu_tiling.c      # tiling 策略与辅助函数
          npu_utils.c       # layout/transpose/quant 等工具
        kernels/
          matmul_kernel.c     # matmul/conv/pool/act 的 kernel 调度
          ...
```

### 4.2 与 AM 的集成方式
- `abstract-machine/am/src/platform/remu/ioe/npu.c`
  - 仅保留轻量封装，转调用 `abstract-machine/npu/platform/remu`。
- 通过 `Makefile` 增加 NPU 平台源码编译路径。

### 4.3 拆分目标
- **清晰职责**：
  - `npu_hw.c` 只做寄存器访问与 DMA 控制
  - `npu_kernels.c` 负责算子实现
  - `npu_tiling.c` 管理 M/N/K 的切分策略
- **便于 RTL 对齐**：将关键算子实现集中，方便比对/替换

## 5. RTL 下沉计划（阶段性）

### 阶段 A：算子稳定化（软件）
- 固化 Conv2D/MatMul/Pooling/Activation 的接口与行为
- 添加 shape/stride 限制说明

### 阶段 B：RTL 版本替换
- 逐步将 Conv/MatMul/Pooling/Act 迁移为硬件执行
- 保持 MMIO 寄存器语义不变

### 阶段 C：融合优化
- BN+ReLU、Conv+ReLU、Conv+Add 等融合
- 量化/激活流水化

## 6. 风险与注意事项
- **SRAM 容量**：当前 16KB 限制需要 tiling 策略长期保留
- **算子布局**：硬件侧需固定 `A[M,K]` 与 `B[K,N]` 约定
- **低层接口稳定**：本计划强调不改动 MMIO 接口，仅扩展上层

---

下一步：根据该计划逐步拆分 `npu.c` 并建立 `npu_ops.h` 对外 API；同步制定 RTL 侧算子优先级列表。