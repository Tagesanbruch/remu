# ONNX 模型分析与适配计划（tvm-sw/onnx）

本报告基于 `docs/onnx_models_analysis.json` 的解析结果生成，覆盖 `tvm-sw/onnx` 目录下所有 ONNX 模型。目标：**在不改动底层 runtime 接口的前提下**，规划软件栈侧的适配路径（算子实现/拼接、输入准备、推理与性能计数）。

## 1. 模型概览（结构/输入/输出/算子）

> 说明：输入/输出为模型 Graph 的可见张量（排除 initializer 权重）。维度中出现 `N`/`batch`/`None` 表示动态维度。

### 1.1 NLP
- **bart_tiny.onnx**
  - **Inputs**: `src_tokens[2,1024]`, `prev_output_tokens[2,1024]`, `target[2048]`
  - **Outputs**: `loss[]`
  - **Ops**: `MatMul, Transpose, Softmax, Gelu, Add, Div, Sqrt, ReduceMean, Dropout, Trilu, SoftmaxCrossEntropyLoss, ...`
  - **Notes**: 输出为 loss；需要 tokenizer 与训练目标 `target`，推理路径偏“训练式”图。

- **bert_toy_postprocessed.onnx**
  - **Inputs**: `input_ids[batch,seq]`, `segment_ids[batch,seq]`, `input_mask[batch,seq]`, `masked_lm_labels[batch,seq]`, `next_sentence_labels[batch]`
  - **Outputs**: `loss[]`
  - **Ops**: `MatMul, LayerNormalization, Softmax, Erf/Tanh(GELU), Add, Div, Mul, SparseSoftmaxCrossEntropy, ...`
  - **Notes**: 输出为 loss；需要 MLM/NSS 标签与掩码。

### 1.2 视觉分类
- **coreml_Resnet50_ImageNet-dq.onnx**
  - **Inputs**: `image[None,3,224,224]`
  - **Outputs**: `classLabel[None,1]`, `classLabelProbs[?]`
  - **Ops**: `DequantizeLinear, Conv, BatchNormalization, Relu, Add, AveragePool, Softmax, ArgMax, ZipMap, ArrayFeatureExtractor, ImageScaler`
  - **Notes**: CoreML 端导出图，包含后处理算子与量化/反量化。

- **image_classification/mobilenetv2-7.onnx**
  - **Inputs**: `input[batch,3,224,224]`
  - **Outputs**: `output[batch,1000]`
  - **Ops**: `Conv, Add, Clip, GlobalAveragePool, Reshape, Gemm`

- **image_classification/resnet50-v1-13.onnx**
  - **Inputs**: `data[1,3,224,224]`
  - **Outputs**: `resnetv17_dense0_fwd[1,1000]`
  - **Ops**: `Conv, BatchNormalization, Relu, Add, MaxPool, GlobalAveragePool, Flatten, Gemm`

- **image_classification/resnet50-v2-7.onnx**
  - **Inputs**: `data[N,3,224,224]`
  - **Outputs**: `resnetv24_dense0_fwd[N,1000]`
  - **Ops**: `Conv, BatchNormalization, Relu, Add, MaxPool, GlobalAveragePool, Reshape, Gemm`

### 1.3 视觉检测/小网络
- **lenet.onnx**
  - **Inputs**: `import/Placeholder:0[1,1,28,28]`
  - **Outputs**: `import/conv4last/BiasAdd:0[1,10,1,1]`
  - **Ops**: `Conv, Add, Relu, MaxPool, Reshape`

- **tinyyolov2-8.onnx**
  - **Inputs**: `image[None,3,416,416]`
  - **Outputs**: `grid[None,125,13,13]`
  - **Ops**: `Conv, BatchNormalization, LeakyRelu, MaxPool, Add, Mul`

## 2. 现有 NPU 能力与算子映射

**NPU 已支持**（当前 REMU NPU 驱动/硬件）：
- `Conv2D` (im2col + GEMM + tiling)
- `GEMM/MatMul` (tiling)
- `ReLU`
- `DMA` (feature/weight/output)

**可由软件栈实现（CPU/AM/klib）**：
- `Add, Mul, Div, Sub`
- `Reshape, Transpose, Flatten, Concat, Slice, Squeeze/Unsqueeze`
- `BatchNormalization`（推理模式：用预计算的 scale/shift）
- `Pooling`（MaxPool/GlobalAveragePool/AveragePool）
- `Softmax, LayerNorm, Gelu`
- `Clip, LeakyRelu`
- `ArgMax, ZipMap, ArrayFeatureExtractor`

> 适配策略：**Conv/GEMM 优先走 NPU**，其余算子走 CPU 参考实现，保持底层接口不变。

## 3. 适配与推理计划（按模型）

### 3.1 视觉分类（优先级高）
**A. ResNet50 v1/v2、MobileNetV2**
- **推理路径**：
  - Conv → BN → ReLU → (Add/Residual) → Pool → FC/Gemm
- **需要实现/复用的 CPU 算子**：BN、Add、GlobalAveragePool、Flatten/Reshape。
- **性能计数**：
  - NPU 侧：使用 `npu_get_cycles/mem_bytes/gemm_count/dma_count`。
  - CPU 侧：用 `AM_TIMER_UPTIME` 或 `rdcycle` 包装算子耗时。
- **输入准备**：
  - 参考 `tvm-sw/onnx/image_classification/run.py`。
  - 预处理：`RGB -> float32`,  减均值 `[123.68,116.78,103.94]`, `NHWC -> NCHW`。
  - **输入名**：`data` 或 `input`。

**B. coreml_Resnet50_ImageNet-dq**
- **差异**：包含 `ImageScaler/DequantizeLinear/ZipMap/ArrayFeatureExtractor`。
- **适配策略**：
  - 使用 CPU 实现前后处理算子（避免改 runtime）。
  - 可在软件层剥离 `ZipMap` 等后处理，仅保留 logits + ArgMax。
- **输入准备**：
  - 仍按 224×224 RGB，遵循 `ImageScaler` 中的 mean/scale。
  - 需解析 `classLabelProbs` 或 `ArgMax` 输出。

### 3.2 LeNet（中优先级）
- **推理路径**：Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → FC。
- **适配策略**：
  - 直接映射到现有 NPU conv/matmul + CPU pooling。
  - 可复用现有 PTQ/INT8 代码路径。
- **输入准备**：`1×1×28×28`，MNIST 灰度图（可复用当前 `test_images.h` 生成流程）。

### 3.3 TinyYOLOv2（中优先级）
- **推理路径**：Conv → BN → LeakyReLU → MaxPool (多层)。
- **适配策略**：
  - Conv 走 NPU；BN/LeakyReLU/Pool 走 CPU。
  - 输出 `grid[1,125,13,13]`，先做 raw 输出验证，再做后处理（bbox decode）作为扩展。
- **输入准备**：`1×3×416×416`，RGB，按模型训练惯例（通常 0-1 归一化或 0-255，需根据模型导出来源确认）。

### 3.4 BERT Toy / BART Tiny（低优先级）
- **主要算子**：MatMul、Softmax、LayerNorm、GELU、Dropout、Loss。
- **适配策略**：
  - MatMul/GEMM 走 NPU；其余全 CPU。
  - Dropout 在推理时可跳过（恒等）。
  - 由于输出为 `loss`，需要构造 `label` 与 `mask` 输入。
- **输入准备**：
  - 需 tokenizer + vocab，当前 repo 未提供，建议先用随机/固定 token 验证数值一致性。
  - 可改写模型或新增推理图（仅输出 logits）以避免 loss 依赖。

## 4. 输入准备与验证策略

### 4.1 基于 `run.py` 的模型
- 目录：`tvm-sw/onnx/image_classification/run.py`
- 建议：
  - 复用其预处理逻辑（mean/resize/NCHW）。
  - 若在 REMU 上运行，需要将输入数据固化为 `.h` 或 `.bin`，并记录输入 tensor 名称。

### 4.2 无脚本模型
- **lenet.onnx**：可复用 MNIST 数据（28×28），与已有 LeNet 测试一致。
- **tinyyolov2-8.onnx**：可用固定图片或随机输入做 smoke test；后续再做真实图片验证。
- **bert/bart**：先用固定 token 测试执行路径与输出稳定性，再完善 tokenizer 和数据集。

## 5. 适配落地步骤（计划）

1. **生成/固化输入**
   - 为每个模型生成 `input.bin` + `input_meta.json`（包含输入名/shape/dtype）。
   - 对有 `run.py` 的模型，优先复用其预处理。

2. **构建算子适配层**
   - 设计一个小型 `op_kernels.c`：Conv/Gemm 调 NPU；其余 CPU 实现。
   - 引入统一的 `tensor` 描述（shape/stride）。

3. **推理执行器**
   - 从 ONNX 解析到顺序执行（先用 Python 生成 C 代码/调度表）。
   - 每个算子插入性能计数：
     - NPU：读取 `npu_get_*` 差值
     - CPU：读取 `AM_TIMER_UPTIME` 或 `rdcycle`

4. **逐模型验证**
   - 优先跑：`lenet.onnx` → `resnet50-v1` → `mobilenetv2` → `tinyyolov2` → `coreml_resnet50` → `bert/bart`
   - 每个模型先做功能正确性（输出 shape/范围），再做精度验证（如 top1）。

5. **性能报告**
   - 每个模型输出：总耗时、NPU GEMM 次数、DMA 次数、CPU 算子耗时统计。

## 6. 备注与风险
- `bart_tiny`/`bert_toy` 的 loss-only 输出不利于推理评测，建议后续导出 inference-only 版本或在软件层剥离 loss。
- `coreml_Resnet50_ImageNet-dq` 包含 `ZipMap`/`ArrayFeatureExtractor`，可能需要在软件层裁剪后处理。
- 大模型算子规模可能再次触发 SRAM 限制，需要继续沿 **M/N 维度 tiling**。

---

生成数据源：`docs/onnx_models_analysis.json`
