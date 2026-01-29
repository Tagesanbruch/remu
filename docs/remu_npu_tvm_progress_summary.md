# REMU NPU / Runtime / TVM-SW 进展总结（2026-01-29）

## 1. 当前进展概览
- **TVM 编译链**：`tvm-sw/compiler/tvm_compiler.py` 已完成 MobileNetV2-7 的端到端代码生成，包含权重量化、bias 融合、激活后 requantize、Dense bias 支持。
- **生成产物（output/mobilenet）**：
  - `mobilenet_inference.c`（无 TODO，Dense bias add 后续 add 已跳过）
  - `mobilenet_weights.bin/.h`
  - `mobilenet_layers.json`
  - `test_mobilenet.c` + `Makefile`
  - `test_data/test_data.h`（内嵌测试输入与参考输出）
- **REMU 运行**：已在 REMU 上运行推理，执行完整，但**分类结果与期望不一致**（Top-1=0，期望=818）。

## 2. 当前问题与风险点
- **输出不一致（关键问题）**：
  - REMU 输出 top-1 = 0，与参考输出（818）不一致。
  - 末端 GEMM 日志显示分块执行（多次 `N=12` / `N=4`），与 `Dense(1280->1000)` 的单次矩阵乘不完全一致，需要核查 `npu_matmul` 内部分块实现与权重排布。
- **输出打印异常**：输出显示 `score: 1-`，疑似 `printf` 格式或目标环境输出异常，但不影响功能性判断。

## 3. 是否存在简化/未实现内容
- **显式 TODO/未实现标记**：在 `runtime/npu` 与 `src/npu` 中未发现显式 `TODO/FIXME/NOT IMPLEMENTED`。
- **潜在简化风险**：
  1. `npu_matmul` 的分块实现可能与权重布局/尺寸管理存在不一致（导致输出错）。
  2. 输出量化链：`Conv(int32) -> Bias -> ReLU6 -> Requantize(int8)` 流程已实现，但需确认 `scale/shift` 与 NPU 实现一致。
  3. 参考输出当前由 **ONNXRuntime** 生成（在 `tvm_compiler.py` 中自动生成），非 TVM 运行时输出。原因：TVM LLVM 后端对动态形状 `T.Any()` 报错，导致无法直接作为参考。

## 4. TVM 编译与处理结果（MobileNetV2-7）
- **Relay 优化结果**：206 ops
  - `nn.conv2d`: 52
  - `expand_dims`: 52
  - `add`: 63
  - `clip`: 35
  - `nn.global_avg_pool2d`: 1
  - `nn.dense`: 1
- **Bias 处理**：
  - Conv bias 已融合（52 个）
  - Dense bias 已融合（1 个）
  - 最后 `add`（Layer 205）确认是 Dense bias 分解，已跳过
- **权重尺寸**：`3,541,984 bytes`（int8 + bias int32）

## 5. REMU NPU Profile 结果摘要
来自 REMU 运行输出（MobileNetV2 测试）：
- `npu_active_cycles`: **1,042,534**
- `memory_traffic_bytes`: **63,870,604**
- `gemm_ops`: **976**
- `activation_ops`: **0**
- `dma_transfers`: **4,820**

观察：GEMM 次数与 Dense 期望并不匹配（Dense 应对应 1 次完整矩阵乘）；profile 中激活统计为 0 也不合理（存在 ReLU6/clip），可能表明 profile 统计路径或调用路径存在偏差。

## 6. 目录结构现状与建议（分析，不执行）
当前目录存在“模型生成产物”“验证测试”“历史脚本”混杂的情况，建议：

### 6.1 建议目录规划
- `tvm-sw/`
  - `compiler/`（仅放编译工具与模板）
  - `output/`（自动生成产物，按模型划分）
    - `mobilenet/`
      - `gen/`（生成的 C、权重、layers.json）
      - `test/`（测试程序 + test_data.h）
- `am-kernels/tests/npu-tests/`
  - 仅保留通用 NPU kernel tests（conv/gemm/pool）
  - **不再放模型 inference 生成文件**
- `runtime/` 与 `src/`：
  - 继续保持 runtime 与 NPU 实现分离
  - 将 NPU 算子 API 统一到 `runtime/npu/` 或 `src/npu/` 中

### 6.2 文件拆分建议
- `tvm_compiler.py`：拆分为
  - `onnx_parser.py`
  - `relay_graph.py`
  - `codegen/`（C/weights/test生成）
  - `cli.py`

## 7. 当前流程下无用文件/脚本（已移动计划）
以下文件为“旧流程或误拷贝”，建议统一移动到 `useless/`（分子目录存档）：
- `am-kernels/tests/npu-tests/tests/mobilenet_inference.c`
- `am-kernels/tests/npu-tests/tests/mobilenet_weights.bin`
- `am-kernels/tests/npu-tests/tests/mobilenet_weights.h`
- `am-kernels/tests/npu-tests/tests/test_mobilenet_verify.c`

## 8. 下一步调试建议
1. **核对 `npu_matmul` 实现**：确认分块策略与权重布局是否匹配 `N=1000`。
2. **对比 Dense 输入/输出**：在 `mobilenet_inference.c` 中插入 dump（或在 REMU 端添加调试）对比最后 1280 输入向量与输出 1000。
3. **验证 Requantize**：确认 `npu_requantize_shift` 与 scale/shift 的定义是否一致。

---
如需进一步拆分目录结构或将 debug 打印自动化，我可以继续补充脚本与结构化规划。
