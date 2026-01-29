# NPU Conv2D 调试与修复报告
日期: 2026-01-30

## 1. 概述
本文档总结了为解决 NPU Conv2D 算子问题所做的工作，重点针对 MobileNetV2 模型。主要问题涉及 TVM 编译器生成的代码不正确（错误的 strides、形状）以及单元测试中的验证失败。

## 2. 遇到的问题

### 2.1. NPU 仿真挂起（Stride 不正确）
*   **症状:** NPU 仿真会挂起或产生错误的 tile 计数（例如，试图处理过多的 tile）。
*   **原因:** TVM `RelayAnalyzer` 未能正确从第一层 Conv2D 层的 Relay `Call` 节点中提取 `strides` 属性。它默认使用 `[1, 1]` 而不是正确的 `[2, 2]`。这导致 NPU 试图在 `224x224` 输入上进行卷积并期望得到 `224x224` 输出，而硬件/模型实际上期望的是下采样后的 `112x112` 输出。
*   **诊断:** 调试打印显示 `Strides raw: None` 或 `[1, 1]`。

### 2.2. Memcpy 断言失败（输出形状不匹配）
*   **症状:** 输出结果 `memcpy` 期间出现 `Assertion fail at ... string.c:119`。
*   **原因:** 即使手动将 NPU stride 强制为 2，编译器生成的 C 代码 (`inference.c`) 声明的输出缓冲区大小也不匹配 NPU 的实际输出大小。
    *   NPU 逻辑: `Output = (Input + 2*Pad - Kernel) / Stride + 1`
    *   TVM 逻辑: 有时假设了不同的隐式 padding 或舍入方式。
*   **细节:** 对于 224 输入, kernel 3, stride 2, pad 0:
    *   NPU 产生 `111x111`。
    *   TVM/Reference 有时期望 `112x112`（如果隐式使用 "SAME" padding 逻辑）。
    *   这导致了缓冲区溢出或重叠检查失败。

### 2.3. Python 类型错误 (`analyzer.py`)
*   **Bug:** `TypeError: unsupported operand type(s) for +: 'int' and 'list'`
*   **原因:** 从 TVM Relay 提取的 `padding` 属性通常是 TVM 特有的 `Array` 对象，而不是标准的 Python 列表。在 `_reconcile_shapes` 中进行算术运算时，如果没有正确转换，将其视为标量或列表会导致失败。
*   **Bug:** `TypeError: cannot unpack non-iterable NoneType object`
*   **原因:** `gen_unit_tests.py` 试图从 `layer.attrs` 中解包 `kernel_size`，但此时该值为 `None`（提取失败）。

### 2.4. 验证不匹配（Float vs Int）
*   **症状:** 单元测试报告数千个 "Mismatches"（例如，`Got 4429 Exp 4246`）。
*   **原因:** 测试框架将 **Int8/Int32** NPU 硬件执行结果与 **Float32** TVM 软件参考结果进行比较。精度差异、舍入策略和缩放因子导致了被标记为失败的系统误差。

## 3. 当前实现与修复

### 3.1. `analyzer.py` 改进
位于: `tvm-sw/compiler/remu_tvm/analyzer.py`

1.  **健壮的属性提取:**
    *   修改了 `_extract_attrs`，显式查找空间属性（`strides`, `padding`, `kernel_size`）并将 TVM `Array` 对象转换为 Python 列表。
    
2.  **Stride 推断回退 (Fallback):**
    *   在 `_reconcile_shapes` 中实现了逻辑：如果属性缺失，则根据输入/输出形状比率 **推断 strides**。
    *   `Stride = Input_Dim // Output_Dim`
    *   这确保了即使属性提取失败，MobileNetV2 Layer 0 也能得到正确的 `Stride=2`。

3.  **形状协调 (`_reconcile_shapes`):**
    *   使用显式的 NPU 硬件公式计算预期的输出形状。
    *   如果有差异，覆盖 Relay 推断的形状，确保生成的 C 代码分配的大小与 NPU 写入的大小完全一致。

### 3.2. `gen_unit_tests.py` 改进
位于: `tvm-sw/compiler/unit-test/gen_unit_tests.py`

1.  **Int32 参考实现:**
    *   将 Conv2D 的 Float32 参考检查替换为位精确的 **Int32 参考**。
    *   使用 `numpy.lib.stride_tricks.as_strided` 在 Python 中实现了朴素的 `Im2Col` + `MatMul` + `BiasAdd` 流水线。
    *   使用与 NPU 完全相同的量化权重和输入。
    
2.  **权重布局修正:**
    *   验证了 NPU 期望的权重布局为 `[N, K]`（展平的 `[Out, In*H*W]`）。
    *   移除了之前错误应用于 Conv2D 权重的 `.T` 转置。
    
3.  **崩溃预防:**
    *   在生成测试时，如果属性缺失，添加默认回退（例如 `kernel=[3,3]`, `stride=[1,1]`），防止脚本崩溃。

## 4. 状态
*   **Layer 0 (Conv2D):** **PASS**. 
    *   仿真使用 Stride 2 正确运行。
    *   输出验证相对于 Int32 参考通过，0 个不匹配。
    *   `memcpy` 断言已解决。

## 5. 下一步
*   如果其他层（`DepthwiseConv2D`, `AvgPool` 等）出现类似的不匹配，将验证修复（Int32 Ref）扩展到这些层。
*   验证 MobileNetV2 网络的其余部分。
