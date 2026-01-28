# NPU 硬件扩展与 TVM 编译栈设计分析

## 2026-01-28

### 一、已实现的 NPU 功能

#### 1.1 NPU 硬件模型 (npu.rs)
- **GEMM 单元**: int8 × int8 → int32 矩阵乘法
- **DMA 引擎**: MM2S (DRAM→SRAM), S2MM (SRAM→DRAM)
- **3 个 16KB SRAM**: Feature / Weight / Output
- **激活单元**: ReLU, ReLU6 (在 Output SRAM 上原地操作)
- **量化单元**: int32 → int8 (可选)

#### 1.2 Register Map
| 寄存器 | 偏移 | 功能 |
|--------|------|------|
| CTRL | 0x00 | 控制 (bit0=reset) |
| STATUS | 0x04 | 状态 (bit0=busy, bit1=done) |
| DMA_* | 0x08-0x18 | DMA 配置 |
| GEMM_* | 0x20-0x2C | GEMM 参数 (M, N, K) |
| ACT_* | 0x50-0x5C | 激活函数配置 |
| QUANT_* | 0x60-0x6C | 量化配置 |
| PERF_* | 0x80-0x94 | 性能计数器 |

### 二、AM 层驱动 API

```c
// 基础操作
void npu_reset(void);
void npu_dma_load_feature(void *src, uint32_t len);
void npu_dma_load_weight(void *src, uint32_t len);
void npu_dma_store_output(void *dst, uint32_t len);
void npu_gemm(uint32_t m, uint32_t n, uint32_t k);
void npu_relu(uint32_t len);

// 高级 API
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);
void npu_conv2d(int8_t *input, int8_t *weight, int32_t *output,
                int batch, int in_c, int in_h, int in_w,
                int out_c, int kh, int kw, int pad, int stride,
                uint32_t act_type);

// 性能计数器
uint32_t npu_get_cycles(void);
uint32_t npu_get_mem_bytes(void);
uint32_t npu_get_gemm_count(void);
uint32_t npu_get_act_count(void);
uint32_t npu_get_dma_count(void);
```

### 三、Tiling 策略

**Tiling 在 AM 驱动层实现** (npu_conv2d 函数中):
- im2col 在 CPU 上执行
- GEMM 分块受限于 SRAM 大小 (16KB)
- 每个 tile: tile_m * tile_k ≤ 16KB, tile_k * tile_n ≤ 16KB, tile_m * tile_n * 4 ≤ 16KB

### 四、测试结果

#### LeNet-5 测试 (PASS)
```
Input:  1x28x28
Conv1:  6x24x24 (5x5) + ReLU
Pool1:  6x12x12 (2x2 max)
Conv2:  16x8x8 (5x5) + ReLU
Pool2:  16x4x4 (2x2 max)
FC1:    256 -> 120
FC2:    120 -> 84
FC3:    84 -> 10

NPU Performance:
- Active Cycles: 5462
- Memory Traffic: 87,426 bytes
- GEMM Operations: 4
- Activations: 2
- DMA Transfers: 15
```

### 五、TVM 编译器 (tvm-sw/compiler/compile_model.py)

功能:
1. 分析 ONNX 模型结构
2. 提取并量化权重 (float32 → int8)
3. 生成 C 代码调用 NPU API

用法:
```bash
cd tvm-sw/compiler
uv run python compile_model.py --model ../onnx/lenet.onnx --output build/
uv run python compile_model.py --model ../onnx/mobilenetv2-7.onnx --analyze-only
```

### 六、下一步计划

1. **大模型支持**: 为 MobileNetV2 等添加更多 Tiling
2. **Pooling 硬件加速**: 目前 MaxPool 在 CPU 执行
3. **Depthwise Conv**: MobileNet 特有的深度可分离卷积
4. **完整推理**: 端到端 ONNX 推理测试
