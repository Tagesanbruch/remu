# NPU 集成开发日志

## 2026-01-28

### 完成的工作

#### 1. NPU 硬件模拟重构

重新设计了 NPU 设备模型，支持完整的 DMA 和内部 SRAM：

**内存映射 (Base: 0x2100_0000):**
| 地址范围 | 大小 | 功能 |
|---------|------|------|
| 0x0000-0x00FF | 256B | 控制寄存器 |
| 0x1000-0x4FFF | 16KB | Feature SRAM |
| 0x5000-0x8FFF | 16KB | Weight SRAM |
| 0x9000-0xCFFF | 16KB | Output SRAM |

**寄存器定义:**
| 偏移 | 名称 | 功能 |
|------|------|------|
| 0x00 | CTRL | 控制/复位 |
| 0x04 | STATUS | 状态 (Busy/Done/Error) |
| 0x08 | DMA_SRC | DMA 源地址 (DRAM) |
| 0x0C | DMA_DST | DMA 目的地址 (DRAM) |
| 0x10 | DMA_LEN | DMA 传输长度 |
| 0x14 | DMA_DIR | DMA 方向 |
| 0x18 | DMA_CTRL | DMA 控制 (写1启动) |
| 0x20 | GEMM_M | 矩阵 M 维度 |
| 0x24 | GEMM_N | 矩阵 N 维度 |
| 0x28 | GEMM_K | 矩阵 K 维度 |
| 0x2C | GEMM_CTRL | GEMM 控制 (写1启动) |
| 0x80 | PERF_CYCLES | 性能计数: 活跃周期 |
| 0x84 | PERF_BYTES | 性能计数: 内存流量 |

**DMA 方向:**
- 0: MM2S -> Feature SRAM
- 1: MM2S -> Weight SRAM  
- 2: S2MM <- Output SRAM

**代码位置:** `src/device/npu.rs`

#### 2. Flash 设备添加

为模型权重存储添加了 16MB Flash 模拟：

- 基地址: 0x3000_0000
- 大小: 16MB
- 支持从文件预加载
- 只读存储映射

**代码位置:** `src/device/flash.rs`

#### 3. AM 驱动实现

在 Abstract-Machine 中实现了 NPU 和 Flash 驱动：

**NPU API (`npu.h`):**
```c
void npu_reset(void);
void npu_matmul(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k);
void npu_dma_load_feature(void *src, uint32_t len);
void npu_dma_load_weight(void *src, uint32_t len);
void npu_dma_store_output(void *dst, uint32_t len);
void npu_gemm(uint32_t m, uint32_t n, uint32_t k);
```

**Flash API (`flash.h`):**
```c
void flash_read(uint32_t offset, void *buf, uint32_t len);
void *flash_get_base(void);
```

#### 4. 测试程序

创建了两个 NPU 测试：

1. **matmul_dma.c** - 基础矩阵乘法测试
   - 4x4 矩阵乘法
   - 验证 DMA 传输和 GEMM 计算
   - ✅ 测试通过

2. **conv2d.c** - 卷积测试  
   - 使用 im2col + GEMM 方法
   - 8x8x1 输入, 3x3 卷积核, 2 输出通道
   - ✅ 测试通过

#### 5. TVM 软件栈框架

搭建了 `tvm-sw/` 目录结构：

```
tvm-sw/
├── README.md           # 文档
├── Makefile           # 构建系统
├── requirements.txt   # Python 依赖
├── compiler/          # TVM 编译脚本
│   └── compile_model.py
└── onnx/              # 预训练模型
    ├── mobilenetv2-7.onnx
    ├── tinyyolov2-8.onnx
    └── ...
```

#### 6. Git LFS 配置

添加 `.gitattributes` 配置大文件存储：
```
*.onnx filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
```

### 测试结果

```
=== NPU MatMul Test ===
Matrix dimensions: M=4, N=4, K=4
TEST PASSED!
NPU Cycles: 1
NPU Memory Traffic: 60 bytes

=== NPU Conv2D Test ===
Input: 8x8x1, Kernel: 3x3, Output: 6x6x2
CONV2D TEST PASSED!
NPU Cycles: 3
NPU Memory: 276 bytes
```

### 待办事项

1. **Im2Col 硬件加速** - 当前 im2col 在 CPU 执行，可考虑在 NPU 中添加专用单元
2. **INT8 量化支持** - TVM 编译器需要配置量化 Pass
3. **Flash 加载命令行参数** - 支持 `--flash=model.bin` 参数
4. **更大规模测试** - 测试 64x64 或更大矩阵
5. **TVM BYOC 集成** - 完成 TVM 后端代码生成

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        REMU 模拟器                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │   CPU   │  │  CLINT  │  │  PLIC   │  │     Memory      │ │
│  │ RV32IMA │  │ (Timer) │  │ (Intr)  │  │   0x8000_0000   │ │
│  └────┬────┘  └─────────┘  └─────────┘  └────────┬────────┘ │
│       │                                          │          │
│       │              System Bus                  │          │
│  ─────┴──────────────────────────────────────────┴────────  │
│       │                    │                     │          │
│  ┌────┴────┐          ┌────┴────┐          ┌────┴────┐     │
│  │   NPU   │          │  Flash  │          │  Serial │     │
│  │0x2100.. │          │0x3000.. │          │0xA000.. │     │
│  ├─────────┤          ├─────────┤          └─────────┘     │
│  │Feature  │          │  16MB   │                          │
│  │SRAM 16K │          │ Model   │                          │
│  ├─────────┤          │ Storage │                          │
│  │Weight   │          └─────────┘                          │
│  │SRAM 16K │                                               │
│  ├─────────┤                                               │
│  │Output   │                                               │
│  │SRAM 16K │                                               │
│  └─────────┘                                               │
└─────────────────────────────────────────────────────────────┘
```

### 性能模型说明

当前性能模型基于以下假设：
- 16x16 脉动阵列 (Systolic Array)
- GEMM 周期数 = M×N×K / 256
- 内存流量 = 输入大小 + 权重大小 + 输出大小

后续可添加：
- NoC 延迟建模
- 内存带宽限制
- Pipeline stall 模拟
