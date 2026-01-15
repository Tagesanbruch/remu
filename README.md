# REMU - Rust RISC-V Emulator

## 实现进度总结

### ✅ 已完成功能

**Phase 1: 核心基础**
- ✅ 项目重命名为 REMU
- ✅ 完整CLI参数系统 (--batch, --log, --diff, --port, --elf, image)
- ✅ NEMU风格日志系统（蓝色输出、文件记录）
- ✅ Makefile + Cargo 集成构建系统
- ✅ Git仓库初始化与提交

**Phase 2: 指令集实现**
- ✅ RV32I 基础指令集
- ✅ RV32M 乘除法扩展
- ✅ RV32A 原子操作扩展
- ✅ 完整寄存器状态管理
- ✅ 内存管理 (PMEM, MROM, SRAM,MMIO)

**Phase 3: 调试与追踪**
- ✅ 轻量级RISC-V反汇编器（~200行，支持RV32IMA）
- ✅ ITRACE 指令追踪（Ring Buffer + 反汇编输出）
- ✅ ABI寄存器名显示（ra, sp, a0...）
- ✅ 完整执行统计输出

**Phase 4: Abstract Machine集成**
- ✅ riscv32-remu platform完整支持
- ✅ IMG/ELF参数正确传递
- ✅ 通过全部35个cpu-tests测试

### ✅ 测试结果

```
test list [35 item(s)]: fact sub-longlong sum shift load-store max quick-sort 
leap-year mov-c unalign mersenne wanshu hello-str if-else switch add-longlong 
recursion pascal string div select-sort dummy crc32 bubble-sort goldbach prime 
bit add mul-longlong min3 fib shuixianhua matrix-mul to-lower-case movsx

Results: 35/35 PASS (100%)
```

### 📊 代码统计

- **Rust源代码**: ~3,500行
- **自定义反汇编器**: ~200行
- **支持指令**: RV32I + M + A (80+ instructions)
- **编译警告**: 0 warnings (已全部清除)

### 🔧 关键技术点

1. **反汇编器**: 无外部依赖，纯Rust实现
2. **泛型Ring Buffer**: 支持任意类型trace entry
3. **SDL2可选**: 通过feature flag控制，默认不编译
4. **zicsr支持**: 使用rv32g ISA确保CSR指令可用

### 下一步计划

**Phase 5: 调试器增强**
- [ ] Expression evaluator (表达式求值)
- [ ] Breakpoint support (断点系统)
- [ ] Watchpoint support (观察点)
- [ ] SDB交互式调试器

**Phase 6: 更多追踪**
- [ ] MTRACE (内存追踪)
- [ ] FTRACE (函数追踪，需ELF符号)
- [ ] DTRACE (设备追踪)

**Phase 7: ELF符号加载**
- [ ] ELF文件解析
- [ ] 符号表提取
- [ ] 函数名解析

**Phase 8: 设备扩展**
- [ ] Keyboard (i8042)
- [ ] VGA (framebuffer + SDL2)
- [ ] Audio
- [ ] Disk
- [ ] CLINT / PLIC

**Phase 9: Difftest**
- [ ] Spike/QEMU对比测试
- [ ] 寄存器状态同步
- [ ] 内存状态对比

## 快速开始

```bash
# 设置环境变量
export REMU_HOME=/path/to/remu

# 运行内置测试
cd $REMU_HOME
make run

# 运行AM程序
cd am-kernels/tests/cpu-tests
make ARCH=riscv32-remu ALL=dummy run

# 查看trace输出
cat build/remu-log.txt
```

## 项目结构

```
remu/
├── src/
│   ├── cpu/         # CPU核心
│   ├── isa/         # 指令集实现
│   │   └── riscv32/
│   │       ├── inst.rs    # 指令执行
│   │       ├── decode.rs  # 指令解码
│   │       └── disasm.rs  # 反汇编器
│   ├── memory/      # 内存管理
│   ├── device/      # 设备模拟
│   ├── monitor/     # 监视器
│   ├── engine/      # 执行引擎
│   └── utils/       # 工具函数
│       ├── log.rs        # 日志系统
│       ├── itrace.rs     # 指令追踪
│       └── ringbuffer.rs # Ring Buffer
├── scripts/         # 构建脚本
├── Makefile        # 主Makefile
└── Cargo.toml      # Rust项目配置
```

## 致谢

基于南京大学ICS课程的NEMU项目，使用Rust重新实现以提供更好的内存安全性和性能。
