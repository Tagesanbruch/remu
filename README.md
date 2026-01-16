# REMU - Rust RISC-V Emulator

基于南京大学的[NEMU](https://github.com/NJU-ProjectN/NEMU)项目，使用Rust重新实现。可以成功boot RT-Thread/Linux。目前Boot Linux速度~70 MIPS.

![BootLinux](pictures/linux_boot.png)



## 开发环境

- Mac Mini M4

## 软件栈版本

- opensbi 1.8.1
- linux 5.15

## 支持的指令集架构

- RV32IMA_Zicsr_Zifencei

## Quick Start

1. 安装依赖

   macOS:

   ```bash
   brew install rust sdl2 dtc
   ```

   Linux (Ubuntu/Debian):

   ```bash
   sudo apt install cargo libsdl2-dev device-tree-compiler
   ```

2. 构建项目

   ```bash
   # 调试模式构建
   cargo build

   # Release 模式构建
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

3. 运行 Linux

   使用预编译好的 OpenSBI 和 Linux 镜像 (需要自行准备或参考 linux-sw 构建)：

   ```bash
   # 运行 Linux Payload
   ./target/release/remu --image /path/to/fw_payload.bin
   ```