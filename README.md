<<<<<<< HEAD
# RustNEMU

NEMU (NJU Emulator) - A complete RISC-V32IM emulator written in Rust.

This is a complete rewrite of [NEMU](https://github.com/NJU-ProjectN/nemu) in Rust, maintaining 1:1 feature parity with configurable build options.

## Features

- **CPU**: RISC-V32IM + partial A extension support
- **Privilege Levels**: Machine, Supervisor, User modes
- **Memory**: Configurable memory size, MMU with SV32 paging
- **Devices**: Serial, Timer, Keyboard, VGA, Audio, Disk, CLINT, PLIC
- **Debugger**: Interactive debugger with breakpoints, watchpoints, expression evaluation
- **Tracing**: Instruction, memory, function, and device traces
- **Configuration**: Menuconfig-style build-time configuration system

## Building

### Prerequisites

Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Configuration

(Configuration system to be implemented - similar to Linux Kconfig)

### Build

```bash
cargo build --release
```

With specific features:
```bash
cargo build --release --features "trace,itrace,mtrace"
```

## Usage

```bash
# Run with image file
./target/release/rustnemu /path/to/image.bin

# With ELF symbols for function tracing
./target/release/rustnemu --elf /path/to/program.elf /path/to/image.bin

# Batch mode (non-interactive)
./target/release/rustnemu --batch /path/to/image.bin

# With logging
RUST_LOG=debug ./target/release/rustnemu /path/to/image.bin
```

## Testing

```bash
# Run built-in tests
cargo test

# Run with built-in image (to be implemented)
./target/release/rustnemu
```

## License

Mulan PSL v2 (same as original NEMU)
=======
# REMU - Rust RISC-V Emulator

[中文版](README_CN.md)

Re-implementation of the [NEMU](https://github.com/NJU-ProjectN/NEMU) project from Nanjing University (ICS course) using Rust. It can successfully boot RT-Thread/Linux. The current Linux boot speed is ~70 MIPS.

![BootLinux](pictures/linux_boot.png)

## Development Environment

- Mac Mini M4

## Software Versions

- OpenSBI 1.8.1
- Linux 5.15

## Supported ISA

- RV32IMA_Zicsr_Zifencei

## Quick Start

1. Install Dependencies

   macOS:

   ```bash
   brew install rust sdl2 dtc
   ```

   Linux (Ubuntu/Debian):

   ```bash
   sudo apt install cargo libsdl2-dev device-tree-compiler
   ```

2. Build Project

   ```bash
   # Debug build
   cargo build

   # Release build
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

3. Run Linux

   Use pre-compiled OpenSBI and Linux images (need to prepare yourself or refer to `linux-sw` for building):

   ```bash
   # Run Linux Payload
   ./target/release/remu --image /path/to/fw_payload.bin
   ```
>>>>>>> temp
