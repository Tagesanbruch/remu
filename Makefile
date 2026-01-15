#***************************************************************************************
# REMU Makefile - Rust-based RISC-V Emulator
#**************************************************************************************/

REMU_HOME ?= $(shell pwd)

# Build configuration
BUILD_DIR := $(REMU_HOME)/build
BINARY := $(BUILD_DIR)/remu

# Include build script and native rules
include $(REMU_HOME)/scripts/build.mk
include $(REMU_HOME)/scripts/native.mk

# Local convenience targets
.PHONY: app clean menuconfig count

app: $(BINARY)

clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf $(BUILD_DIR)

menuconfig:
	@echo "menuconfig not fully implemented for REMU"
	@echo "Edit .config file directly or use Cargo features"

count:
	@echo "Counting Rust source lines..."
	@find src -name '*.rs' -exec cat {} \; | grep -v '^$$' | wc -l

# Note: run, gdb, batch targets are provided by scripts/native.mk
