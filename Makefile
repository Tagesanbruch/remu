#***************************************************************************************
# REMU Makefile - Rust-based RISC-V Emulator
#**************************************************************************************/

REMU_HOME ?= $(shell pwd)

# Build configuration
BUILD_DIR := $(REMU_HOME)/build
BINARY := $(BUILD_DIR)/remu

# Include config rules (provides menuconfig, defines CONFIG_*)
include $(REMU_HOME)/scripts/config.mk

# Include build script and native rules
include $(REMU_HOME)/scripts/build.mk
include $(REMU_HOME)/scripts/native.mk

# Autoconf: Regenerate config.rs when .config changes
$(REMU_HOME)/src/generated/config.rs: $(REMU_HOME)/.config
	@echo "Regenerating config.rs..."
	@python3 $(REMU_HOME)/scripts/gen_config.py

# Ensure build depends on config.rs
$(BINARY): $(REMU_HOME)/src/generated/config.rs

# Local convenience targets
.PHONY: app clean count

app: $(BINARY)

clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf $(BUILD_DIR)

count:
	@echo "Counting Rust source lines..."
	@find src -name '*.rs' -exec cat {} \; | grep -v '^$$' | wc -l

# Note: run, gdb, batch targets are provided by scripts/native.mk
