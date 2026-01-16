#***************************************************************************************
<<<<<<< HEAD
# Copyright (c) 2014-2022 Zihao Yu, Nanjing University
# REMU Makefile - adapted for Rust/Cargo build system
=======
# REMU Makefile - Rust-based RISC-V Emulator
>>>>>>> temp
#**************************************************************************************/

REMU_HOME ?= $(shell pwd)

<<<<<<< HEAD
# Default target
.DEFAULT_GOAL = app

# Build directory
BUILD_DIR = $(REMU_HOME)/build

# Cargo build settings
CARGO = cargo
CARGO_FLAGS = --release --no-default-features
BINARY = target/release/remu
NAME = remu

# Include rules for menuconfig (if exists, optional for now)
-include $(REMU_HOME)/scripts/config.mk

# Main targets
.PHONY: app run clean menuconfig

app: $(BUILD_DIR)/$(NAME)
	@echo "Build complete: $(BUILD_DIR)/$(NAME)"

$(BUILD_DIR)/$(NAME): FORCE
	@mkdir -p $(BUILD_DIR)
	@echo "Building REMU with Cargo..."
	@$(CARGO) build $(CARGO_FLAGS)
	@cp $(BINARY) $(BUILD_DIR)/$(NAME)
	@echo "Binary copied to $(BUILD_DIR)/$(NAME)"

run: $(BUILD_DIR)/$(NAME)
	@$(BUILD_DIR)/$(NAME) --batch

clean:
	@echo "Cleaning build artifacts..."
	@$(CARGO) clean
	@rm -rf $(BUILD_DIR)

menuconfig:
	@echo "menuconfig not yet implemented for REMU"
	@echo "Edit .config file directly or use 'cargo build --features=...'"

count:
	@echo "Counting lines of Rust source files..."
	@find src -name '*.rs' -exec cat {} \; | grep -v '^$$' | wc -l

# Force rebuild check
FORCE:

.PHONY: FORCE
=======
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
>>>>>>> temp
