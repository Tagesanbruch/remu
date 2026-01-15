#***************************************************************************************
# Copyright (c) 2014-2022 Zihao Yu, Nanjing University
# REMU Makefile - adapted for Rust/Cargo build system
#**************************************************************************************/

REMU_HOME ?= $(shell pwd)

# Default target
.DEFAULT_GOAL = app

# Build directory
BUILD_DIR = $(REMU_HOME)/build

# Cargo build settings
CARGO = cargo
CARGO_FLAGS = --release --features=trace
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
