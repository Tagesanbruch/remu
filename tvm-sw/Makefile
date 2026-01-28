# TVM-SW Makefile
# Build AI applications for REMU platform

# Configuration
PYTHON := python3
TVM_HOME ?= $(shell python3 -c "import tvm; print(tvm.__path__[0])" 2>/dev/null)
AM_HOME ?= $(abspath ../abstract-machine)

# Directories
BUILD_DIR := build
COMPILER_DIR := compiler
RUNTIME_DIR := runtime
TEST_DIR := tests

# Default target
.PHONY: all
all: compile

# Setup Python environment with uv
.PHONY: setup
setup:
	@echo "Setting up Python environment..."
	uv venv
	. .venv/bin/activate && uv pip install -r requirements.txt

# Compile a model
.PHONY: compile
compile: $(BUILD_DIR)
	@echo "Compiling model..."
	$(PYTHON) $(COMPILER_DIR)/compile_model.py \
		--model $(MODEL) \
		--output $(BUILD_DIR)

# Build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf .venv
	rm -rf __pycache__
	find . -name "*.pyc" -delete

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

# Help
.PHONY: help
help:
	@echo "TVM-SW Build System"
	@echo ""
	@echo "Targets:"
	@echo "  setup    - Setup Python virtual environment"
	@echo "  compile  - Compile ONNX model (MODEL=path/to/model.onnx)"
	@echo "  clean    - Clean build artifacts"
	@echo "  test     - Run tests"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL    - Path to ONNX model file"
	@echo "  AM_HOME  - Path to abstract-machine"
