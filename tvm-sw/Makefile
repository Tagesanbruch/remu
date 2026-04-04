# TVM-SW Makefile
# Build AI applications for REMU platform using TVM
#
# Usage:
#   make docker-build      # Build Docker image with TVM
#   make compile-resnet    # Compile ResNet50 via Docker
#   make compile-mobilenet # Compile MobileNetV2 via Docker
#   make compile MODEL=path/to/model.onnx NAME=model_name

# Configuration
PYTHON := python3
AM_HOME ?= $(abspath ../abstract-machine)

# Docker configuration
DOCKER_IMAGE := remu-tvm:latest
DOCKER_ENV ?=
DOCKER_RUN := docker run --rm $(DOCKER_ENV) -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE)

# Directories
BUILD_DIR := build
COMPILER_DIR := compiler
OUTPUT_DIR := $(COMPILER_DIR)/output
ONNX_DIR := onnx
RUNTIME_DIR := runtime
TEST_DIR := tests

# Model paths (corrected paths)
RESNET_ONNX := $(ONNX_DIR)/image_classification/resnet50-v2-7.onnx
MOBILENET_ONNX := $(ONNX_DIR)/image_classification/mobilenetv2-7.onnx
LENET_ONNX := $(ONNX_DIR)/lenet.onnx

# Default target
.PHONY: all
all: help

.PHONY: help
help:
	@echo "TVM-SW: TVM Compiler for REMU NPU"
	@echo ""
	@echo "Setup:"
	@echo "  make docker-build     - Build Docker image with TVM"
	@echo "  make docker-shell     - Open interactive shell in Docker"
	@echo ""
	@echo "Compilation:"
	@echo "  make compile-resnet   - Compile ResNet50 for NPU"
	@echo "  make compile-mobilenet- Compile MobileNetV2 for NPU"
	@echo "  make compile-lenet    - Compile LeNet5 for NPU"
	@echo "  make compile MODEL=<path> NAME=<name> - Compile custom model"
	@echo ""
	@echo "Verification:"
	@echo "  make verify-tvm       - Verify TVM installation in Docker"
	@echo "  make verify-model MODEL=<path> IMAGE=<path> - Verify TVM model output"
	@echo "  make benchmark MODEL=<path> - Benchmark ONNX vs TVM"
	@echo ""
	@echo "Image Tools:"
	@echo "  make convert-images DIR=<path> - Convert images to NPU binary format"
	@echo ""
	@echo ""

# ============================================================================
# Docker Management
# ============================================================================

.PHONY: docker-build
docker-build:
	@echo "Building Docker image with TVM..."
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile docker/

.PHONY: docker-shell
docker-shell:
	@echo "Opening Docker shell..."
	docker run -it --rm -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) /bin/bash

.PHONY: verify-tvm
verify-tvm:
	@echo "Verifying TVM installation..."
	$(DOCKER_RUN) python3 -c "import tvm; print('TVM version:', tvm.__version__); \
		from tvm import relay; print('Relay frontend OK'); \
		print('TVM installation verified!')"

# ============================================================================
# Model Compilation via Docker
# ============================================================================

.PHONY: compile
compile:
ifndef MODEL
	$(error MODEL is not set. Usage: make compile MODEL=path/to/model.onnx NAME=model_name)
endif
ifndef NAME
	$(error NAME is not set. Usage: make compile MODEL=path/to/model.onnx NAME=model_name)
endif
	@echo "Compiling $(MODEL) -> $(NAME)"
	@mkdir -p $(OUTPUT_DIR)/$(NAME)
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/tvm_compiler.py \
		$(MODEL) $(OUTPUT_DIR)/$(NAME) $(NAME)

.PHONY: compile-resnet
compile-resnet: $(RESNET_ONNX)
	@echo "Compiling ResNet50..."
	@mkdir -p $(OUTPUT_DIR)/resnet50
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/tvm_compiler.py \
		$(RESNET_ONNX) $(OUTPUT_DIR)/resnet50 resnet50

.PHONY: compile-mobilenet
compile-mobilenet: $(MOBILENET_ONNX)
	@echo "Compiling MobileNetV2..."
	@mkdir -p $(OUTPUT_DIR)/mobilenet
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/tvm_compiler.py \
		$(MOBILENET_ONNX) $(OUTPUT_DIR)/mobilenet mobilenet

.PHONY: compile-lenet
compile-lenet: $(LENET_ONNX)
	@echo "Compiling LeNet..."
	@mkdir -p $(OUTPUT_DIR)/lenet
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/tvm_compiler.py \
		$(LENET_ONNX) $(OUTPUT_DIR)/lenet lenet

# ============================================================================
# Legacy: Local Python (requires TVM installed locally)
# ============================================================================

.PHONY: compile-local
compile-local: $(BUILD_DIR)
	@echo "Compiling model (local Python)..."
	$(PYTHON) $(COMPILER_DIR)/compile_model.py \
		--model $(MODEL) \
		--output $(BUILD_DIR)

# Build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# ============================================================================
# Model Verification
# ============================================================================

.PHONY: verify-model
verify-model:
ifndef MODEL
	$(error MODEL is not set. Usage: make verify-model MODEL=path/to/model.onnx IMAGE=path/to/image.jpg)
endif
ifndef IMAGE
	$(error IMAGE is not set. Usage: make verify-model MODEL=path/to/model.onnx IMAGE=path/to/image.jpg)
endif
	@echo "Verifying TVM model output..."
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/verify_tvm_model.py \
		--model $(MODEL) --image $(IMAGE)

.PHONY: verify-mobilenet
verify-mobilenet:
	@echo "Verifying MobileNetV2 with test images..."
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/verify_tvm_model.py \
		--model $(MOBILENET_ONNX) --image_dir $(ONNX_DIR)/image_classification/test_images/

.PHONY: verify-resnet
verify-resnet:
	@echo "Verifying ResNet50 with test images..."
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/verify_tvm_model.py \
		--model $(RESNET_ONNX) --image_dir $(ONNX_DIR)/image_classification/test_images/

.PHONY: benchmark
benchmark:
ifndef MODEL
	$(error MODEL is not set. Usage: make benchmark MODEL=path/to/model.onnx)
endif
	@echo "Benchmarking model..."
	$(DOCKER_RUN) python3 $(COMPILER_DIR)/verify_tvm_model.py \
		--model $(MODEL) --benchmark --runs 10

# ============================================================================
# Image Tools
# ============================================================================

.PHONY: convert-images
convert-images:
ifndef DIR
	$(error DIR is not set. Usage: make convert-images DIR=path/to/images OUTPUT=path/to/output)
endif
	@echo "Converting images to NPU binary format..."
	@mkdir -p $(OUTPUT_DIR)/images
	$(DOCKER_RUN) python3 tools/image_to_bin.py \
		--image_dir $(DIR) \
		--output $(OUTPUT_DIR)/images \
		--format int8 \
		--generate_asm \
		--generate_header \
		--combine

.PHONY: convert-test-images
convert-test-images:
	@echo "Converting test images..."
	@mkdir -p $(OUTPUT_DIR)/test_images
	$(DOCKER_RUN) python3 tools/image_to_bin.py \
		--image_dir $(ONNX_DIR)/image_classification/test_images \
		--output $(OUTPUT_DIR)/test_images \
		--format int8 \
		--generate_asm \
		--generate_header \
		--combine

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

.PHONY: clean-output
clean-output:
	rm -rf $(OUTPUT_DIR)

.PHONY: clean-docker
clean-docker:
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-output clean-docker
