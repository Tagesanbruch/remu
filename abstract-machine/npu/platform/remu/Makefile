# NPU Library Makefile for REMU Platform
#
# This Makefile builds the NPU library for the remu platform.
# It should be included from the AM platform makefile.

NPU_DIR := $(REMU_AM_HOME)/npu/platform/remu

# NPU source files
NPU_SRCS := $(NPU_DIR)/src/npu_hw.c \
            $(NPU_DIR)/src/npu_tiling.c \
            $(NPU_DIR)/src/npu_utils.c \
            $(NPU_DIR)/src/npu_kernels.c

# NPU include path
NPU_INC := $(NPU_DIR)/include

# Export for use in platform makefile
NPU_OBJS := $(NPU_SRCS:%.c=%.o)
