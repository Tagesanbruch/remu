include $(REMU_AM_HOME)/scripts/isa/riscv.mk
include $(REMU_AM_HOME)/scripts/platform/npu.mk
CFLAGS  += -DISA_H=\"riscv/riscv.h\"
COMMON_CFLAGS += -march=rv32im -mabi=ilp32  # overwrite
# LDFLAGS       += -melf32lriscv                    # overwrite

AM_SRCS += riscv/npu/libgcc/div.S \
           riscv/npu/libgcc/muldi3.S \
           riscv/npu/libgcc/multi3.c \
           riscv/npu/libgcc/ashldi3.c \
           riscv/npu/libgcc/unused.c
