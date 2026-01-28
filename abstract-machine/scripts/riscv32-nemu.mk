include $(AM_HOME)/scripts/isa/riscv.mk
include $(AM_HOME)/scripts/platform/nemu.mk
CFLAGS  += -DISA_H=\"riscv/riscv.h\"
NEMU_ARCH_FLAGS := -march=rv32im -mabi=ilp32
# LDFLAGS       += -melf32lriscv                     # overwrite

# Override variables initialized in isa/riscv.mk
COMMON_CFLAGS := $(NEMU_ARCH_FLAGS)
CFLAGS  := $(filter-out -march=% -mabi=%,$(CFLAGS)) $(NEMU_ARCH_FLAGS)
ASFLAGS := $(filter-out -march=% -mabi=%,$(ASFLAGS)) $(NEMU_ARCH_FLAGS)

AM_SRCS += riscv/nemu/start.S \
           riscv/nemu/cte.c \
           riscv/nemu/trap.S \
           riscv/nemu/vme.c
