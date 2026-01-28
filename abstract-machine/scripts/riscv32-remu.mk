include $(AM_HOME)/scripts/isa/riscv.mk
include $(AM_HOME)/scripts/platform/remu.mk
CFLAGS  += -D ISA_H=\"riscv/riscv.h\"
COMMON_CFLAGS += -march=rv32im_zicsr -mabi=ilp32

AM_SRCS += riscv/remu/start.S \
           riscv/remu/cte.c \
           riscv/remu/trap.S \
           riscv/remu/vme.c
