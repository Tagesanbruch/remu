CROSS_COMPILE := riscv64-linux-gnu-
ifneq ($(shell which riscv64-unknown-elf-gcc),)
CROSS_COMPILE := riscv64-unknown-elf-
endif
COMMON_CFLAGS := -fno-pic -march=rv32g -mabi=ilp32
CFLAGS        += $(COMMON_CFLAGS) -static
ASFLAGS       += $(COMMON_CFLAGS) -O2
LDFLAGS       += -melf32lriscv

# overwrite ARCH_H defined in $(AM_HOME)/Makefile
ARCH_H := arch/riscv.h
