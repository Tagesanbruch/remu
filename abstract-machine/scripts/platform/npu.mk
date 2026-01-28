NPU_HOME ?= $(REMU_AM_HOME)/../npc
AM_SRCS := riscv/npu/start.S \
           riscv/npu/trm.c \
           riscv/npu/ioe.c \
           riscv/npu/timer.c \
           riscv/npu/input.c \
           riscv/npu/cte.c \
           riscv/npu/trap.S \
           riscv/npu/npu_ops.c \
           platform/dummy/vme.c \
           platform/dummy/mpe.c

CFLAGS    += -fdata-sections -ffunction-sections
LDFLAGS   += -T $(REMU_AM_HOME)/scripts/linker.ld \
						 --defsym=_pmem_start=0x80000000 --defsym=_entry_offset=0x0
LDFLAGS   += --gc-sections -e _start
CFLAGS += -DMAINARGS=\"$(mainargs)\"
CFLAGS += -I$(REMU_AM_HOME)/am/src/platform/npu/include
# NPUFLAGS += -l $(shell dirname $(IMAGE).elf)/npu-log.txt
# LOG = $(shell dirname $(IMAGE).elf)/npu-log.txt
.PHONY: $(REMU_AM_HOME)/am/src/riscv/npu/trm.c

image: $(IMAGE).elf
	@$(OBJDUMP) -d $(IMAGE).elf > $(IMAGE).txt
	@echo + OBJCOPY "->" $(IMAGE_REL).bin
	@$(OBJCOPY) -S --set-section-flags .bss=alloc,contents -O binary $(IMAGE).elf $(IMAGE).bin

run: image
	$(MAKE) -C $(NPU_HOME) ISA=$(ISA) runnpu ARGS="$(NPUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npu-log.txt

gdb: image
	$(MAKE) -C $(NPU_HOME) ISA=$(ISA) gdb ARGS="$(NPUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npu-log.txt

batch: image
	$(MAKE) -C $(NPU_HOME) ISA=$(ISA) batch ARGS="$(NPUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npu-log.txt

perf_test: image
	$(MAKE) -C $(NPU_HOME) ISA=$(ISA) perf_test ARGS="$(NPUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npu-log.txt