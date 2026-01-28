AM_SRCS := riscv/ysyxsoc/start.S \
           riscv/ysyxsoc/trm.c \
           riscv/ysyxsoc/ioe.c \
           riscv/ysyxsoc/timer.c \
           riscv/ysyxsoc/input.c \
           riscv/ysyxsoc/cte.c \
           riscv/ysyxsoc/trap.S \
           platform/dummy/vme.c \
           platform/dummy/mpe.c

CFLAGS    += -fdata-sections -ffunction-sections
LDFLAGS   += -T $(AM_HOME)/am/src/riscv/ysyxsoc/linker.ld \
						 --defsym=_pmem_start=0x20000000 --defsym=_entry_offset=0x0
LDFLAGS   += --gc-sections -e _start
CFLAGS += -DMAINARGS=\"$(mainargs)\"
CFLAGS += -I$(AM_HOME)/am/src/platform/ysyxsoc/include
# NPCFLAGS += -l $(shell dirname $(IMAGE).elf)/ysyxsoc-log.txt
# LOG = $(shell dirname $(IMAGE).elf)/ysyxsoc-log.txt
.PHONY: $(AM_HOME)/am/src/riscv/ysyxsoc/trm.c

image: $(IMAGE).elf
	@$(OBJDUMP) -d $(IMAGE).elf > $(IMAGE).txt
	@echo + OBJCOPY "->" $(IMAGE_REL).bin
	@$(OBJCOPY) -S --set-section-flags .bss=alloc,contents -O binary $(IMAGE).elf $(IMAGE).bin

run: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) runsoc ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/ysyxsoc-log.txt

gdb: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) gdb ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/ysyxsoc-log.txt

perf_test: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) perf_test ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/ysyxsoc-log.txt