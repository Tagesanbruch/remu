NPC_HOME ?= $(REMU_AM_HOME)/../npc
AM_SRCS := riscv/npc/start.S \
           riscv/npc/trm.c \
           riscv/npc/ioe.c \
           riscv/npc/timer.c \
           riscv/npc/input.c \
           riscv/npc/cte.c \
           riscv/npc/trap.S \
           platform/dummy/vme.c \
           platform/dummy/mpe.c

CFLAGS    += -fdata-sections -ffunction-sections
LDFLAGS   += -T $(REMU_AM_HOME)/scripts/linker.ld \
						 --defsym=_pmem_start=0x80000000 --defsym=_entry_offset=0x0
LDFLAGS   += --gc-sections -e _start
CFLAGS += -DMAINARGS=\"$(mainargs)\"
CFLAGS += -I$(REMU_AM_HOME)/am/src/platform/npc/include
# NPCFLAGS += -l $(shell dirname $(IMAGE).elf)/npc-log.txt
# LOG = $(shell dirname $(IMAGE).elf)/npc-log.txt
.PHONY: $(REMU_AM_HOME)/am/src/riscv/npc/trm.c

image: $(IMAGE).elf
	@$(OBJDUMP) -d $(IMAGE).elf > $(IMAGE).txt
	@echo + OBJCOPY "->" $(IMAGE_REL).bin
	@$(OBJCOPY) -S --set-section-flags .bss=alloc,contents -O binary $(IMAGE).elf $(IMAGE).bin

run: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) run ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npc-log.txt

gdb: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) gdb ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npc-log.txt

batch: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) batch ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npc-log.txt

perf_test: image
	$(MAKE) -C $(NPC_HOME) ISA=$(ISA) perf_test ARGS="$(NPCFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf LOG=$(shell dirname $(IMAGE).elf)/npc-log.txt