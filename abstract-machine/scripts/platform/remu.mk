AM_SRCS := platform/remu/trm.c \
           platform/remu/ioe/ioe.c \
           platform/remu/ioe/timer.c \
           platform/remu/ioe/input.c \
           platform/remu/ioe/gpu.c \
           platform/remu/ioe/audio.c \
           platform/remu/ioe/disk.c \
           platform/remu/mpe.c

CFLAGS    += -fdata-sections -ffunction-sections
LDFLAGS   += -T $(AM_HOME)/scripts/linker.ld \
             --defsym=_pmem_start=0x80000000 --defsym=_entry_offset=0x0
LDFLAGS   += --gc-sections -e _start
REMUFLAGS += -l $(shell dirname $(IMAGE).elf)/remu-log.txt

CFLAGS += -DMAINARGS=\"$(mainargs)\"
CFLAGS += -I$(AM_HOME)/am/src/platform/remu/include
.PHONY: $(AM_HOME)/am/src/platform/remu/trm.c

image: $(IMAGE).elf
	@$(OBJDUMP) -d $(IMAGE).elf > $(IMAGE).txt
	@echo + OBJCOPY "->" $(IMAGE_REL).bin
	@$(OBJCOPY) -S --set-section-flags .bss=alloc,contents -O binary $(IMAGE).elf $(IMAGE).bin

run: image
	$(MAKE) -C $(REMU_HOME) ISA=$(ISA) run ARGS="$(REMUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf

gdb: image
	$(MAKE) -C $(REMU_HOME) ISA=$(ISA) gdb ARGS="$(REMUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf

batch: image
	$(MAKE) -C $(REMU_HOME) ISA=$(ISA) batch ARGS="$(REMUFLAGS)" IMG=$(IMAGE).bin ELF=$(IMAGE).elf