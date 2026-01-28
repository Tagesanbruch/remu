include $(REMU_AM_HOME)/scripts/isa/riscv.mk
include $(REMU_AM_HOME)/scripts/platform/npc.mk
CFLAGS  += -DISA_H=\"riscv/riscv.h\"

# Define the correct architecture with zicsr
NPC_ARCH_FLAGS := -march=rv32im -mabi=ilp32

# Override variables initialized in isa/riscv.mk
COMMON_CFLAGS := $(NPC_ARCH_FLAGS)
CFLAGS  := $(filter-out -march=% -mabi=%,$(CFLAGS)) $(NPC_ARCH_FLAGS)
ASFLAGS := $(filter-out -march=% -mabi=%,$(ASFLAGS)) $(NPC_ARCH_FLAGS)

AM_SRCS += riscv/npc/libgcc/div.S \
           riscv/npc/libgcc/muldi3.S \
           riscv/npc/libgcc/multi3.c \
           riscv/npc/libgcc/ashldi3.c \
           riscv/npc/libgcc/unused.c
