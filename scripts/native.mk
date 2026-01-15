#***************************************************************************************
# Copyright (c) 2014-2022 Zihao Yu, Nanjing University
#
# REMU is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#**************************************************************************************/

-include $(REMU_HOME)/../Makefile

# NOTE: build.mk is included by Makefile directly, so we don't include it here
# to avoid duplicate targets warning.
# include $(REMU_HOME)/scripts/build.mk

include $(REMU_HOME)/tools/difftest.mk

# Some convenient rules

override ARGS ?= --log=$(BUILD_DIR)/remu-log.txt
override ARGS += $(ARGS_DIFF)


# Command to execute REMU
IMG ?=
ELF ?=
ifneq ($(strip $(ELF_OFFSET)),)
    override ARGS += --elf-offset=$(ELF_OFFSET)
endif

ifeq ($(strip $(ELF)),)
    override ARGS += 
else
    override ARGS += --elf=$(ELF)
endif

ifeq ($(strip $(BATCH)),)
    override ARGS += 
else
    override ARGS += --batch
endif

REMU_EXEC := $(BINARY) $(ARGS) $(IMG)

run-env: $(BINARY) $(DIFF_REF_SO) $(REMU_HOME)/src/generated/config.rs


run: run-env
	$(call git_commit, "run REMU")
	@echo "+ REMU $(ARGS) $(IMG)"
	@$(REMU_EXEC)

gdb: run-env
	$(call git_commit, "gdb REMU")
	gdb -s $(BINARY) --args $(REMU_EXEC)

batch: run-env
	$(call git_commit, "batch REMU")
	@$(REMU_EXEC) -b

clean-tools = $(dir $(shell find ./tools -maxdepth 2 -mindepth 2 -name "Makefile"))
$(clean-tools):
	-@$(MAKE) -s -C $@ clean
clean-tools: $(clean-tools)
clean-all: clean distclean clean-tools

.PHONY: run gdb run-env clean-tools clean-all $(clean-tools)
