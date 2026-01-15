.DEFAULT_GOAL = app

# Add necessary options if the target is a shared library
ifeq ($(SHARE),1)
SO = -so
CFLAGS  += -fPIC -Werror -fvisibility=hidden
LDFLAGS += -shared -fPIC
endif

WORK_DIR  = $(shell pwd)
BUILD_DIR = $(WORK_DIR)/build

INC_PATH := $(WORK_DIR)/include $(INC_PATH)
OBJ_DIR  = $(BUILD_DIR)/obj-$(NAME)$(SO)
BINARY   = $(BUILD_DIR)/$(NAME)$(SO)

# Compilation flags
ifeq ($(CC),clang)
CXX := clang++
else
CXX := g++
endif
LD := $(CXX)
INCLUDES = $(addprefix -I, $(INC_PATH))
CFLAGS  := -O2 -MMD -Wall -Werror $(INCLUDES) $(CFLAGS)
LDFLAGS := -O2 $(LDFLAGS)

OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o) $(CXXSRC:%.cc=$(OBJ_DIR)/%.o)

# Compilation patterns
$(OBJ_DIR)/%.o: %.c
	@echo + CC $<
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -c -o $@ $<
	$(call call_fixdep, $(@:.o=.d), $@)

$(OBJ_DIR)/%.o: %.cc
	@echo + CXX $<
	@mkdir -p $(dir $@)
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -c -o $@ $<
	$(call call_fixdep, $(@:.o=.d), $@)

# Depencies
-include $(OBJS:.o=.d)

# Some convenient rules

.PHONY: app clean count search

app: $(BINARY)

$(BINARY):: $(OBJS) $(ARCHIVES)
	@echo + LD $@
	@$(LD) -o $@ $(OBJS) $(LDFLAGS) $(ARCHIVES) $(LIBS) -lreadline

count:
	@echo "Counting lines of .c and .h files in src folder..."
	@find src -type f \( -name '*.c' -o -name '*.h' \) -exec cat {} \; | grep -v '^$$' > temp_file.txt
	@echo "Total lines count of .c and .h files: `cat temp_file.txt | wc -l`"
# 	@echo "PA0_lines : 3415"
	@rm temp_file.txt

clean:
	-rm -rf $(BUILD_DIR)

# search:
# 	@if [ -z "$(STRING)" ]; then \
# 	    echo "Please specify a search string. Usage: make search string=<string>"; \
# 	else \
# 	    echo "Searching for '$(STRING)' in src directory..."; \
# 	    grep -nHR --color=always "$(STRING)" src | \
# 	    awk -F: '{print "\033[0;32m" $$1 "\033[0m:\033[0;31m" $$2 "\033[0m:" $$3}'; \
# 	fi