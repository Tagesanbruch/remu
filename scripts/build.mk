#***************************************************************************************
# REMU Build Script - Rust/Cargo build system
#**************************************************************************************/

.DEFAULT_GOAL = app

# Build directory setup
WORK_DIR  = $(shell pwd)
BUILD_DIR = $(WORK_DIR)/build
BINARY    = $(BUILD_DIR)/remu

# Rust/Cargo build configuration
CARGO ?= cargo
CARGO_BUILD_FLAGS ?= --release
CARGO_FEATURES = trace
ifeq ($(CONFIG_DEVICE),y)
CARGO_FEATURES += device
# Explicitly pass linker search paths for macOS/Homebrew
RUSTFLAGS += $(shell pkg-config --libs sdl2)
endif
export CARGO_FEATURES

# Compilation target
.PHONY: app

app: $(BINARY)

$(BINARY): FORCE
	@mkdir -p $(BUILD_DIR)
	@echo "Building REMU with Cargo..."
	@echo "CONFIG_DEVICE=$(CONFIG_DEVICE) CARGO_FEATURES=$(CARGO_FEATURES)"
	@RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) build $(CARGO_BUILD_FLAGS) --features "$(CARGO_FEATURES)"
	@cp target/release/remu $(BINARY)
	@echo "Binary ready: $(BINARY)"

# Force rebuild check
FORCE:
.PHONY: FORCE
