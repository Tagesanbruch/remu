/**
 * NPU Driver for REMU Platform
 * 
 * This is a compatibility layer that includes the modular NPU implementation.
 * The actual implementation is split into:
 *   - npu_hw.c:      Low-level hardware access
 *   - npu_kernels.c: P0/P1/P2 operator implementations
 *   - npu_tiling.c:  Tiling utilities
 *   - npu_utils.c:   Data layout transformations
 * 
 * See: abstract-machine/npu/platform/remu/
 */

// Include the modular NPU implementation
#include "npu_ops.h"

// For backward compatibility, these functions are exported from npu_ops.h
// All implementations are in abstract-machine/npu/platform/remu/src/
