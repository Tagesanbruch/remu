#ifndef NPU_OPS_H__
#define NPU_OPS_H__

#include <stdint.h>

#define NPU_CTRL_REG 0x00
#define NPU_SIZE_REG 0x04

// Offsets for MMIO buffers (as defined in TensorCore.scala)
#define NPU_WEIGHT_OFFSET 0x10
#define NPU_FEATURE_OFFSET 0x50
#define NPU_RESULT_OFFSET 0x90

void npu_init();
void npu_load_weights(int8_t *w);
void npu_load_features(int8_t *f);
void npu_start(int size); // size=16 usually
void npu_wait_done();
void npu_get_result(int32_t *res);

#endif
