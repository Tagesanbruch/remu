/**
 * Auto-generated LeNet inference code for REMU NPU
 */

#include <am.h>
#include <klib.h>
#include "npu.h"
#include "lenet_weights.h"
#include "lenet_test.h"

// Intermediate buffers
static int8_t input_buf[1 * 28 * 28];
static int32_t conv1_out[6 * 24 * 24];
static int8_t pool1_out[6 * 12 * 12];
static int32_t conv2_out[16 * 8 * 8];
static int8_t pool2_out[16 * 4 * 4];
static int32_t fc1_out[120];
static int8_t fc1_out_q[120];
static int32_t fc2_out[84];
static int8_t fc2_out_q[84];
static int32_t fc3_out[10];

// ReLU activation
static void relu_i32(int32_t *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

// Max pooling 2x2 with quantization
static void maxpool2x2_q(int32_t *in, int8_t *out, int c, int h, int w, int shift) {
    int oh = h / 2, ow = w / 2;
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t v0 = in[ch*h*w + (y*2)*w + (x*2)];
                int32_t v1 = in[ch*h*w + (y*2)*w + (x*2+1)];
                int32_t v2 = in[ch*h*w + (y*2+1)*w + (x*2)];
                int32_t v3 = in[ch*h*w + (y*2+1)*w + (x*2+1)];
                int32_t mx = v0;
                if (v1 > mx) mx = v1;
                if (v2 > mx) mx = v2;
                if (v3 > mx) mx = v3;
                int32_t q = mx >> shift;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out[ch*oh*ow + y*ow + x] = (int8_t)q;
            }
        }
    }
}

// Quantize i32 to i8
static void quantize(int32_t *in, int8_t *out, int n, int shift) {
    for (int i = 0; i < n; i++) {
        int32_t q = in[i] >> shift;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

// LeNet inference
int lenet_inference(const int8_t *input) {
    npu_reset();

    // Copy input
    memcpy(input_buf, input, 28 * 28);

    // Conv1: 1x28x28 -> 6x24x24, kernel=5x5
    npu_conv2d(input_buf, (int8_t*)conv1_weight, conv1_out,
               1, 1, 28, 28, 6, 5, 5, 0, 1, NPU_ACT_RELU);

    // Pool1: 6x24x24 -> 6x12x12
    maxpool2x2_q(conv1_out, pool1_out, 6, 24, 24, 8);

    // Conv2: 6x12x12 -> 16x8x8, kernel=5x5
    npu_conv2d(pool1_out, (int8_t*)conv2_weight, conv2_out,
               1, 6, 12, 12, 16, 5, 5, 0, 1, NPU_ACT_RELU);

    // Pool2: 16x8x8 -> 16x4x4
    maxpool2x2_q(conv2_out, pool2_out, 16, 8, 8, 8);

    // FC1: 256 -> 120
    npu_matmul(pool2_out, (int8_t*)fc1_weight, fc1_out, 1, 120, 256);
    relu_i32(fc1_out, 120);
    quantize(fc1_out, fc1_out_q, 120, 8);

    // FC2: 120 -> 84
    npu_matmul(fc1_out_q, (int8_t*)fc2_weight, fc2_out, 1, 84, 120);
    relu_i32(fc2_out, 84);
    quantize(fc2_out, fc2_out_q, 84, 8);

    // FC3: 84 -> 10
    npu_matmul(fc2_out_q, (int8_t*)fc3_weight, fc3_out, 1, 10, 84);

    // Find argmax
    int max_idx = 0;
    int32_t max_val = fc3_out[0];
    for (int i = 1; i < 10; i++) {
        if (fc3_out[i] > max_val) {
            max_val = fc3_out[i];
            max_idx = i;
        }
    }

    return max_idx;
}

void print_npu_stats(void) {
    printf("NPU Stats:\n");
    printf("  Cycles:      %u\n", npu_get_cycles());
    printf("  Mem Traffic: %u bytes\n", npu_get_mem_bytes());
    printf("  GEMM Ops:    %u\n", npu_get_gemm_count());
    printf("  Activations: %u\n", npu_get_act_count());
    printf("  DMA Xfers:   %u\n", npu_get_dma_count());
}

int main() {
    ioe_init();

    printf("=== LeNet ONNX Inference Test ===\n\n");

    int correct = 0;
    for (int i = 0; i < NUM_TEST_DIGITS; i++) {
        int pred = lenet_inference(test_digits[i]);
        int label = test_labels[i];
        printf("Test %d: label=%d, pred=%d %s\n", 
               i, label, pred, (pred == label) ? "[OK]" : "[FAIL]");
        if (pred == label) correct++;
    }

    printf("\nAccuracy: %d/%d\n\n", correct, NUM_TEST_DIGITS);

    print_npu_stats();

    if (correct == NUM_TEST_DIGITS) {
        printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("\n=== SOME TESTS FAILED ===\n");
        return 1;
    }
}