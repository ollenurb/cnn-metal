#include "src/layers.hpp"
#include <iostream>
#include <cmath>

/* ============================================================
 * Helpers
 * ============================================================ */

static int tests_run    = 0;
static int tests_passed = 0;

static void check(bool condition, const char* label) {
    tests_run++;
    if (condition) {
        tests_passed++;
        std::cout << "  [PASS] " << label << "\n";
    } else {
        std::cout << "  [FAIL] " << label << "\n";
    }
}

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

/* print a CHW tensor slice (single channel) to stdout */
static void print_channel(const Tensor& t, uint32_t channel) {
    uint32_t h = t.shape[2];
    uint32_t w = t.shape[3];
    for (uint32_t r = 0; r < h; r++) {
        for (uint32_t c = 0; c < w; c++) {
            std::printf("%7.2f ", t.data[channel * h * w + r * w + c]);
        }
        std::printf("\n");
    }
}

/* fill every element of a tensor with a constant value */
static void fill(Tensor& t, float value) {
    uint32_t n = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    for (uint32_t i = 0; i < n; i++) t.data[i] = value;
}

/* fill with a simple ramp 0, 1, 2, ... */
static void fill_ramp(Tensor& t) {
    uint32_t n = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    for (uint32_t i = 0; i < n; i++) t.data[i] = static_cast<float>(i);
}

/* fill with alternating +value / -value */
static void fill_alternating(Tensor& t, float value) {
    uint32_t n = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    for (uint32_t i = 0; i < n; i++) t.data[i] = (i % 2 == 0) ? value : -value;
}


/* ============================================================
 * Test: Conv2D output shape (stride = 1)
 *
 *  input  : 1x1x6x6
 *  kernel : 3x3, stride 1
 *  expected output shape: 1x1x4x4   (6-3)/1 + 1 = 4
 * ============================================================ */
static void test_conv_output_shape_stride1() {
    std::cout << "\n[Conv2D] output shape, k=3 stride=1\n";

    Conv2DLayer conv(3, 1);
    conv.set_input_shape({1, 1, 6, 6});
    auto out_shape = conv.output_size();

    check(out_shape[0] == 1, "batch dim preserved");
    check(out_shape[1] == 1, "channel dim preserved");
    check(out_shape[2] == 4, "height: (6-3)/1+1 = 4");
    check(out_shape[3] == 4, "width:  (6-3)/1+1 = 4");
}

/* ============================================================
 * Test: Conv2D output shape (stride = 2)
 *
 *  input  : 1x1x8x8
 *  kernel : 3x3, stride 2
 *  expected output shape: 1x1x3x3   (8-3)/2 + 1 = 3 (integer division)
 * ============================================================ */
static void test_conv_output_shape_stride2() {
    std::cout << "\n[Conv2D] output shape, k=3 stride=2\n";

    Conv2DLayer conv(3, 2);
    conv.set_input_shape({1, 1, 8, 8});
    auto out_shape = conv.output_size();

    check(out_shape[2] == 3, "height: (8-3)/2+1 = 3");
    check(out_shape[3] == 3, "width:  (8-3)/2+1 = 3");
}

/* ============================================================
 * Test: Conv2D identity kernel (all zeros weight, except centre = 1)
 *
 *  A 3x3 kernel with only the centre weight = 1 and all others = 0
 *  should produce the same values as the input (minus the border
 *  pixels which the kernel skips and leaves as 0).
 *
 *  input  : 1x1x6x6, filled with ramp 0..35
 *  kernel : 3x3, stride 1, weights = identity
 *  expected: output[r][c] == input[r][c] for interior pixels
 * ============================================================ */
static void test_conv_identity_kernel() {
    std::cout << "\n[Conv2D] identity kernel (centre weight = 1)\n";

    Conv2DLayer conv(3, 1);
    conv.set_input_shape({1, 1, 6, 6});

    /* set identity kernel: only centre element is 1 */
    Tensor& w = conv.get_weights();
    fill(w, 0.0f);
    w.data[4] = 1.0f; /* centre of 3x3 = index 4 */

    Tensor input({1, 1, 6, 6});
    fill_ramp(input);

    auto out_shape = conv.output_size();
    Tensor output(out_shape);
    fill(output, 0.0f);

    conv.forward(input, output);

    std::cout << "  input channel 0:\n";
    print_channel(input, 0);
    std::cout << "  output channel 0:\n";
    print_channel(output, 0);

    /*
     * With stride=1 and k=3, k_radius=1:
     * output thread index (ox, oy) maps to input centre (ox*1, oy*1).
     * Bounds check skips threads where in_x < 1 or in_x+1 >= 6,
     * i.e. output positions 0 and 3 (out_w=4, in positions 0 and 3+1=4... wait:
     *   ox=0 -> in_x=0 < k_radius=1  -> skipped
     *   ox=3 -> in_x=3, in_x+1=4 < 6 -> NOT skipped   (in_x+k_radius = 4 < 6)
     *
     * So only ox=0 and oy=0 rows/cols are skipped (left/top border).
     * Output positions (1..3, 1..3) should equal input at the same position.
     */
    bool interior_ok = true;
    uint32_t out_h = out_shape[2], out_w = out_shape[3];
    uint32_t in_w  = 6;
    for (uint32_t oy = 1; oy < out_h; oy++) {
        for (uint32_t ox = 1; ox < out_w; ox++) {
            float got      = output.data[oy * out_w + ox];
            float expected = input.data[oy * in_w + ox]; /* stride=1, same coords */
            if (!approx_eq(got, expected)) {
                std::printf("  mismatch at (%u,%u): got %.2f expected %.2f\n", oy, ox, got, expected);
                interior_ok = false;
            }
        }
    }
    check(interior_ok, "interior output matches input (identity kernel)");
}

/* ============================================================
 * Test: Conv2D uniform kernel (all weights = 1) on constant input
 *
 *  input  : 1x1x6x6, all ones
 *  kernel : 3x3 all-ones, stride 1
 *  expected interior output: 9.0  (sum of 3x3 patch of ones)
 * ============================================================ */
static void test_conv_sum_kernel() {
    std::cout << "\n[Conv2D] all-ones kernel on all-ones input\n";

    Conv2DLayer conv(3, 1);
    conv.set_input_shape({1, 1, 6, 6});

    Tensor& w = conv.get_weights();
    fill(w, 1.0f);

    Tensor input({1, 1, 6, 6});
    fill(input, 1.0f);

    auto out_shape = conv.output_size();
    Tensor output(out_shape);
    fill(output, 0.0f);

    conv.forward(input, output);

    std::cout << "  output channel 0:\n";
    print_channel(output, 0);

    /* all interior output positions should be 9 */
    uint32_t out_h = out_shape[2], out_w = out_shape[3];
    bool ok = true;
    for (uint32_t oy = 1; oy < out_h; oy++) {
        for (uint32_t ox = 1; ox < out_w; ox++) {
            if (!approx_eq(output.data[oy * out_w + ox], 9.0f)) {
                ok = false;
            }
        }
    }
    check(ok, "interior output == 9.0 (sum of 3x3 ones patch)");
}

/* ============================================================
 * Test: ReLU zeroes negative values, preserves positives
 *
 *  input  : 1x1x4x4, alternating +2 / -2
 *  expected: every negative becomes 0, every positive stays
 * ============================================================ */
static void test_relu() {
    std::cout << "\n[ReLU] zeroes negatives, preserves positives\n";

    ReLULayer relu;
    relu.set_input_shape({1, 1, 4, 4});

    Tensor input({1, 1, 4, 4});
    fill_alternating(input, 2.0f);

    Tensor output({1, 1, 4, 4});
    fill(output, 0.0f);

    relu.forward(input, output);

    std::cout << "  input channel 0:\n";
    print_channel(input, 0);
    std::cout << "  output channel 0:\n";
    print_channel(output, 0);

    bool ok = true;
    uint32_t n = 4 * 4;
    for (uint32_t i = 0; i < n; i++) {
        float expected = (i % 2 == 0) ? 2.0f : 0.0f;
        if (!approx_eq(output.data[i], expected)) {
            std::printf("  mismatch at %u: got %.2f expected %.2f\n", i, output.data[i], expected);
            ok = false;
        }
    }
    check(ok, "output matches expected ReLU values");
}

/* ============================================================
 * Test: ReLU output shape unchanged
 * ============================================================ */
static void test_relu_shape() {
    std::cout << "\n[ReLU] output shape preserved\n";

    ReLULayer relu;
    relu.set_input_shape({1, 3, 8, 8});
    auto out_shape = relu.output_size();

    check(out_shape[0] == 1, "batch preserved");
    check(out_shape[1] == 3, "channels preserved");
    check(out_shape[2] == 8, "height preserved");
    check(out_shape[3] == 8, "width preserved");
}

/* ============================================================
 * Test: MaxPool output shape (stride = 2)
 *
 *  input  : 1x1x8x8
 *  kernel : 3x3, stride 2
 *  expected output shape: 1x1x3x3   (8-3)/2 + 1 = 3
 * ============================================================ */
static void test_maxpool_output_shape() {
    std::cout << "\n[MaxPool] output shape, k=3 stride=2\n";

    MaxPoolLayer pool(3, 2);
    pool.set_input_shape({1, 1, 8, 8});
    auto out_shape = pool.output_size();

    check(out_shape[2] == 3, "height: (8-3)/2+1 = 3");
    check(out_shape[3] == 3, "width:  (8-3)/2+1 = 3");
}

/* ============================================================
 * Test: MaxPool picks maximum in each window
 *
 *  input  : 1x1x6x6, ramp 0..35
 *  kernel : 3x3, stride 1
 *  For an output at (oy, ox) the input centre is (oy, ox),
 *  the window covers rows [oy-1, oy+1] x cols [ox-1, ox+1].
 *  The maximum of that window equals the bottom-right element:
 *  input[(oy+1)*6 + (ox+1)].
 * ============================================================ */
static void test_maxpool_values() {
    std::cout << "\n[MaxPool] values match expected window maximum\n";

    MaxPoolLayer pool(3, 1);
    pool.set_input_shape({1, 1, 6, 6});

    Tensor input({1, 1, 6, 6});
    fill_ramp(input);

    auto out_shape = pool.output_size();
    Tensor output(out_shape);
    fill(output, 0.0f);

    pool.forward(input, output);

    std::cout << "  input channel 0:\n";
    print_channel(input, 0);
    std::cout << "  output channel 0:\n";
    print_channel(output, 0);

    /*
     * For a ramp input in row-major order, the maximum in any 3x3
     * window centred at (oy, ox) is the bottom-right element:
     * input[(oy+1)*6 + (ox+1)].
     * Border output positions (oy=0 or ox=0) are skipped by the kernel.
     */
    uint32_t out_h = out_shape[2], out_w = out_shape[3];
    bool ok = true;
    for (uint32_t oy = 1; oy < out_h; oy++) {
        for (uint32_t ox = 1; ox < out_w; ox++) {
            float got      = output.data[oy * out_w + ox];
            float expected = input.data[(oy + 1) * 6 + (ox + 1)];
            if (!approx_eq(got, expected)) {
                std::printf("  mismatch at (%u,%u): got %.2f expected %.2f\n", oy, ox, got, expected);
                ok = false;
            }
        }
    }
    check(ok, "interior output == bottom-right of each 3x3 window");
}

/* ============================================================
 * Test: MaxPool with stride=2 reduces spatial size and picks max
 *
 *  input  : 1x1x6x6, all elements = their index value (ramp)
 *  kernel : 3x3, stride 2
 *  output shape: (6-3)/2+1 = 2 x 2
 *
 *  output (oy=0, ox=0): centre at in(0,0) -> skipped (in_x=0 < k_radius=1)
 *  output (oy=0, ox=1): centre at in(0,2) -> skipped (in_y=0 < 1)
 *  output (oy=1, ox=0): centre at in(2,0) -> skipped (in_x=0 < 1)
 *  output (oy=1, ox=1): centre at in(2,2) -> window rows[1..3] cols[1..3]
 *                        max = input[3*6+3] = 21
 * ============================================================ */
static void test_maxpool_stride2() {
    std::cout << "\n[MaxPool] stride=2 values\n";

    MaxPoolLayer pool(3, 2);
    pool.set_input_shape({1, 1, 6, 6});

    Tensor input({1, 1, 6, 6});
    fill_ramp(input);

    auto out_shape = pool.output_size();
    Tensor output(out_shape);
    fill(output, 0.0f);

    pool.forward(input, output);

    std::cout << "  input channel 0:\n";
    print_channel(input, 0);
    std::cout << "  output channel 0:\n";
    print_channel(output, 0);

    /* only (oy=1, ox=1) is a valid non-border output position */
    uint32_t out_w = out_shape[3];
    float got = output.data[1 * out_w + 1];
    check(approx_eq(got, 21.0f), "output(1,1) == 21.0 (max of window rows[1..3] cols[1..3])");
}


/* ============================================================
 * Entry point
 * ============================================================ */
int main() {
    std::cout << "========================================\n";
    std::cout << " Forward pass tests\n";
    std::cout << "========================================\n";

    test_conv_output_shape_stride1();
    test_conv_output_shape_stride2();
    test_conv_identity_kernel();
    test_conv_sum_kernel();

    test_relu_shape();
    test_relu();

    test_maxpool_output_shape();
    test_maxpool_values();
    test_maxpool_stride2();

    std::cout << "\n========================================\n";
    std::cout << " Results: " << tests_passed << " / " << tests_run << " passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
