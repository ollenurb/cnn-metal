#include <metal_stdlib>
using namespace metal;

/*
 * ===================================================================
 *                         FORWARD KERNELS
 * ===================================================================
 */

kernel void conv_3d(
    device const float* x       [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* result        [[buffer(2)]],
    constant uint& k            [[buffer(3)]],
    constant uint& stride       [[buffer(4)]],
    constant uint& width        [[buffer(5)]],
    constant uint& height       [[buffer(6)]],
    uint3 index                 [[thread_position_in_grid]],
    uint3 grid                  [[threads_per_grid]]
) {
    uint k_radius = k/2;
    uint index_x = index.x * stride;
    uint index_y = index.y * stride;

    /* bounds checking */
    if (
        index_x < k_radius ||
        index_x + k_radius >= width ||
        index_y < k_radius ||
        index_y + k_radius >= height
    ) return;

    /* accumulate result */
    float sum = 0;
    for (uint i = 0; i < k; i++) {
        for (uint j = 0; j < k; j++) {
            uint c = index_x - k_radius + j;
            uint r = index_y - k_radius + i;
            /* x[index.z][r][c] */
            sum += weights[i * k + j] * x[(index.z * width * height) + (r * width) + c];
        }
    }
    result[(index.z * grid.x * grid.y) + (index.y * grid.x) + index.x] = sum;
}

kernel void relu_3d(
    device const float* x       [[buffer(0)]],
    device float* result        [[buffer(1)]],
    uint3 index                 [[thread_position_in_grid]],
    uint3 grid                  [[threads_per_grid]]
) {
    float res = x[(index.z * grid.x * grid.y) + (index.y * grid.x) + index.x];
    if (res < 0) {
        res = 0;
    }
    result[(index.z * grid.x * grid.y) + (index.y * grid.x) + index.x] = res;
}

kernel void max_pool_3d(
    device const float* x       [[buffer(0)]],
    device float* result        [[buffer(1)]],
    constant uint& k            [[buffer(2)]],
    constant uint& stride       [[buffer(3)]],
    constant uint& width        [[buffer(4)]],
    constant uint& height       [[buffer(5)]],
    uint3 index                 [[thread_position_in_grid]],
    uint3 grid                  [[threads_per_grid]]
) {
    uint k_radius = k/2;
    uint index_x = index.x * stride;
    uint index_y = index.y * stride;
    if (index_x < k_radius || index_x + k_radius >= width || index_y < k_radius || index_y + k_radius >= height) return;

    float max = -INFINITY;
    for (uint i = 0; i < k; i++) {
        for (uint j = 0; j < k; j++) {
            uint c = index_x - k_radius + j;
            uint r = index_y - k_radius + i;

            /* update max */
            float cur = x[(index.z * width * height) + (r * width) + c];
            if (cur > max)  max = cur;
        }
    }
    result[(index.z * grid.x * grid.y) + (index.y * grid.x) + index.x] = max;
}


/*
 * ===================================================================
 *                         BACKWARD KERNELS
 * ===================================================================
 */

/*
 * relu_backward_3d
 * grad_input[i] = grad_output[i] * (x_pre_relu[i] > 0 ? 1.0 : 0.0)
 */
kernel void relu_backward_3d(
    device const float* x           [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float*       grad_input  [[buffer(2)]],
    uint3 index                     [[thread_position_in_grid]],
    uint3 grid                      [[threads_per_grid]]
) {
    uint idx    = (index.z * grid.x * grid.y) + (index.y * grid.x) + index.x;
    grad_input[idx] = grad_output[idx] * (x[idx] > 0.0f ? 1.0f : 0.0f);
}

/*
 * max_pool_backward_3d
 *
 * For each output position, re-find the argmax in the input window,
 * and write grad_output to that position in grad_input.
 * Since stride >= k, windows don't overlap, so no atomics needed.
 *
 * Buffers:
 *   x           - original input to forward max_pool
 *   grad_output - gradient from next layer (same shape as forward output)
 *   grad_input  - gradient w.r.t. x (same shape as x, should be zero-initialized)
 *   k, stride, width, height - same as forward
 *
 * Grid: same as forward output (grid.x * grid.y * grid.z)
 */
kernel void max_pool_backward_3d(
    device const float* x           [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float*       grad_input  [[buffer(2)]],
    constant uint& k                [[buffer(3)]],
    constant uint& stride           [[buffer(4)]],
    constant uint& width            [[buffer(5)]],
    constant uint& height           [[buffer(6)]],
    uint3 index                     [[thread_position_in_grid]],
    uint3 grid                      [[threads_per_grid]]
) {
    uint k_radius = k / 2;
    uint index_x = index.x * stride;
    uint index_y = index.y * stride;

    if (index_x < k_radius || index_x + k_radius >= width ||
        index_y < k_radius || index_y + k_radius >= height) return;

    /* find index of the maximum in the input window */
    float max_val = -INFINITY;
    uint max_r = 0;
    uint max_c = 0;
    for (uint i = 0; i < k; i++) {
        for (uint j = 0; j < k; j++) {
            uint c = index_x - k_radius + j;
            uint r = index_y - k_radius + i;
            float cur = x[(index.z * width * height) + (r * width) + c];
            if (cur > max_val) {
                max_val = cur;
                max_r = r;
                max_c = c;
            }
        }
    }

    /* route gradient only to the element with max value */
    uint out_idx = (index.z * grid.x * grid.y) + (index.y * grid.x) + index.x;
    grad_input[(index.z * width * height) + (max_r * width) + max_c] = grad_output[out_idx];
}


/*
 * conv_backward_input_3d
 *
 * Computes grad_input w.r.t. the input x of the forward conv_3d.
 *
 * Forward:
 *   out[z][oy][ox] = sum_{i,j} weights[i][j] * x[z][oy*stride - k_radius + i][ox*stride - k_radius + j]
 *
 * So each input position x[z][r][c] contributed to output positions where:
 *   r = oy*stride - k_radius + i  =>  i = r - oy*stride + k_radius
 *   c = ox*stride - k_radius + j  =>  j = c - ox*stride + k_radius
 *
 * grad_input[z][r][c] = sum over valid (oy, ox) of
 *     grad_output[z][oy][ox] * weights[r - oy*stride + k_radius][c - ox*stride + k_radius]
 *
 * Grid: dispatched over (width, height, channels) — i.e. the input shape.
 *
 * Buffers:
 *   grad_output  - gradient from next layer (shape: channels * out_h * out_w)
 *   weights      - conv kernel (k * k)
 *   grad_input   - output of this kernel (same shape as x)
 *   k, stride, width, height - same as forward
 *   out_w, out_h - spatial dimensions of grad_output / forward output
 */
kernel void conv_backward_input_3d(
    device const float* grad_output [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       grad_input  [[buffer(2)]],
    constant uint& k                [[buffer(3)]],
    constant uint& stride           [[buffer(4)]],
    constant uint& width            [[buffer(5)]],
    constant uint& height           [[buffer(6)]],
    constant uint& out_w            [[buffer(7)]],
    constant uint& out_h            [[buffer(8)]],
    uint3 index                     [[thread_position_in_grid]],
    uint3 grid                      [[threads_per_grid]]
) {
    /* index.x = c, index.y = r, index.z = channel */
    uint c = index.x;
    uint r = index.y;
    uint z = index.z;

    uint k_radius = k / 2;

    float sum = 0.0f;

    for (uint oy = 0; oy < out_h; oy++) {
        /* i = r - oy*stride + k_radius; needs 0 <= i < k */
        int i = (int)r - (int)(oy * stride) + (int)k_radius;
        if (i < 0 || i >= (int)k) continue;

        for (uint ox = 0; ox < out_w; ox++) {
            int j = (int)c - (int)(ox * stride) + (int)k_radius;
            if (j < 0 || j >= (int)k) continue;

            float go = grad_output[(z * out_w * out_h) + (oy * out_w) + ox];
            sum += go * weights[(uint)i * k + (uint)j];
        }
    }

    grad_input[(z * width * height) + (r * width) + c] = sum;
}


/*
 * conv_backward_weights_3d
 *
 * Computes grad_weights w.r.t. the kernel weights of the forward conv_3d.
 *
 * grad_weights[i][j] = sum over z, oy, ox of
 *     grad_output[z][oy][ox] * x[z][oy*stride - k_radius + i][ox*stride - k_radius + j]
 *
 * Grid: dispatched over (k, k, 1).
 * Each thread computes one element of grad_weights by looping over all
 * channels and all output spatial positions.
 *
 * Buffers:
 *   x            - original input to forward conv
 *   grad_output  - gradient from next layer
 *   grad_weights - output (k * k), does not need zero-init since each thread writes exactly once
 *   k, stride, width, height - same as forward
 *   out_w, out_h - spatial dimensions of forward output
 *   channels     - number of channels (depth)
 */
kernel void conv_backward_weights_3d(
    device const float* x           [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float*       grad_weights[[buffer(2)]],
    constant uint& k                [[buffer(3)]],
    constant uint& stride           [[buffer(4)]],
    constant uint& width            [[buffer(5)]],
    constant uint& height           [[buffer(6)]],
    constant uint& out_w            [[buffer(7)]],
    constant uint& out_h            [[buffer(8)]],
    constant uint& channels         [[buffer(9)]],
    uint2 index                     [[thread_position_in_grid]]
) {
    /* index.x = j (kernel col), index.y = i (kernel row) */
    uint i = index.y;
    uint j = index.x;
    uint k_radius = k / 2;

    float sum = 0.0f;

    for (uint z = 0; z < channels; z++) {
        for (uint oy = 0; oy < out_h; oy++) {
            for (uint ox = 0; ox < out_w; ox++) {
                uint input_r = oy * stride - k_radius + i;
                uint input_c = ox * stride - k_radius + j;

                /* bounds check (unsigned underflow also caught by >= check) */
                if (input_r >= height || input_c >= width) continue;

                float go = grad_output[(z * out_w * out_h) + (oy * out_w) + ox];
                float xv = x[(z * width * height) + (input_r * width) + input_c];
                sum += go * xv;
            }
        }
    }

    grad_weights[i * k + j] = sum;
}
