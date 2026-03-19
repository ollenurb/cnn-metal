#include <metal_stdlib>
using namespace metal;


kernel void conv_2d(
    device const float* x       [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* result        [[buffer(2)]],
    constant uint& k            [[buffer(3)]],
    uint2 index                 [[thread_position_in_grid]],
    uint2 grid                  [[threads_per_grid]]
) {
    // kernel radius is k/2
    uint k_radius = k/2;
    if (index.x < k_radius || index.x >= grid.x - k_radius || index.y < k_radius || index.y >= grid.y - k_radius) return;

    float sum = 0;
    for (uint i = 0; i < k; i++) {
        for (uint j = 0; j < k; j++) {
            uint c = index.x - k_radius + j;
            uint r = index.y - k_radius + i;
            sum += weights[i * k + j] * x[r * grid.x + c];
        }
    }
    result[index.y * grid.x + index.x] = sum;
}

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

kernel void conv_3d_old(
    device const float* x       [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* result        [[buffer(2)]],
    constant uint& k            [[buffer(3)]],
    uint3 index                 [[thread_position_in_grid]],
    uint3 grid                  [[threads_per_grid]]
) {
    uint k_radius = k/2;
    if (index.x < k_radius || index.x >= grid.x - k_radius || index.y < k_radius || index.y >= grid.y - k_radius) return;

    float sum = 0;
    for (uint i = 0; i < k; i++) {
        for (uint j = 0; j < k; j++) {
            uint c = index.x - k_radius + j;
            uint r = index.y - k_radius + i;

            sum += weights[i * k + j] * x[(index.z * grid.x * grid.y) + (r * grid.x) + c];
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


kernel void add_arrays(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] + Y[index];
}

kernel void multiply_arrays(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] * Y[index];
}


kernel void central_difference(
    device const float* delta [[buffer(0)]],
    device const float* X     [[buffer(1)]],
    device float* result      [[buffer(2)]],
    uint index                [[thread_position_in_grid]],
    uint arrayLength          [[threads_per_grid]])
{
    if (index == 0)
    {
        result[index] = (X[index + 1] - X[index]) /  *delta;
    }
    else if (index == arrayLength - 1)
    {
        result[index] = (X[index] - X[index - 1]) /  *delta;
    }
    else
    {
        result[index] = (X[index + 1] - X[index - 1]) / (2 * *delta);
    }
}

int linear_IDX(int pos1, int pos2, int shape1, int shape2)
{
    return pos1 * shape2 + pos2;
}

kernel void quadratic2d(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint2 index           [[thread_position_in_grid]],
    uint2 grid            [[threads_per_grid]])
{
    int idx = linear_IDX(index.x, index.y, grid.x, grid.y);
    result[idx] = X[idx] * X[idx] / (grid.x * grid.x) + Y[idx] * Y[idx] / (grid.y * grid.y);
}

kernel void laplacian2d(
    device const float* X [[buffer(0)]],
    device float* result  [[buffer(1)]],
    uint2 index           [[thread_position_in_grid]],
    uint2 grid            [[threads_per_grid]])
{
    if (index.x > 0 and index.y < grid.y - 1 and index.y > 0 and index.x < grid.x - 1) {
        int idx = linear_IDX(index.x, index.y, grid.x, grid.y);

        int idx_xm1 = linear_IDX(index.x-1, index.y, grid.x, grid.y);
        int idx_xp1 = linear_IDX(index.x+1, index.y, grid.x, grid.y);

        int idx_ym1 = linear_IDX(index.x, index.y-1, grid.x, grid.y);
        int idx_yp1 = linear_IDX(index.x, index.y+1, grid.x, grid.y);


        // Five-point stencil:
        result[idx] = X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1] - 4 * X[idx];
    }
}

kernel void laplacian2d9p(
                  device const float* X [[buffer(0)]],
                  device float* result  [[buffer(1)]],
                  uint2 index           [[thread_position_in_grid]],
                  uint2 grid     [[threads_per_grid]])
{
    if (index.x > 0 and index.y < grid.y - 1 and index.y > 0 and index.x < grid.x - 1){
        int idx = linear_IDX(index.x, index.y, grid.x, grid.y);

        int idx_xm1 = linear_IDX(index.x-1, index.y, grid.x, grid.y);
        int idx_xp1 = linear_IDX(index.x+1, index.y, grid.x, grid.y);

        int idx_ym1 = linear_IDX(index.x, index.y-1, grid.x, grid.y);
        int idx_yp1 = linear_IDX(index.x, index.y+1, grid.x, grid.y);


        int idx_xm1ym1 = linear_IDX(index.x-1, index.y-1, grid.x, grid.y);
        int idx_xp1yp1 = linear_IDX(index.x+1, index.y+1, grid.x, grid.y);

        int idx_xm1yp1 = linear_IDX(index.x-1, index.y+1, grid.x, grid.y);
        int idx_xp1ym1 = linear_IDX(index.x+1, index.y-1, grid.x, grid.y);


        // Five-point stencil:
        result[idx] =
            0.25 * (X[idx_xm1ym1] + X[idx_xp1yp1] + X[idx_xm1yp1] + X[idx_xp1ym1])+
            0.5 * (X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1]) -
            3 * X[idx];
    }
}
