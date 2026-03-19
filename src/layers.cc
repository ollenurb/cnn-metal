#include "layers.hpp"
#include "src/context.hpp"
#include <cstdint>

/* ========== Conv2D implementation ========== */
Conv2DLayer::Conv2DLayer(uint8_t k, uint8_t stride = 1) : _weights({1, 1, k, k}), _k(k), _stride(stride) { }

void Conv2DLayer::forward(const Tensor &x, Tensor &result) {
    uint32_t k32 = static_cast<uint32_t>(_k);
    uint32_t stride32 = static_cast<uint32_t>(_stride);
    uint32_t in_w = x.shape[3];
    uint32_t in_h = x.shape[2];
    auto out_size = output_size();
    uint32_t out_w = out_size[3];
    uint32_t out_h = out_size[2];

    MetalContext::get_instance()->blocking_dispatch({{
        "conv_3d",                               // function name
        {
            x.buffer,
            _weights.buffer,
            result.buffer
        },                                       // buffers
        {
            {&k32, sizeof(k32)},
            {&stride32, sizeof(stride32)},
            {&in_w, sizeof(in_w)},
            {&in_h, sizeof(in_h)}
        },                                       // constants
        { out_w, out_h, x.shape[1] }             // number of threads = output size
    }});
}

void Conv2DLayer::backward(const Tensor& loss_grad) {}

std::array<uint32_t, SHAPE_MAX> Conv2DLayer::output_size() {
    return {
        _input_shape[0],
        _input_shape[1],
        (_input_shape[2] - _k) / _stride + 1,
        (_input_shape[3] - _k) / _stride + 1
    };
}

/* TODO: weigths */
Tensor& Conv2DLayer::get_weights() {
    return _weights;
}

/* ========== ReLU implementation ========== */
void ReLULayer::forward(const Tensor &x, Tensor &result) {
    MetalContext::get_instance()->blocking_dispatch({
        {"relu_3d", {x.buffer, result.buffer}, {}, {x.shape[3], x.shape[2], x.shape[1]}},
    });
}

void ReLULayer::backward(const Tensor& loss_grad) {}

std::array<uint32_t, SHAPE_MAX> ReLULayer::output_size() {
    return _input_shape;
}

/* ========== MaxPool implementation ========== */
MaxPoolLayer::MaxPoolLayer(uint8_t k, uint8_t stride): _k(k), _stride(stride) { }

void MaxPoolLayer::forward(const Tensor &x, Tensor &result) {
    uint32_t k32 = static_cast<uint32_t>(_k);
    uint32_t stride32 = static_cast<uint32_t>(_stride);
    uint32_t in_w = x.shape[3];
    uint32_t in_h = x.shape[2];
    auto out_size = output_size();
    uint32_t out_w = out_size[3];
    uint32_t out_h = out_size[2];

    MetalContext::get_instance()->blocking_dispatch({{
        "max_pool_3d",
        { x.buffer, result.buffer },
        {
            {&k32, sizeof(k32)},
            {&stride32, sizeof(stride32)},
            {&in_w, sizeof(in_w)},
            {&in_h, sizeof(in_h)},
        },
        { out_w, out_h, x.shape[1] }
    }});
}

void MaxPoolLayer::backward(const Tensor& loss_grad) {}

std::array<uint32_t, SHAPE_MAX> MaxPoolLayer::output_size() {
    // same as convolution
    return {
        _input_shape[0],
        _input_shape[1],
        (_input_shape[2] - _k) / _stride + 1,
        (_input_shape[3] - _k) / _stride + 1
    };
}
