#include "layers.hpp"
#include "src/context.hpp"
#include <cstdint>

/* ========== Conv2D implementation ========== */
Conv2DLayer::Conv2DLayer(uint8_t k, uint8_t stride) : _weights({1, 1, k, k}), _k(k), _stride(stride) { }

void Conv2DLayer::forward(const Tensor &x, Tensor &result) {
    uint32_t k32 = static_cast<uint32_t>(_k);
    uint32_t stride32 = static_cast<uint32_t>(_stride);
    uint32_t in_w = x.shape[3];
    uint32_t in_h = x.shape[2];
    auto out_size = output_size(x.shape);
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

void Conv2DLayer::backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) {
    uint32_t k32      = static_cast<uint32_t>(_k);
    uint32_t stride32 = static_cast<uint32_t>(_stride);
    uint32_t in_w     = input.shape[3];         // input spatial dims
    uint32_t in_h     = input.shape[2];
    uint32_t out_w    = loss_grad.shape[3];     // output spatial dims
    uint32_t out_h    = loss_grad.shape[2];
    uint32_t channels = input.shape[1];

    MetalContext::get_instance()->blocking_dispatch({
        {
            "conv_backward_input_3d",
            {
                loss_grad.buffer,
                _weights.buffer,
                input_grad.buffer
            },
            {
                {&k32,      sizeof(k32)},
                {&stride32, sizeof(stride32)},
                {&in_w,     sizeof(in_w)},
                {&in_h,     sizeof(in_h)},
                {&out_w,    sizeof(out_w)},
                {&out_h,    sizeof(out_h)}
            },
            { in_w, in_h, channels }            // dispatched over input shape
        },
        {
            "conv_backward_weights_3d",
            {
                input.buffer,
                loss_grad.buffer,
                _weights_grad.value().buffer
            },
            {
                {&k32,      sizeof(k32)},
                {&stride32, sizeof(stride32)},
                {&in_w,     sizeof(in_w)},
                {&in_h,     sizeof(in_h)},
                {&out_w,    sizeof(out_w)},
                {&out_h,    sizeof(out_h)},
                {&channels, sizeof(channels)}
            },
            { k32, k32, 1 } // dispatched over (k, k)
        }
    });
}

void Conv2DLayer::allocate_gradients() {
    _weights_grad = Tensor(_weights.shape);
}

void Conv2DLayer::free_gradients() {
    _weights_grad.reset();
}

std::array<uint32_t, SHAPE_MAX> Conv2DLayer::output_size(const std::array<uint32_t, SHAPE_MAX>& shape) {
    return {
        shape[0],
        shape[1],
        (shape[2] - _k) / _stride + 1,
        (shape[3] - _k) / _stride + 1
    };
}

/* TODO: weigths */
Tensor& Conv2DLayer::get_weights() {
    return _weights;
}

/* ========== ReLU implementation ========== */
void ReLULayer::forward(const Tensor &x, Tensor &result) {
    MetalContext::get_instance()->blocking_dispatch({
        {"relu_3d", {x.buffer, result.buffer}, {}, {x.shape[3], x.shape[2], x.shape[1]}}
    });
}

void ReLULayer::backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) {
    MetalContext::get_instance()->blocking_dispatch({
        {
            "relu_3d_backward",
            {input.buffer, loss_grad.buffer, input_grad.buffer},
            {},
            {input.shape[3], input.shape[2], input.shape[1]}
        }
    });
}

std::array<uint32_t, SHAPE_MAX> ReLULayer::output_size(const std::array<uint32_t, SHAPE_MAX>& shape) {
    return shape;
}

/* ========== MaxPool implementation ========== */
MaxPoolLayer::MaxPoolLayer(uint8_t k, uint8_t stride): _k(k), _stride(stride) { }

void MaxPoolLayer::forward(const Tensor &x, Tensor &result) {
    uint32_t k32 = static_cast<uint32_t>(_k);
    uint32_t stride32 = static_cast<uint32_t>(_stride);
    uint32_t in_w = x.shape[3];
    uint32_t in_h = x.shape[2];
    auto out_size = output_size(x.shape);
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

void MaxPoolLayer::backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) {

}

std::array<uint32_t, SHAPE_MAX> MaxPoolLayer::output_size(const std::array<uint32_t, SHAPE_MAX>& shape) {
    // same as convolution
    return {
        shape[0],
        shape[1],
        (shape[2] - _k) / _stride + 1,
        (shape[3] - _k) / _stride + 1
    };
}
