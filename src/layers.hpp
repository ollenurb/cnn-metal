#pragma once

#include "context.hpp"
#include <iostream>
#include <sstream>

#define SHAPE_MAX 4

/* Thin wrapper around a flat array on the GPU + shape info */
struct Tensor {
    MTL::Buffer* buffer;
    float* data;
    std::array<uint32_t, SHAPE_MAX> shape;

    Tensor(std::array<uint32_t, SHAPE_MAX> _shape) : shape(_shape) {
        auto device = MetalContext::get_instance()->get_device();
        size_t byte_size = sizeof(float) * std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>{});
        buffer = device->newBuffer(byte_size, MTL::ResourceStorageModeShared);

        std::stringstream ss;
        for (auto &s : shape) ss << s << ", ";
        std::cout << "Tensor: \n"
                    << " - Allocated " << byte_size << " bytes on GPU\n"
                    << " - Shape " << ss.str() << std::endl;

        data = static_cast<float*>(buffer->contents());
    }

    void copy_from(const Tensor& other) {
        assert(std::equal(shape.begin(), shape.end(), other.shape.begin(), other.shape.end()));
        size_t byte_size = sizeof(float) * std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>{});
        std::memcpy(data, other.data, byte_size);
    }

    ~Tensor() {
        buffer->release();
    }
};

/* Layer */
class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(const Tensor& x, Tensor& result) = 0;
    virtual void backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) = 0;
    virtual void allocate_gradients() {}
    virtual void free_gradients() {}
    virtual std::array<uint32_t, SHAPE_MAX> output_size(const std::array<uint32_t, SHAPE_MAX>& shape) = 0;
};

/* Convolution */
class Conv2DLayer : public Layer {
private:
    Tensor _weights;
    std::optional<Tensor> _weights_grad;
    uint8_t _k;
    uint8_t _stride;

public:
    Conv2DLayer(uint8_t k, uint8_t stride = 1);
    void forward(const Tensor& x, Tensor& result) override;
    void backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) override;
    void allocate_gradients() override;
    void free_gradients() override;
    std::array<uint32_t, SHAPE_MAX> output_size(const std::array<uint32_t, SHAPE_MAX>& shape) override;

    Tensor& get_weights();
};

/* ReLU */
class ReLULayer : public Layer {
public:
    void forward(const Tensor& x, Tensor& result) override;
    void backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) override;
    std::array<uint32_t, SHAPE_MAX> output_size(const std::array<uint32_t, SHAPE_MAX>& shape) override;
};

/* MaxPooling */
class MaxPoolLayer : public Layer {
private:
   uint8_t _k;
   uint8_t _stride;

public:
    MaxPoolLayer(uint8_t k, uint8_t stride = 1);
    void forward(const Tensor& x, Tensor& result) override;
    void backward(const Tensor& input, const Tensor& loss_grad, Tensor& input_grad) override;
    std::array<uint32_t, SHAPE_MAX> output_size(const std::array<uint32_t, SHAPE_MAX>& shape) override;
};
