#pragma once

#include "src/layers.hpp"

class Model {
private:
    /* train/inference mode */
    bool _train;
    /* input shape */
    std::array<uint32_t, SHAPE_MAX> _input_shape;

    /* employ a double buffering approach to save space (only in inference mode) */
    std::optional<Tensor> _src;
    std::optional<Tensor> _dest;

    /* layers define the model architecture */
    std::vector<std::unique_ptr<Layer>> _layers;

    /* activations should be saved in train mode */
    std::vector<Tensor> _activations;

    /*
     * since in inference mode we don't need to save intermediate activations,
     * we just use double buffering to save memory
     */
    void _forward_inference(const Tensor& input, Tensor& output) {
        _src->copy_from(input);
        auto src = &_src.value();
        auto dst = &_dest.value();
        for (auto& layer : _layers) {
            layer->forward(*src, *dst);
            std::swap(src, dst);
        }
        output.copy_from(*src);
    }

    /* forward train mode that save intermediate activations for backpropagation */
    void _forward_train(const Tensor& input, Tensor& output) {
        auto prev = &input;
        for (size_t i = 0; i < _layers.size(); i++) {
            _layers[i]->forward(*prev, _activations[i]);
            prev = &_activations[i];
        }
        output.copy_from(*prev);
    }

    /*  model architecture is statically defined here */
    void _allocate_layers() {
        /* block 0 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2, 2));
        /* block 1 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2, 2));
        /* block 2 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2, 2));
    }

    /* allocate activations tensors */
    void _allocate_activations() {
        _activations.reserve(_layers.size());
        auto shape = _input_shape;
        for (auto& layer : _layers) {
            shape = layer->output_size(shape);
            _activations.emplace_back(shape);
        }
    }

    void _allocate_gradients() {
        for (auto& layer : _layers) {
            layer->allocate_gradients();
        }
    }

    /* allocate buffers for inference */
    void _allocate_buffers() {
        _src.emplace(_input_shape);
        _dest.emplace(_input_shape);
    }

    /* deallocate inference buffers */
    void _free_buffers() {
        _src.reset();
        _dest.reset();
    }

public:
    /* create a new instance of the convolutional model */
    explicit Model(std::array<uint32_t, SHAPE_MAX> input_shape, bool is_train = true)
        : _train(is_train), _input_shape(input_shape) {
            _allocate_layers();
            set_train(is_train);
        }

    /* forward pass */
    void forward(const Tensor& input, Tensor& output) {
        if (_train) {
            _forward_train(input, output);
        }
        else {
            _forward_inference(input, output);
        }
    }

    /* backward pass */
    void backward(const Tensor& grad_output) {
        if (!_train) return;
    }

    void set_train(bool new_value) {
        if (_train == new_value) return;
        _train = new_value;
        if (_train) {
            _allocate_activations();
            _free_buffers();
        }
        else {
            _activations.clear();
            _allocate_buffers();
        }
    }
};
