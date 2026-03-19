#pragma once

#include "src/layers.hpp"

class Model {
private:
    /* train/inference mode */
    bool _train;
    /* input shape */
    std::array<uint32_t, SHAPE_MAX> _input_shape;

    /* employ a double buffering approach to save space (only in inference mode) */
    std::unique_ptr<Tensor> _src;
    std::unique_ptr<Tensor> _dest;

    /* layers define the model architecture */
    std::vector<std::unique_ptr<Layer>> _layers;

    /* activations should be saved in train mode */
    std::vector<std::unique_ptr<Tensor>> _activations;

    /*
     * since in inference mode we don't need to save intermediate activations,
     * we just use double buffering to save memory
     */
    void _forward_inference(const Tensor& input, Tensor& output) {
        _src->copy_from(input);
        auto src = _src.get();
        auto dst = _dest.get();
        for (auto& layer : _layers) {
            layer->forward(*src, *dst);
            std::swap(src, dst);
        }
        output.copy_from(*src);
    }

    /* forward train mode that save intermediate activations for backpropagation */
    void _forward_train(const Tensor& input, Tensor& output) {
        if (!_train) return;
    }

    void _propagate_shapes() {
        auto shape = _input_shape;
        for (auto& layer : _layers) {
            layer->set_input_shape(shape);
            shape = layer->output_size();
        }
    }

    /*  model architecture is statically defined here */
    void _allocate_layers() {
        /* block 0 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2));
        /* block 1 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2));
        /* block 2 */
        _layers.push_back(std::make_unique<Conv2DLayer>(3));
        _layers.push_back(std::make_unique<ReLULayer>());
        _layers.push_back(std::make_unique<MaxPoolLayer>(2));
    }

    /* allocate activations tensors */
    void _allocate_activations() {
        for (auto& layer : _layers) {
            auto output_size = layer->output_size();
            _activations.push_back(std::make_unique<Tensor>(output_size));
        }
    }

    /* allocate buffers for inference */
    void _allocate_buffers() {
        _src = std::make_unique<Tensor>(_input_shape);
        _dest = std::make_unique<Tensor>(_input_shape);
    }

    /* deallocate inference buffers */
    void _free_buffers() {
        _src.reset(nullptr);
        _dest.reset(nullptr);
    }

public:
    /* create a new instance of the convolutional model */
    explicit Model(std::array<uint32_t, SHAPE_MAX> input_shape, bool is_train = true)
        : _src(nullptr), _dest(nullptr) {
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
