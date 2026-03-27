
#include "context.hpp"
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

MetalContext* MetalContext::_instance = nullptr;

MetalContext* MetalContext::get_instance() {
    if (!_instance) {
        _instance = new MetalContext();
    }
    return _instance;
}

MTL::Device* MetalContext::get_device() {
    return _device;
}

/* load kernels from a given library */
void MetalContext::_load_kernels(MTL::Library *op_library) {
    NS::Error *error = nullptr;

    auto function_names = op_library->functionNames();
    for (size_t i = 0; i < function_names->count(); i++) {
        auto name_nsstring = function_names->object(i)->description();
        auto name_utf8 = name_nsstring->utf8String();
        /* load function into the map */
        _function_map[name_utf8] = (op_library->newFunction(name_nsstring));
        /* create a pipeline for each function */
        _function_pipeline_map[name_utf8] = _device->newComputePipelineState(_function_map[name_utf8], &error);

        if (_function_pipeline_map[name_utf8] == nullptr) {
            std::stringstream ss;
            ss << "Failed to create pipeline state object for "
               << name_utf8 << ", error "
               << error->description()->utf8String() << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

MetalContext::MetalContext() {
    _device = MTL::CreateSystemDefaultDevice();

    /*
    * load the shader (.metal file) containing compute kernels (METAL_LIB is a
    * macro variable that is filled by the CMake build system during build time)
    */
    NS::Error *error = nullptr;
    auto filepath = NS::String::string(METAL_LIB, NS::ASCIIStringEncoding);
    MTL::Library *op_library = _device->newLibrary(filepath, &error);

    if (op_library == nullptr) {
        std::stringstream ss;
        ss << "Failed to find the default library.\n" << error->description()->utf8String() << std::endl;
        throw std::runtime_error(ss.str());
    }

    /* load kernels */
    _load_kernels(op_library);

    _command_queue = _device->newCommandQueue();
    if (_command_queue == nullptr) {
        std::stringstream ss;
        ss << "Failed to find the command queue." << std::endl;
        throw std::runtime_error(ss.str());
    }
}

MTL::Size MetalContext::_default_thread_group_size(const MTL::ComputePipelineState* pipeline_state, MTL::Size grid) {
    NS::UInteger max_threads_group = pipeline_state->maxTotalThreadsPerThreadgroup();
    NS::UInteger warp_size = pipeline_state->threadExecutionWidth();

    size_t x = std::min((size_t) warp_size, grid.width);
    size_t y = std::min(max_threads_group / x, grid.height);
    size_t z = std::min(max_threads_group / (x*y), grid.depth);
    return MTL::Size::Make(x, y, z);
}

void MetalContext::blocking_dispatch(const std::vector<KernelDispatch> &dispatches) {
    MTL::CommandBuffer *command_buffer = _command_queue->commandBuffer();
    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();
    assert(compute_encoder != nullptr);

    for (const auto &d : dispatches) {
        auto group_size = MTL::Size::Make(std::get<0>(d.group_size), std::get<1>(d.group_size), std::get<2>(d.group_size));
        auto grid_size = MTL::Size::Make(std::get<0>(d.grid_size), std::get<1>(d.grid_size), std::get<2>(d.grid_size));
        /*
         * bind the compiled kernel and device-specific configuration to the
         * command encoder
         */
        auto *pipeline_state = _function_pipeline_map[d.method];
        compute_encoder->setComputePipelineState(pipeline_state);
        /* bind each buffer to the command encoder */
        for (size_t i = 0; i < d.buffers.size(); ++i) {
            compute_encoder->setBuffer(d.buffers[i], 0, i);
        }
        /* automatically compute group size if left with default values */
        if (group_size.width == 0 && group_size.height == 0 && group_size.depth == 0) {
            group_size = _default_thread_group_size(pipeline_state, grid_size);
        }
        /* set constants data AFTER the buffers */
        for (size_t i = 0; i < d.params.size(); ++i) {
            compute_encoder->setBytes(d.params[i].data, d.params[i].size, i + d.buffers.size());
        }

        std::cout << "Dispatching '" << d.method << "' with grid_size = ("
                    << grid_size.width << ", "
                    << grid_size.height << ", "
                    << grid_size.depth << "), group_size = ("
                    << group_size.width << ", "
                    << group_size.height << ", "
                    << group_size.depth << ") " << std::endl;
        /* encode the compute command */
        compute_encoder->dispatchThreads(grid_size, group_size);
    }
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

// TOFIX:: we need to ensure that the compute_encoder is returned so that when
// commit_and_wait is called it uses it to call commit and waitUntilCompleted
void MetalContext::encode(const std::vector<KernelDispatch>& dispatches) {
    MTL::CommandBuffer *command_buffer = _command_queue->commandBuffer();
    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();
    assert(compute_encoder != nullptr);

    for (const auto &d : dispatches) {
        auto group_size = MTL::Size::Make(std::get<0>(d.group_size), std::get<1>(d.group_size), std::get<2>(d.group_size));
        auto grid_size = MTL::Size::Make(std::get<0>(d.grid_size), std::get<1>(d.grid_size), std::get<2>(d.grid_size));
        /* bind the compiled kernel and device-specific configuration to the command encoder */
        auto *pipeline_state = _function_pipeline_map[d.method];
        compute_encoder->setComputePipelineState(pipeline_state);
        /* bind each buffer to the command encoder */
        for (size_t i = 0; i < d.buffers.size(); ++i) {
            compute_encoder->setBuffer(d.buffers[i], 0, i);
        }
        /* automatically compute group size if left with default values */
        if (group_size.width == 0 && group_size.height == 0 && group_size.depth == 0) {
            group_size = _default_thread_group_size(pipeline_state, grid_size);
        }
        std::cout << "Dispatching '" << d.method << "' with grid_size = ("
                  << grid_size.width << ", "
                  << grid_size.height << ", "
                  << grid_size.depth << "), group_size = ("
                  << group_size.width << ", "
                  << group_size.height << ", "
                  << group_size.depth << ") " << std::endl;
        /* encode the compute command */
        compute_encoder->dispatchThreads(grid_size, group_size);
    }
}

void MetalContext::commit_and_wait() {
    MTL::CommandBuffer *command_buffer = _command_queue->commandBuffer();
    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();
    assert(compute_encoder != nullptr);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}
