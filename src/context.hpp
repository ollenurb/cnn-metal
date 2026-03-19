/*
A singleton class to manage a Metal GPU
*/
#pragma once

#include <Metal/Metal.hpp>
#include <map>

typedef struct InlineParam {
    const void* data;
    size_t size;
} InlineParam;

typedef struct KernelDispatch {
    const char* method;
    std::vector<MTL::Buffer*> buffers;
    std::vector<InlineParam> params;
    std::tuple<size_t, size_t, size_t> grid_size = std::make_tuple(0, 0, 0);
    std::tuple<size_t, size_t, size_t> group_size = std::make_tuple(0, 0, 0);
} KernelDispatch;

class MetalContext {

private:
    static MetalContext *_instance;
    MetalContext();
    /* load kernels from a given library (used in the constructor) */
    void _load_kernels(MTL::Library *);
    /* holds functions loaded from the library */
    std::map<std::string, MTL::Function *> _function_map;
    /*
     * map function names to pipeline state objects (holds the compiled kernel and
     * other device-specific informations)
     */
    std::map<std::string, MTL::ComputePipelineState *> _function_pipeline_map;
    /* command queue to send commands to the GPU */
    MTL::CommandQueue *_command_queue;
    /* device associated with this gpu context */
    MTL::Device *_device;
    /* compute the default thread group size */
    MTL::Size _default_thread_group_size(const MTL::ComputePipelineState*, MTL::Size);

public:
    /* dispatch multiple kernels to the GPU */
    void blocking_dispatch(const std::vector<KernelDispatch>&);
    void encode(const std::vector<KernelDispatch>&);
    void commit_and_wait();

    MTL::Device *get_device();

    /*
     * we assume here that we are going to need just a single instance of a
     * Metal GPU context across the whole program.
     */
    static MetalContext *get_instance();
};
