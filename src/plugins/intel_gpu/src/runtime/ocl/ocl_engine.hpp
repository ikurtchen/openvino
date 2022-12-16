// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ocl_device.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {
namespace ocl {

class ocl_engine : public engine {
public:
    ocl_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf, const InferenceEngine::ITaskExecutor::Ptr task_executor);
    engine_types type() const override { return engine_types::ocl; };
    runtime_types runtime_type() const override { return runtime_types::ocl; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout, const size_t offset = 0) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::cl_mem; }

    const cl::Context& get_cl_context() const;
    const cl::Device& get_cl_device() const;
    const cl::UsmHelper& get_usm_helper() const;

    bool extension_supported(std::string extension) const;

    stream_ptr create_stream() const override;
    stream_ptr create_stream(void *handle) const override;
    stream& get_program_stream() const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Returns onednn engine object which shares device and context with current engine
    // If onednn engine has not been created yet, it creates on-demand.
    dnnl::engine& get_onednn_engine() const override;
#endif

    static std::shared_ptr<cldnn::engine> create(const device::ptr device, runtime_types runtime_type,
                const engine_configuration& configuration, const InferenceEngine::ITaskExecutor::Ptr task_executor);

private:
    std::string _extensions;
    std::unique_ptr<stream> _program_stream;
    std::unique_ptr<cl::UsmHelper> _usm_helper;

#ifdef ENABLE_ONEDNN_FOR_GPU
    mutable std::mutex onednn_mutex;
    mutable std::shared_ptr<dnnl::engine> _onednn_engine;
#endif
};

}  // namespace ocl
}  // namespace cldnn
