#pragma once

#include <webgpu/webgpu_cpp.h>
#include <array>
#include <cstdint>
#include <stdexcept>

#include "shader.hpp"

namespace webgpu_fft {

/// Reusable buffer bindings. The caller owns submission and synchronization.
class Binding {
   public:
    Binding(wgpu::ComputePipeline pipeline,
            wgpu::BindGroup group,
            std::uint32_t batch_count)
        : pipeline_(std::move(pipeline)),
          group_(std::move(group)),
          batch_count_(batch_count) {}

    void dispatch(const wgpu::ComputePassEncoder& pass) const {
        pass.SetPipeline(pipeline_);
        pass.SetBindGroup(0, group_);
        pass.DispatchWorkgroups(batch_count_);
    }

    void encode(const wgpu::CommandEncoder& encoder) const {
        auto pass = encoder.BeginComputePass();
        dispatch(pass);
        pass.End();
    }

   private:
    wgpu::ComputePipeline pipeline_;
    wgpu::BindGroup group_;
    std::uint32_t batch_count_;
};

/// Header-only Emdawnwebgpu/Dawn plan. All data stays in caller-owned buffers.
class Plan {
   public:
    explicit Plan(const wgpu::Device& device,
                  int length,
                  bool paired = false,
                  bool inverse = false)
        : device_(device), length_(length) {
        const auto code = shader(length, paired, inverse);
        if (!device_.GetLimits(&limits_))
            throw std::runtime_error(
                "WebGPU FFT could not query device limits");
        wgpu::ShaderSourceWGSL source{};
        source.code = code.c_str();
        wgpu::ShaderModuleDescriptor shader_descriptor{};
        shader_descriptor.nextInChain = &source;
        shader_descriptor.label = "webgpu-fft";
        const auto module = device.CreateShaderModule(&shader_descriptor);
        wgpu::ComputePipelineDescriptor descriptor{};
        descriptor.label = "webgpu-fft";
        descriptor.compute.module = module;
        descriptor.compute.entryPoint = "main";
        pipeline_ = device.CreateComputePipeline(&descriptor);
    }

    int length() const { return length_; }

    /// Offsets are byte offsets aligned to minStorageBufferOffsetAlignment.
    Binding bind(const wgpu::Buffer& input,
                 const wgpu::Buffer& output,
                 std::uint32_t batch_count = 1,
                 std::uint64_t input_offset = 0,
                 std::uint64_t output_offset = 0) const {
        if (batch_count == 0 ||
            batch_count > limits_.maxComputeWorkgroupsPerDimension)
            throw std::invalid_argument(
                "WebGPU FFT batch count exceeds device limits");
        if (input.Get() == output.Get())
            throw std::invalid_argument("WebGPU FFT requires distinct buffers");
        const std::uint64_t bytes = 16ULL * length_ * batch_count;
        std::array<wgpu::BindGroupEntry, 2> entries{};
        const std::array buffers{input, output};
        const std::array offsets{input_offset, output_offset};
        for (std::uint32_t i = 0; i < 2; ++i) {
            if (!buffers[i] || offsets[i] > buffers[i].GetSize() ||
                bytes > buffers[i].GetSize() - offsets[i] ||
                bytes > limits_.maxStorageBufferBindingSize ||
                offsets[i] % limits_.minStorageBufferOffsetAlignment != 0)
                throw std::invalid_argument("WebGPU FFT buffer range invalid");
            if (!(buffers[i].GetUsage() & wgpu::BufferUsage::Storage))
                throw std::invalid_argument(
                    "WebGPU FFT buffers need Storage usage");
            entries[i].binding = i;
            entries[i].buffer = buffers[i];
            entries[i].offset = offsets[i];
            entries[i].size = bytes;
        }
        wgpu::BindGroupDescriptor descriptor{};
        descriptor.layout = pipeline_.GetBindGroupLayout(0);
        descriptor.entryCount = entries.size();
        descriptor.entries = entries.data();
        return {pipeline_, device_.CreateBindGroup(&descriptor), batch_count};
    }

   private:
    wgpu::Device device_;
    wgpu::ComputePipeline pipeline_;
    wgpu::Limits limits_{};
    int length_;
};

}  // namespace webgpu_fft
