#pragma once

#include <webgpu/webgpu_cpp.h>
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <utility>

#include "program.hpp"

namespace webgpu_fft {

struct Dispatch {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup group;
    std::uint32_t x = 1;
    std::uint32_t y = 1;
};

/// Reusable bindings and RAII scratch; no allocations or submits in encode.
class Binding {
   public:
    Binding(std::vector<Dispatch> passes, std::vector<wgpu::Buffer> resources)
        : passes_(std::move(passes)), resources_(std::move(resources)) {}

    Binding(wgpu::ComputePipeline pipeline,
            wgpu::BindGroup group,
            std::uint32_t batch_count)
        : Binding({{std::move(pipeline), std::move(group), batch_count, 1}},
                  {}) {}

    void dispatch(const wgpu::ComputePassEncoder& pass) const {
        for (const auto& item : passes_) {
            pass.SetPipeline(item.pipeline);
            pass.SetBindGroup(0, item.group);
            pass.DispatchWorkgroups(item.x, item.y);
        }
    }
    void encode(const wgpu::CommandEncoder& encoder) const {
        auto pass = encoder.BeginComputePass();
        dispatch(pass);
        pass.End();
    }

   private:
    std::vector<Dispatch> passes_;
    std::vector<wgpu::Buffer> resources_;
};

/// Emdawnwebgpu/Dawn plan. Owns immutable tables; bind owns reusable scratch.
class Plan {
   public:
    explicit Plan(const wgpu::Device& device,
                  int length,
                  bool paired = false,
                  bool inverse = false)
        : Plan(device, Options{length, paired, inverse, Transform::C2C}) {}

    explicit Plan(const wgpu::Device& device, Options options)
        : device_(device), program_(program(options)) {
        if (!device_.GetLimits(&limits_))
            throw std::runtime_error(
                "WebGPU FFT could not query device limits");
        if (!program_.code.empty()) {
            const auto table_bytes = program_.table.size() * sizeof(float);
            if (table_bytes > limits_.maxStorageBufferBindingSize ||
                table_bytes > limits_.maxBufferSize)
                throw std::invalid_argument(
                    "WebGPU FFT table exceeds device limits");
            std::array<wgpu::BindGroupLayoutEntry, 4> entries{};
            for (std::uint32_t i = 0; i < entries.size(); ++i) {
                entries[i].binding = i;
                entries[i].visibility = wgpu::ShaderStage::Compute;
                entries[i].buffer.type =
                    i == 3   ? wgpu::BufferBindingType::Uniform
                    : i == 1 ? wgpu::BufferBindingType::Storage
                             : wgpu::BufferBindingType::ReadOnlyStorage;
            }
            wgpu::BindGroupLayoutDescriptor group_descriptor{};
            group_descriptor.entryCount = entries.size();
            group_descriptor.entries = entries.data();
            layout_ = device_.CreateBindGroupLayout(&group_descriptor);
            wgpu::PipelineLayoutDescriptor layout_descriptor{};
            layout_descriptor.bindGroupLayoutCount = 1;
            layout_descriptor.bindGroupLayouts = &layout_;
            const auto pipeline_layout =
                device_.CreatePipelineLayout(&layout_descriptor);
            const auto module = make_module(program_.code);
            for (const auto& stage : program_.stages) {
                if (stage.entry_point == "main" ||
                    pipelines_.contains(stage.entry_point))
                    continue;
                pipelines_[stage.entry_point] =
                    make_pipeline(module, stage.entry_point, pipeline_layout);
            }
            table_ = make_buffer(table_bytes, wgpu::BufferUsage::Storage |
                                                  wgpu::BufferUsage::CopyDst);
            device_.GetQueue().WriteBuffer(table_, 0, program_.table.data(),
                                          table_bytes);
            // WriteBuffer copies the immutable host table before returning.
            std::vector<float>().swap(program_.table);
        }
        if (!program_.small_code.empty())
            pipelines_["main"] =
                make_pipeline(make_module(program_.small_code), "main", {});
    }

    int length() const { return program_.length; }
    Transform transform() const { return program_.transform; }

    /// Offsets in bytes, storage-aligned. Each binding owns its own scratch.
    Binding bind(const wgpu::Buffer& input,
                 const wgpu::Buffer& output,
                 std::uint32_t batch_count = 1,
                 std::uint64_t input_offset = 0,
                 std::uint64_t output_offset = 0) const {
        if (batch_count == 0 ||
            (program_.small &&
             batch_count > limits_.maxComputeWorkgroupsPerDimension))
            throw std::invalid_argument(
                "WebGPU FFT batch count exceeds device limits");
        if (input.Get() == output.Get())
            throw std::invalid_argument("WebGPU FFT requires distinct buffers");
        const std::array sizes{program_.input_bytes() * batch_count,
                               program_.output_bytes() * batch_count};
        const std::array buffers{input, output};
        const std::array offsets{input_offset, output_offset};
        for (std::size_t i = 0; i < 2; ++i) {
            if (!buffers[i] || offsets[i] > buffers[i].GetSize() ||
                sizes[i] > buffers[i].GetSize() - offsets[i] ||
                sizes[i] > limits_.maxStorageBufferBindingSize ||
                offsets[i] % limits_.minStorageBufferOffsetAlignment != 0)
                throw std::invalid_argument("WebGPU FFT buffer range invalid");
            if (!(buffers[i].GetUsage() & wgpu::BufferUsage::Storage))
                throw std::invalid_argument(
                    "WebGPU FFT buffers need Storage usage");
        }
        const auto scratch_size = 16ULL * program_.fft_length * batch_count;
        if (program_.stages.size() > 1 &&
            (scratch_size > limits_.maxStorageBufferBindingSize ||
             scratch_size > limits_.maxBufferSize))
            throw std::invalid_argument(
                "WebGPU FFT scratch exceeds device limits");
        const std::uint64_t groups =
            (std::uint64_t(program_.fft_length) * batch_count + 63) / 64;
        const auto x = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            groups, limits_.maxComputeWorkgroupsPerDimension));
        const auto y = (groups + x - 1) / x;
        if (y > limits_.maxComputeWorkgroupsPerDimension)
            throw std::invalid_argument(
                "WebGPU FFT dispatch exceeds device limits");
        std::vector<wgpu::Buffer> resources;
        std::array<wgpu::Buffer, 2> scratch;
        if (program_.stages.size() > 1) {
            for (auto& buffer : scratch) {
                buffer = make_buffer(scratch_size, wgpu::BufferUsage::Storage);
                resources.push_back(buffer);
            }
        }
        std::vector<Dispatch> passes;
        for (std::size_t i = 0; i < program_.stages.size(); ++i) {
            const auto& stage = program_.stages[i];
            const bool small = stage.entry_point == "main";
            const auto& pipeline = pipelines_.at(stage.entry_point);
            std::array<wgpu::BindGroupEntry, 4> entries{};
            entries[0].binding = 0;
            entries[0].buffer = i == 0 ? input : scratch[(i - 1) % 2];
            entries[0].offset = i == 0 ? input_offset : 0;
            entries[0].size = i == 0 ? sizes[0] : scratch_size;
            const bool last = i + 1 == program_.stages.size();
            entries[1].binding = 1;
            entries[1].buffer = last ? output : scratch[i % 2];
            entries[1].offset = last ? output_offset : 0;
            entries[1].size = last ? sizes[1] : scratch_size;
            if (!small) {
                const std::array<std::uint32_t, 8> params{
                    static_cast<std::uint32_t>(program_.length),
                    static_cast<std::uint32_t>(program_.fft_length),
                    batch_count,
                    stage.span,
                    static_cast<std::uint32_t>(program_.transform),
                    stage.flags,
                    0,
                    0};
                auto uniform =
                    make_buffer(sizeof(params), wgpu::BufferUsage::Uniform |
                                                    wgpu::BufferUsage::CopyDst);
                device_.GetQueue().WriteBuffer(uniform, 0, params.data(),
                                               sizeof(params));
                resources.push_back(uniform);
                entries[2].binding = 2;
                entries[2].buffer = table_;
                entries[2].size = table_.GetSize();
                entries[3].binding = 3;
                entries[3].buffer = uniform;
                entries[3].size = sizeof(params);
            }
            wgpu::BindGroupDescriptor descriptor{};
            descriptor.layout =
                small ? pipeline.GetBindGroupLayout(0) : layout_;
            descriptor.entryCount = small ? 2 : 4;
            descriptor.entries = entries.data();
            passes.push_back({pipeline, device_.CreateBindGroup(&descriptor),
                              small ? batch_count : x,
                              small ? 1u : static_cast<std::uint32_t>(y)});
        }
        return {std::move(passes), std::move(resources)};
    }

   private:
    wgpu::Buffer make_buffer(std::uint64_t size,
                             wgpu::BufferUsage usage) const {
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = size;
        descriptor.usage = usage;
        return device_.CreateBuffer(&descriptor);
    }
    wgpu::ShaderModule make_module(const std::string& code) const {
        wgpu::ShaderSourceWGSL source{};
        source.code = code.c_str();
        wgpu::ShaderModuleDescriptor descriptor{};
        descriptor.nextInChain = &source;
        return device_.CreateShaderModule(&descriptor);
    }
    wgpu::ComputePipeline make_pipeline(
        const wgpu::ShaderModule& module,
        const std::string& entry,
        const wgpu::PipelineLayout& layout) const {
        wgpu::ComputePipelineDescriptor descriptor{};
        descriptor.layout = layout;
        descriptor.compute.module = module;
        descriptor.compute.entryPoint = entry.c_str();
        return device_.CreateComputePipeline(&descriptor);
    }
    wgpu::Device device_;
    Program program_;
    wgpu::Limits limits_{};
    wgpu::BindGroupLayout layout_;
    wgpu::Buffer table_;
    std::map<std::string, wgpu::ComputePipeline> pipelines_;
};

}  // namespace webgpu_fft
