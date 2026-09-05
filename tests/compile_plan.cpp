#include <webgpu_fft/plan.hpp>

// Compiles the complete public wrapper against the consumer's WebGPU headers.
void record_fft(const wgpu::Device& device,
                const wgpu::CommandEncoder& encoder,
                const wgpu::Buffer& input,
                const wgpu::Buffer& output) {
    webgpu_fft::Plan plan(device, 36, true);
    const auto binding = plan.bind(input, output, 4);
    binding.encode(encoder);
}
