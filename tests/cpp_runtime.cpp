#include <emscripten.h>
#include <webgpu_fft/plan.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

extern "C" void fft_report(int success, const char* detail);

class RuntimeTest : public std::enable_shared_from_this<RuntimeTest> {
   public:
    void start() {
        instance_ = wgpu::Instance(wgpuCreateInstance(nullptr));
        const auto self = shared_from_this();
        wgpu::RequestAdapterOptions options{};
        options.powerPreference = wgpu::PowerPreference::HighPerformance;
        instance_.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                   wgpu::StringView) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    self->fail("adapter request failed");
                    return;
                }
                self->adapter_ = std::move(adapter);
                self->request_device();
            });
    }

   private:
    void fail(const char* message) {
        failed_ = true;
        fft_report(0, message);
    }
    void request_device() {
        const auto self = shared_from_this();
        wgpu::DeviceDescriptor descriptor{};
        descriptor.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView,
               RuntimeTest* test) {
                test->fail("uncaptured WebGPU validation error");
            },
            this);
        adapter_.RequestDevice(
            &descriptor, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestDeviceStatus status, wgpu::Device device,
                   wgpu::StringView) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    self->fail("device request failed");
                    return;
                }
                self->device_ = std::move(device);
                self->run();
            });
    }
    wgpu::Buffer buffer(std::uint64_t bytes, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = bytes;
        descriptor.usage = usage;
        return device_.CreateBuffer(&descriptor);
    }
    void run() {
        if (failed_) return;
        if (test_ == 12) {
            fft_report(1, "12 Emdawnwebgpu C++ Plan runtime cases PASS");
            return;
        }
        const int n = test_ < 4 ? 36 : test_ < 8 ? 257 : 1024;
        const bool paired = (test_ % 4) >= 2, real = test_ % 2 != 0;
        constexpr int BATCHES = 2;
        constexpr std::uint64_t OFFSET = 256;
        const int input_width = real ? 2 : 4,
                  output_width = real ? n / 2 + 1 : n;
        std::vector<float> input(BATCHES * n * input_width);
        for (int b = 0; b < BATCHES; ++b)
            for (int j = 0; j < n; ++j) {
                const int i = (b * n + j) * input_width;
                input[i] = static_cast<float>(std::sin(j * 0.31 + b));
                if (real)
                    input[i + 1] =
                        paired ? static_cast<float>(std::cos(j * 0.13) * 1e-8)
                               : 0;
                else {
                    input[i + 1] = static_cast<float>(std::cos(j * 0.17 + b));
                    input[i + 2] = paired ? 1e-8F : 0;
                    input[i + 3] = paired ? -2e-8F : 0;
                }
            }
        const std::uint64_t input_bytes = input.size() * sizeof(float),
                            output_bytes = 16ULL * BATCHES * output_width;
        const auto usage = wgpu::BufferUsage::Storage |
                           wgpu::BufferUsage::CopySrc |
                           wgpu::BufferUsage::CopyDst;
        auto a = buffer(OFFSET + input_bytes, usage),
             b = buffer(OFFSET + output_bytes, usage),
             c = buffer(OFFSET + input_bytes, usage);
        auto read =
            buffer(output_bytes + input_bytes,
                   wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
        device_.GetQueue().WriteBuffer(a, OFFSET, input.data(), input_bytes);
        using webgpu_fft::Transform;
        webgpu_fft::Plan forward(
            device_, {.length = n,
                      .paired = paired,
                      .transform = real ? Transform::R2C : Transform::C2C});
        webgpu_fft::Plan inverse(
            device_, {.length = n,
                      .paired = paired,
                      .inverse = true,
                      .transform = real ? Transform::C2R : Transform::C2C});
        auto fwd = forward.bind(a, b, BATCHES, OFFSET, OFFSET);
        auto inv = inverse.bind(b, c, BATCHES, OFFSET, OFFSET);
        const auto encoder = device_.CreateCommandEncoder();
        fwd.encode(encoder);
        inv.encode(encoder);
        encoder.CopyBufferToBuffer(b, OFFSET, read, 0, output_bytes);
        encoder.CopyBufferToBuffer(c, OFFSET, read, output_bytes, input_bytes);
        const auto command = encoder.Finish();
        device_.GetQueue().Submit(1, &command);
        const auto self = shared_from_this();
        read.MapAsync(
            wgpu::MapMode::Read, 0, output_bytes + input_bytes,
            wgpu::CallbackMode::AllowSpontaneous,
            [self, read, input = std::move(input), n, paired, real,
             output_width, output_bytes,
             input_bytes](wgpu::MapAsyncStatus status, wgpu::StringView) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    self->fail("readback mapping failed");
                    return;
                }
                const auto values = std::span(
                    static_cast<const float*>(read.GetConstMappedRange()),
                    (output_bytes + input_bytes) / sizeof(float));
                double error = 0;
                auto compare = [&](double actual, double expected) {
                    if (!std::isfinite(actual)) {
                        error = INFINITY;
                        return;
                    }
                    error = std::max(error, std::abs(actual - expected) /
                                                (1 + std::abs(expected)));
                };
                const int width = real ? 2 : 4;
                const auto restored =
                    values.subspan(output_bytes / sizeof(float));
                for (std::size_t i = 0; i < input.size(); i += width) {
                    compare(double(restored[i]) + restored[i + (real ? 1 : 2)],
                            double(input[i]) + input[i + (real ? 1 : 2)]);
                    if (!real)
                        compare(double(restored[i + 1]) + restored[i + 3],
                                double(input[i + 1]) + input[i + 3]);
                }
                for (int batch = 0; batch < BATCHES; ++batch)
                    for (int k :
                         {0, 1, 7, output_width / 2, output_width - 1}) {
                        double r = 0, im = 0;
                        for (int j = 0; j < n; ++j) {
                            const int at = (batch * n + j) * width;
                            const double a =
                                double(input[at]) + input[at + (real ? 1 : 2)];
                            const double b =
                                real ? 0
                                     : double(input[at + 1]) + input[at + 3];
                            const double angle = -2 * std::numbers::pi *
                                                 ((std::uint64_t(j) * k) % n) /
                                                 n;
                            r += a * std::cos(angle) - b * std::sin(angle);
                            im += a * std::sin(angle) + b * std::cos(angle);
                        }
                        const int at = 4 * (batch * output_width + k);
                        compare(double(values[at]) + values[at + 2], r);
                        compare(double(values[at + 1]) + values[at + 3], im);
                    }
                read.Unmap();
                if (!std::isfinite(error) || error > (paired ? 3e-10 : 3e-4)) {
                    self->fail("C++ runtime numerical comparison failed");
                    return;
                }
                std::printf("PASS C++ Plan N=%d paired=%d real=%d error=%.3e\n",
                            n, paired, real, error);
                ++self->test_;
                self->run();
            });
    }
    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    int test_ = 0;
    bool failed_ = false;
};

int main() {
    static const auto test = std::make_shared<RuntimeTest>();
    test->start();
    return 0;
}
