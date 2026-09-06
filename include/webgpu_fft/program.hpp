#pragma once

#include <complex>
#include <cstdint>
#include <utility>

#include "shader.hpp"

namespace webgpu_fft {

enum class Transform { C2C, R2C, C2R };
struct Options {
    int length = 0;
    bool paired = false;
    bool inverse = false;
    Transform transform = Transform::C2C;
    bool optimized = true;
};
struct Stage {
    std::string entry_point;
    std::uint32_t span = 0;
    std::uint32_t flags = 0;
};
/// Pure-host plan description, also useful for testing WGSL outside C++.
struct Program {
    int length = 0;
    int fft_length = 0;
    Transform transform = Transform::C2C;
    bool paired = false;
    bool inverse = false;
    bool small = false;
    bool bluestein = false;
    bool optimized = true;
    std::vector<float> table;
    std::string code;
    std::string small_code;
    std::vector<Stage> stages;
    std::uint64_t input_bytes() const {
        if (transform == Transform::R2C) return 8ULL * length;
        return 16ULL * (transform == Transform::C2R ? length / 2 + 1 : length);
    }
    std::uint64_t output_bytes() const {
        if (transform == Transform::C2R) return 8ULL * length;
        return 16ULL * (transform == Transform::R2C ? length / 2 + 1 : length);
    }
};

inline Program program(Options options) {
    const int n = options.length;
    if (n < 2 || n > 1048576)
        throw std::invalid_argument(
            "WebGPU FFT plan length must be in [2,1048576]");
    if (options.transform != Transform::C2C &&
        options.transform != Transform::R2C &&
        options.transform != Transform::C2R)
        throw std::invalid_argument(
            "WebGPU FFT transform must be C2C, R2C, or C2R");
    Program result;
    result.length = n;
    result.transform = options.transform;
    result.paired = options.paired;
    result.optimized = options.optimized;
    result.inverse = options.transform == Transform::C2C
                         ? options.inverse
                         : options.transform == Transform::C2R;
    result.small = n <= 256;
    result.bluestein = !result.small && (n & (n - 1)) != 0;
    int m = n;
    if (result.bluestein) {
        m = 1;
        while (m < 2 * n - 1) m *= 2;
    }
    result.fft_length = m;
    if (result.small)
        result.small_code =
            shader(n, options.paired, result.inverse, options.optimized);
    if (result.small && options.transform == Transform::C2C) {
        result.stages.push_back({"main", 0, 0});
        return result;
    }
    result.code = std::string(detail::arithmetic) +
                  (options.paired ? detail::paired : detail::scalar) +
                  detail::large;
    result.table.reserve(
        4 * ((result.small ? 0 : m / 2) + (result.bluestein ? n + m : 0) + 1));
    auto append = [&](std::complex<double> value) {
        const float rh = static_cast<float>(value.real()),
                    ih = static_cast<float>(value.imag());
        result.table.insert(result.table.end(),
                            {rh, ih, static_cast<float>(value.real() - rh),
                             static_cast<float>(value.imag() - ih)});
    };
    std::vector<std::complex<double>> roots;
    if (!result.small) {
        roots.reserve(m / 2);
        for (int j = 0; j < m / 2; ++j) {
            const double angle = -2 * std::numbers::pi * j / m;
            roots.emplace_back(std::cos(angle), std::sin(angle));
            append(roots.back());
        }
    }
    if (result.bluestein) {
        std::vector<std::complex<double>> kernel(m);
        for (int j = 0; j < n; ++j) {
            const auto square = (std::uint64_t(j) * j) % (2ULL * n);
            const double angle =
                (result.inverse ? 1 : -1) * std::numbers::pi * square / n;
            const std::complex<double> chirp(std::cos(angle), std::sin(angle));
            append(chirp);
            kernel[j] = std::conj(chirp);
            if (j != 0) kernel[m - j] = std::conj(chirp);
        }
        // Immutable convolution kernel only: an O(M log M) CPU setup FFT.
        for (int j = 1, k = 0; j < m; ++j) {
            int bit = m / 2;
            while (k & bit) {
                k ^= bit;
                bit /= 2;
            }
            k ^= bit;
            if (j < k) std::swap(kernel[j], kernel[k]);
        }
        for (int span = 2; span <= m; span *= 2) {
            for (int base = 0; base < m; base += span) {
                for (int j = 0; j < span / 2; ++j) {
                    const auto a = kernel[base + j];
                    const auto b =
                        kernel[base + j + span / 2] * roots[j * (m / span)];
                    kernel[base + j] = a + b;
                    kernel[base + j + span / 2] = a - b;
                }
            }
        }
        for (const auto value : kernel) append(value);
    }
    double scale = result.inverse && !result.small ? 1.0 / n : 1.0;
    if (result.bluestein) scale /= m;
    append({scale, 0.0});
    result.stages.push_back({"pack", 0,
                             result.small       ? 0u
                             : result.bluestein ? 6u
                                                : 2u});
    auto stages = [&](bool inverse) {
        if (options.optimized)
            result.stages.push_back({"block_butterfly", 64, inverse ? 1u : 0u});
        for (int span = options.optimized ? 128 : 2; span <= m; span *= 2)
            result.stages.push_back({options.optimized ? "butterfly_pair" : "butterfly",
                                     static_cast<std::uint32_t>(span),
                                     inverse ? 1u : 0u});
    };
    if (result.small)
        result.stages.push_back({"main", 0, 0});
    else
        stages(result.inverse && !result.bluestein);
    if (result.bluestein) {
        result.stages.push_back({"multiply", 0, 0});
        stages(true);
    }
    result.stages.push_back({"unpack", 0, result.bluestein ? 4u : 0u});
    return result;
}

}  // namespace webgpu_fft
