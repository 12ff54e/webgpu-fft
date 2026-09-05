#include <webgpu_fft/program.hpp>
#include <webgpu_fft/shader.hpp>

#include <iostream>
#include <locale>
#include <stdexcept>

struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};

int main() {
    for (int length : {-1, 0, 1, 257}) {
        bool rejected = false;
        try {
            (void)webgpu_fft::shader(length, true);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 1;
    }
    const auto reference = webgpu_fft::shader(36, true);
    const auto original = std::locale();
    std::locale::global(std::locale(original, new CommaDecimal));
    const auto localized = webgpu_fft::shader(36, true);
    std::locale::global(original);
    if (localized != reference) return 2;
    const auto generic = webgpu_fft::shader(36, true, false, false);
    if (generic.find("radix_three") != std::string::npos ||
        generic.find("for(var p=0u") == std::string::npos ||
        reference.find("radix_three") == std::string::npos)
        return 9;
    if (webgpu_fft::program(
            {36, true, false, webgpu_fft::Transform::C2C, false})
            .small_code != generic)
        return 10;
    for (int length = 2; length <= 256; ++length) {
        for (bool paired : {false, true}) {
            for (bool inverse : {false, true}) {
                for (bool optimized : {false, true}) {
                    const auto code =
                        webgpu_fft::shader(length, paired, inverse, optimized);
                    if (code.find("@workgroup_size(" + std::to_string(length) +
                                  ")") == std::string::npos)
                        return 3;
                }
            }
        }
    }
    for (int length : {-1, 0, 1, 1048577}) {
        bool rejected = false;
        try {
            (void)webgpu_fft::program({length});
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 4;
    }
    for (int length : {3, 36, 257, 1000, 1024, 65537}) {
        for (const auto kind :
             {webgpu_fft::Transform::C2C, webgpu_fft::Transform::R2C,
              webgpu_fft::Transform::C2R}) {
            const auto description =
                webgpu_fft::program({length, true, false, kind});
            if (description.stages.empty()) return 5;
            const auto real_bytes = 8ULL * length;
            const auto half_bytes = 16ULL * (length / 2 + 1);
            if (kind == webgpu_fft::Transform::R2C &&
                (description.input_bytes() != real_bytes ||
                 description.output_bytes() != half_bytes ||
                 description.inverse))
                return 6;
            if (kind == webgpu_fft::Transform::C2R &&
                (description.input_bytes() != half_bytes ||
                 description.output_bytes() != real_bytes ||
                 !description.inverse))
                return 7;
            if (description.bluestein &&
                (description.fft_length < 2 * length - 1 ||
                 (description.fft_length & (description.fft_length - 1)) != 0))
                return 8;
        }
    }
    std::cout << "Native generator validation PASS\n";
}
