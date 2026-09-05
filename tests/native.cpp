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
    for (int length = 2; length <= 256; ++length) {
        for (bool paired : {false, true}) {
            for (bool inverse : {false, true}) {
                const auto code = webgpu_fft::shader(length, paired, inverse);
                if (code.find("@workgroup_size(" + std::to_string(length) +
                              ")") == std::string::npos)
                    return 3;
            }
        }
    }
    std::cout << "Native generator validation PASS\n";
}
