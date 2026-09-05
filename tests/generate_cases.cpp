#include <webgpu_fft/shader.hpp>

#include <fstream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::ofstream output(argv[1]);
    output << '[';
    bool first = true;
    for (int n :
         {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 30, 36, 64, 128, 256}) {
        for (bool paired : {false, true}) {
            for (bool inverse : {false, true}) {
                if (!first) output << ',';
                first = false;
                output << "{\"n\":" << n << ",\"paired\":" << paired
                       << ",\"inverse\":" << inverse << ",\"shader\":\"";
                for (char c : webgpu_fft::shader(n, paired, inverse)) {
                    if (c == '\n')
                        output << "\\n";
                    else if (c == '"' || c == '\\')
                        output << '\\' << c;
                    else
                        output << c;
                }
                output << "\"}";
            }
        }
    }
    output << ']';
    return output ? 0 : 1;
}
