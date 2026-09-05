#include <webgpu_fft/program.hpp>
#include <webgpu_fft/shader.hpp>

#include <fstream>
#include <iomanip>
#include <string_view>

namespace {
void write_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (char c : value) {
        if (c == '\n')
            output << "\\n";
        else if (c == '"' || c == '\\')
            output << '\\' << c;
        else
            output << c;
    }
    output << '"';
}

void write_program(std::ostream& output, const webgpu_fft::Program& program) {
    output << "{\"length\":" << program.length
           << ",\"fft_length\":" << program.fft_length
           << ",\"transform\":" << static_cast<int>(program.transform)
           << ",\"paired\":" << program.paired
           << ",\"inverse\":" << program.inverse
           << ",\"small\":" << program.small
           << ",\"optimized\":" << program.optimized
           << ",\"bluestein\":" << program.bluestein << ",\"code\":";
    write_string(output, program.code);
    output << ",\"small_code\":";
    write_string(output, program.small_code);
    output << ",\"table\":[";
    bool first = true;
    for (float value : program.table) {
        if (!first) output << ',';
        first = false;
        output << value;
    }
    output << "],\"stages\":[";
    first = true;
    for (const auto& stage : program.stages) {
        if (!first) output << ',';
        first = false;
        output << "{\"entry_point\":";
        write_string(output, stage.entry_point);
        output << ",\"span\":" << stage.span << ",\"flags\":" << stage.flags
               << '}';
    }
    output << "]}";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && (argc != 3 || std::string_view(argv[2]) != "--extended"))
        return 1;
    std::ofstream output(argv[1]);
    output.imbue(std::locale::classic());
    output << std::setprecision(9);
    output << '[';
    bool first = true;
    if (argc == 3) {
        for (int length : {3, 36, 257, 509, 1000, 1024, 4093}) {
            for (bool paired : {false, true}) {
                for (const auto transform :
                     {webgpu_fft::Transform::C2C, webgpu_fft::Transform::R2C,
                      webgpu_fft::Transform::C2R}) {
                    for (bool inverse : {false, true}) {
                        if (inverse && transform != webgpu_fft::Transform::C2C)
                            continue;
                        if (!first) output << ',';
                        first = false;
                        write_program(
                            output, webgpu_fft::program(
                                        {length, paired, inverse, transform}));
                    }
                }
            }
        }
        output << ']';
        return output ? 0 : 1;
    }
    for (int n :
         {2,  3,  4,  5,  6,  7,  8,  9,   11,  12,  16,  17,  18,  24,  27, 30,
          36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256}) {
        for (bool paired : {false, true}) {
            for (bool inverse : {false, true}) {
                for (bool optimized : {false, true}) {
                    if (!first) output << ',';
                    first = false;
                    output << "{\"n\":" << n << ",\"paired\":" << paired
                           << ",\"inverse\":" << inverse
                           << ",\"optimized\":" << optimized
                           << ",\"shader\":\"";
                    for (char c :
                         webgpu_fft::shader(n, paired, inverse, optimized)) {
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
    }
    output << ']';
    return output ? 0 : 1;
}
