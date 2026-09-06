#pragma once

#include <cmath>
#include <iomanip>
#include <locale>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source.hpp"

namespace webgpu_fft {

// Batched complex DIF FFT. Each workgroup owns one contiguous transform;
// complex values are vec4(real_hi, imag_hi, real_lo, imag_lo). Forward uses
// exp(-i*2*pi*k*j/N); inverse uses the positive sign and normalizes by N.
// No padding: mixed-radix stages preserve the caller's exact sample grid.
inline std::string shader(int length,
                          bool paired,
                          bool inverse = false,
                          bool optimized = true) {
    // The scalar compiler already optimizes the generic loop well. The
    // specialized branch regressed large batches; retain the exact reference.
    optimized = optimized && paired;
    if (length < 2 || length > 256)
        throw std::invalid_argument("WebGPU FFT length must be in [2,256]");
    std::vector<int> radices;
    int remaining = length;
    for (int radix = 2; remaining > 1; ++radix) {
        while (remaining % radix == 0) {
            radices.push_back(radix);
            remaining /= radix;
        }
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9);
    out << "const N = " << length << "u;\n"
        << "@group(0) @binding(0) var<storage,read> input: array<vec4f>;\n"
        << "@group(0) @binding(1) var<storage,read_write> output: "
           "array<vec4f>;\n"
        << "var<workgroup> values: array<vec4f," << length << ">;\n"
        << "var<workgroup> rounding: array<atomic<u32>," << length << ">;\n";
    out << detail::arithmetic;
    out << "const roots = array<vec4f," << length << ">(\n";
    for (int k = 0; k < length; ++k) {
        const double angle =
            (inverse ? 2.0 : -2.0) * std::numbers::pi * k / length;
        const double r = std::cos(angle), i = std::sin(angle);
        const float rh = static_cast<float>(r), ih = static_cast<float>(i);
        out << "vec4f(" << rh << "f," << ih << "f,"
            << static_cast<float>(r - rh) << "f," << static_cast<float>(i - ih)
            << "f),\n";
    }
    out << ");\n";
    out << (paired ? detail::paired : detail::scalar);
    if (optimized) {
        const double root = std::sqrt(3.0) / 2;
        const float hi = static_cast<float>(root);
        out << "const PAIRED = " << (paired ? "true" : "false") << ";\n"
            << "const INVERSE = " << (inverse ? "true" : "false") << ";\n"
            << "const SQRT_THREE_OVER_TWO = vec2f(" << hi << "f,"
            << static_cast<float>(root - hi) << "f);\n"
            << detail::optimized;
    }
    out << "@compute @workgroup_size(" << length << ")\n"
        << "fn main(@builtin(local_invocation_index) lane:u32,"
        << "@builtin(workgroup_id) group:vec3u){\n"
        << "let "
           "base=group.x*N;values[lane]=input[base+lane];workgroupBarrier();\n";
    int span = length;
    for (int radix : radices) {
        const int step = span / radix;
        if (optimized && length >= 36 && length % 2 == 0 &&
            (radix == 2 || radix == 3)) {
            // One lane owns a whole butterfly: its input/output locations
            // are disjoint from other lanes, so only a stage-end fence is
            // needed. Preserve each output's paired arithmetic ordering.
            out << "{if(lane<" << length / radix << "u){let block=lane/" << step
                << "u;let j=lane%" << step << "u;let start=block*" << span
                << "u+j;\n"
                << "let a=values[start];let b=values[start+" << step << "u];\n";
            if (radix == 2) {
                out << "let s0=cadd(a,b,lane);let s1=cadd(a,-b,lane);\n";
            } else {
                out << "let c=values[start+" << 2 * step
                    << "u];let pair=cadd(b,c,lane);let s0=cadd(a,pair,lane);\n"
                    << "let "
                       "center=cadd(a,cscale(pair,vec2f(-0.5,0.0),lane),lane);"
                       "\n"
                    << "let "
                       "skew=rotate_i(cscale(cadd(b,-c,lane),SQRT_THREE_OVER_"
                       "TWO,lane));\n"
                    << "let s1=cadd(center," << (inverse ? "" : "-")
                    << "skew,lane);let s2=cadd(center," << (inverse ? "-" : "")
                    << "skew,lane);\n";
            }
            for (int q = 0; q < radix; ++q) {
                out << "{let position=start+" << q * step
                    << "u;let value=twiddle(s" << q << ",(" << q * length / span
                    << "u*j)%N,lane);\n";
                if (step == 1) {
                    out << "var index=0u;var k=position;\n";
                    int digit_span = length, weight = 1;
                    for (int r : radices) {
                        digit_span /= r;
                        out << "index+=(k/" << digit_span << "u)*" << weight
                            << "u;k%=" << digit_span << "u;\n";
                        weight *= r;
                    }
                    if (inverse) {
                        const double scale = 1.0 / length;
                        const float hi = static_cast<float>(scale);
                        out << "output[base+index]=cscale(value,vec2f(" << hi
                            << "f," << static_cast<float>(scale - hi)
                            << "f),lane);}\n";
                    } else
                        out << "output[base+index]=value;}\n";
                } else
                    out << "values[position]=value;}\n";
            }
            out << (step == 1 ? "}}\n" : "}workgroupBarrier();}\n");
            span = step;
            continue;
        }
        if (optimized) {
            out << "{let block=lane/" << span << "u;let j=lane%" << step
                << "u;let q=(lane%" << span << "u)/" << step << "u;\n"
                << "let start=block*" << span << "u+j;\n";
            if (radix == 2) {
                out << "let sum=cadd(values[start],select(values[start+" << step
                    << "u],-values[start+" << step << "u],q!=0u),lane);\n";
            } else if (radix == 3) {
                out << "let sum=radix_three(values[start],values[start+" << step
                    << "u],values[start+" << 2 * step << "u],q,lane);\n";
            } else {
                out << "var sum=values[start];for(var p=1u;p<" << radix
                    << "u;p++){sum=cadd(sum,twiddle(values[start+p*" << step
                    << "u],(p*q*" << length / radix << "u)%N,lane),lane);}\n";
            }
            out << "let value=twiddle(sum,(q*j*" << length / span
                << "u)%N,lane);\n";
            if (step == 1) {
                // The final stage writes natural-order output directly, so it
                // needs neither an in-place read fence nor a following barrier.
                out << "var index=0u;var k=lane;\n";
                int digit_span = length, weight = 1;
                for (int digit_radix : radices) {
                    digit_span /= digit_radix;
                    out << "index+=(k/" << digit_span << "u)*" << weight
                        << "u;k%=" << digit_span << "u;\n";
                    weight *= digit_radix;
                }
                if (inverse) {
                    const double norm = 1.0 / length;
                    const float hi = static_cast<float>(norm);
                    out << "output[base+index]=cscale(value,vec2f(" << hi
                        << "f," << static_cast<float>(norm - hi)
                        << "f),lane);}\n";
                } else
                    out << "output[base+index]=value;}\n";
            } else {
                out << "workgroupBarrier();values[lane]=value;workgroupBarrier("
                       ");}\n";
            }
            span = step;
            continue;
        }
        out << "{let block=lane/" << span << "u;let j=lane%" << step
            << "u;let q=(lane%" << span << "u)/" << step << "u;\n"
            << "var sum=vec4f(0.0);for(var p=0u;p<" << radix << "u;p++){\n"
            << "let v=values[block*" << span << "u+j+p*" << step << "u];\n"
            << "sum=cadd(sum,cmul(v,roots[(p*q*" << length / radix
            << "u)%N],lane),lane);}\n"
            << "let value=cmul(sum,roots[(q*j*" << length / span
            << "u)%N],lane);workgroupBarrier();values[lane]=value;"
            << "workgroupBarrier();}\n";
        span = step;
    }
    if (optimized) {
        out << "}\n";
        return out.str();
    }
    out << "var k=lane;var index=0u;\n";
    span = length;
    for (int radix : radices) {
        span /= radix;
        out << "index+=(k%" << radix << "u)*" << span << "u;k/=" << radix
            << "u;\n";
    }
    if (inverse) {
        const double norm = 1.0 / length;
        const float hi = static_cast<float>(norm);
        out << "output[base+lane]=cmul(values[index],vec4f(" << hi << "f,0.0,"
            << static_cast<float>(norm - hi) << "f,0.0),lane);\n";
    } else {
        out << "output[base+lane]=values[index];\n";
    }
    out << "}\n";
    return out.str();
}

}  // namespace webgpu_fft
