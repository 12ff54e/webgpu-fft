# Qualification record

2026-09-06, Windows Chrome 152, RTX 3060 Ti, Dawn D3D12/DXC. This is one
adapter/backend qualification, not a guarantee for all WebGPU implementations.

- 136 GPU/CPU comparisons passed: 17 lengths (2,3,4,5,6,7,8,9,11,12,16,17,
  30,36,64,128,256), both directions, both precisions, both C++ and JS shader
  emitters. Four contiguous sequences and 256-byte binding offsets per case.
- N=36 paired-f32 scaled maximum component error: forward `7.604e-14`,
  inverse `2.836e-15`. Both emitters produced the same measured errors.
- Largest reported paired-f32 forward error was `1.615e-12` for N=256;
  the independent CPU double direct DFT has its own numerical error.
- Native CMake generator/CTest, generated-kernel consistency, JS API rejection
  tests, TypeScript strict consumer check, and Emdawnwebgpu C++ header syntax
  check passed.

## Benchmark

`benchmark.html?n=36&batch=4096&repeats=10`. Seven samples after three warmup
passes. Times include command encoding/submission and queue completion, but
exclude allocation, pipeline compilation, upload, and readback. Other solver
GPU work was stopped. Browser does not expose an adapter description to this
page; adapter identification above comes from the established Chrome session.

| Algorithm | Precision | Median ms/pass | Range ms/pass |
| --- | --- | ---: | ---: |
| FFT | f32 | 0.553 | 0.386–0.589 |
| Direct DFT | f32 | 1.853 | 1.816–1.922 |
| FFT | paired-f32 | 2.786 | 2.602–3.019 |
| Direct DFT | paired-f32 | 1.696 | 1.474–1.768 |

The scalar FFT was about 3.35× faster than this direct DFT, but the paired
FFT was about 1.64× slower. Small sizes, strict rounding barriers, compiler
behavior, and workgroup synchronization can outweigh asymptotic savings.
Do not assume FFT always wins. This standalone direct DFT loads each sequence
into workgroup memory; it is not the original cuMES toroidal projection and
these numbers do not measure end-to-end solver speed or packing overhead.
No timing claim is made for other sizes or adapters.
