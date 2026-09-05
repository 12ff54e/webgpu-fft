# Qualification record

## 0.2 real-packed and arbitrary-length extension

Tested on the same Windows Chrome / RTX 3060 Ti session described below.
Both public language APIs preserve the existing small C2C path and add real
packing plus device-resident multipass radix-2/Bluestein transforms. All 197
extended browser cases passed.

- Expanded JS tests cover real/complex transforms and both precisions through
  lengths 65,536 and 65,537, with independent reduced-phase CPU DFT checks,
  all-point round trips, two batches, aligned offsets, output guards, and
  repeated binding encodes. Separate C++-emitted programs test through 4093.
- Maximum-length power-of-two (1,048,576) and adjacent non-power-of-two
  (1,048,575) support was additionally checked with GPU round trips in both
  precisions. The paired Bluestein round-trip error was `1.718e-15`.
  The paired component-error gate is `3e-11` across the extended suite; scalar
  gate `3e-4`. These gates are test thresholds, not worst-case error proofs.
- Selected paired scaled errors: N=65,537 C2C forward `2.452e-13`, R2C
  `8.678e-13`; N=65,536 C2C forward `8.528e-14`, R2C `8.055e-13`.
  C++-generated N=4093 R2C error was `5.781e-13`.
- A 65,536-point, 65-batch f32 transform exercised two-dimensional dispatch
  beyond 65,535 workgroups; scaled error `8.534e-8`.
- Eight actual **Emdawnwebgpu C++ runtime** cases exercise `Plan::bind` and
  `Binding::encode`, not just emitted shaders: N=257/1024, real and complex,
  f32 and paired-f32, two batches and aligned input/output offsets. Forward
  transforms are compared with CPU DFT bins and GPU forward/inverse round trips
  compare every sample. Largest paired scaled error `1.716e-13`; scalar
  `2.856e-6`. JavaScript is only used for the browser shell/status bridge.

The immutable Bluestein convolution kernel is prepared with a host double
FFT during setup. All transforms of caller data, packing, spectral products,
and normalization execute on GPU, without per-execution uploads/readbacks.

## 0.1 small-transform baseline

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
