# Qualification record

## Specialized small FFT kernels

The generic generator remains selectable with `optimized=false`. Its original
twiddle generation (including tiny trigonometric values at integer multiples
of π), reduction order, and rounding barriers are retained. The optimized path
specializes radix 2 and 3, uses exact identity/sign/quarter-turn twiddles, and
writes the last stage directly in natural order, removing two workgroup
barriers. Paired operations retain their strict f32 rounding barriers.

All 496 small GPU/CPU comparisons and 248 direct same-input comparisons between
generic and specialized variants passed. Both emitters, precisions, directions,
and 31 lengths are covered, including powers of three through 243 and mixed
radix combinations. The 197 real/large/upper-limit regressions also passed.
The actual Emdawnwebgpu C++ runtime test was expanded from eight to 12 cases
with N=36 real/C2C in both precisions; all passed. N=36 paired scaled errors in
the small suite: forward `7.280e-14`,
inverse `2.492e-15`. Different arithmetic ordering is expected to change low
bits; no bit-identical iterative-solver trajectory is promised.

Initial same-input timing: N=36, batch=4096, 30 repetitions/sample, seven
samples, foreground Chrome on the RTX 3060 Ti. Queue-completion wall time,
excluding setup/upload/readback:

| Kernel | Precision | Median ms/pass | Range ms/pass |
| --- | --- | ---: | ---: |
| Optimized FFT | f32 | 0.333 | 0.300–0.352 |
| Generic FFT | f32 | 0.336 | 0.328–0.369 |
| Optimized FFT | paired-f32 | 0.313 | 0.289–0.382 |
| Generic FFT | paired-f32 | 0.562 | 0.500–0.600 |

The paired FFT improved by about 1.80× in this comparison; scalar timings were
essentially unchanged. These same-session measurements are not comparable
with older runs under different browser/queue scheduling conditions.

The larger solver-shaped workload exposed a scalar regression hidden by the
smaller benchmark. N=36, batch=31,680 (`20*99*16`), 20 repetitions/sample,
seven samples, same timing convention:

| Kernel candidate | Precision | Median ms/pass | Range ms/pass |
| --- | --- | ---: | ---: |
| Specialized FFT | paired-f32 | 1.851 | 1.799–1.875 |
| Generic FFT | paired-f32 | 3.736 | 3.692–3.798 |
| Direct DFT | paired-f32 | 10.152 | 10.103–10.246 |
| Specialized FFT (rejected) | f32 | 2.251 | 2.199–2.348 |
| Generic FFT | f32 | 0.299 | 0.248–0.302 |

The paired specialization improved throughput by about 2.02× at the solver's
batch size. The scalar specialization from commit `aa75fd7` was rejected:
current f32 generation returns the **exact generic shader** regardless of the
optimization flag. Native and JS source-equality tests check that policy for
every length 2–256 and both directions. Existing generic GPU cases therefore
cover the retained scalar implementation. This is a precision-specific choice,
not a claim that the specialized arithmetic is universally faster.

## 0.2 real-packed and arbitrary-length extension

Tested on the same Windows Chrome / RTX 3060 Ti session described below.
Both public language APIs preserve the existing small C2C path and add real
packing plus device-resident multipass radix-2/Bluestein transforms. All 197
extended browser cases passed. All 136 original small-transform browser cases
also passed again after the extension.

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

### Large-transform benchmark smoke

128 contiguous C2C batches, 50 repetitions per sample, seven samples after
warmup, same Chrome/RTX session. Queue-completion wall timing includes command
encoding/submission, but excludes setup, uploads, and readbacks.

| Length / path | Precision | Median ms/pass | Range ms/pass |
| --- | --- | ---: | ---: |
| 1024 / radix-2 | f32 | 0.198 | 0.159–2.138 |
| 1024 / radix-2 | paired-f32 | 0.377 | 0.361–0.381 |
| 1009 / Bluestein | f32 | 0.660 | 0.640–3.379 |
| 1009 / Bluestein | paired-f32 | 1.280 | 1.278–1.319 |

Some scalar samples had substantial browser/queue scheduling outliers. These
are reproducibility records, not GPU timestamp measurements or universal
performance claims. Preliminary five-repetition runs were too noisy for
comparison and are omitted. No large direct-DFT comparison was performed.

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
