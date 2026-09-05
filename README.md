# webgpu-fft

Small, reusable **batched complex FFTs on WebGPU**, for JavaScript/TypeScript
and Emscripten C++ using Emdawnwebgpu (also compatible with Dawn's C++ API).
No framework, runtime JS dependencies, buffer allocation, readbacks, or hidden
queue submissions. Bind caller-owned GPU buffers once and record into an
existing compute pass or command encoder.

This is an experimental implementation, not an official WebGPU/Dawn library.
It targets many small transforms, not large audio/image FFTs.

## Contract

- Transform lengths: integers **2–256**, including non-powers of two (e.g. 36).
- Mixed-radix decimation in frequency, one workgroup per transform. Prime
  factors use direct butterflies; large prime lengths are supported but slow.
- Contiguous batches, out of place, complex-to-complex. No padding or strides.
- One complex sample is four `f32`s: `[real_hi, imag_hi, real_lo, imag_lo]`.
  Both precision modes use the same 16-byte layout. In `f32`, low inputs are
  ignored and low outputs are zero. In `paired-f32`, real/imaginary values are
  the sums of their high and low parts.
- Forward: `Y[k] = sum_j X[j] exp(-2πijk/N)`. Inverse uses the positive sign
  and divides by N. Batches do not mix.
- Buffers require STORAGE usage; input and output must be distinct. Binding
  offsets must satisfy the device's storage-buffer alignment. Each binding
  covers exactly `16 * length * batchCount` bytes.
- Batch count must fit `maxComputeWorkgroupsPerDimension`; bindings must fit
  `maxStorageBufferBindingSize`. Split larger batches into aligned ranges.
- Paired-f32 is **not IEEE binary64**: it keeps the f32 exponent range and
  roughly 48 significant bits for well-scaled arithmetic. Subnormals,
  overflow, and ill-conditioned cancellation are not promised binary64
  behavior. Explicit per-lane workgroup atomic round trips prevent shader
  compiler reassociation from destroying the error-free arithmetic.

## JavaScript / TypeScript

Import directly from a checkout, or use a Git dependency (not published to npm):

```sh
npm install github:12ff54e/webgpu-fft
# TypeScript projects whose toolchain does not include WebGPU declarations:
npm install --save-dev @webgpu/types
```

```js
import {FFTPlan} from '@12ff54e/webgpu-fft';
// Browser without a bundler: import {FFTPlan} from './webgpu-fft/src/index.js';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const length = 36, batchCount = 100;
const size = 16 * length * batchCount;
const input = device.createBuffer({size,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
const output = device.createBuffer({size,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
// Upload or produce input on the GPU before encoding the FFT.
const plan = await FFTPlan.create(device,
  {length, precision: 'paired-f32', inverse: false});
const fft = plan.bind(input, output, {batchCount});
const encoder = device.createCommandEncoder();
fft.encode(encoder);
// Other GPU passes can consume output here without a readback.
device.queue.submit([encoder.finish()]);
```

`fft.dispatch(existingComputePass)` avoids creating a pass and sets pipeline
and bind group 0. `shader({length, precision, inverse})` exposes WGSL directly
for custom integration. `FFTPlan.create` compiles asynchronously; do this
outside hot loops. Keep plans and bindings for reuse. Destroy data buffers
only after their work completes; recreate plans after device loss. WebGPU
validation errors/device loss remain the application's responsibility.

## Emscripten C++ / Emdawnwebgpu

```cmake
add_subdirectory(deps/webgpu-fft)
target_link_libraries(my_app PRIVATE webgpu_fft::webgpu_fft)
target_compile_options(my_app PRIVATE "--use-port=emdawnwebgpu")
target_link_options(my_app PRIVATE "--use-port=emdawnwebgpu")
```

```cpp
#include <webgpu_fft/plan.hpp>

// device, input, output, and encoder are caller-owned wgpu objects.
webgpu_fft::Plan plan(device, 36, true);  // paired-f32, forward
auto fft = plan.bind(input, output, 100);
fft.encode(encoder);
// Caller finishes and submits encoder. No EM_JS is needed.
```

Requires C++20 and modern `webgpu/webgpu_cpp.h`. Device setup and browser
lifecycle belong to the application. The synchronous C++ plan constructor
uses Dawn's normal validation/error callbacks. `shader(int length, bool paired,
bool inverse = false)` is independent of WebGPU headers and can also be used
from native tooling. CMake target `webgpu_fft` remains available without the
namespace alias. No Emscripten toolchain is needed for the native generator.

## Correctness and benchmarking

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
npm test
python3 -m http.server 8080 --directory build
```

Open `http://localhost:8080/` in a WebGPU-enabled browser. The suite tests both
C++-generated and JS-generated shaders against an independent CPU double DFT:
17 lengths, both precisions, both directions, four batches, nonzero low parts,
and nonzero aligned offsets (136 cases). Scaled component-error gates are
`1e-4` for f32 and `3e-11` for paired-f32; these are test bounds, not universal
error guarantees. CPU DFT error also contributes to the measured difference.
`npm test` checks generated sources and host-side API rejection paths.

For C++ wrapper header validation with Emscripten:

```sh
em++ -std=c++20 --use-port=emdawnwebgpu -Iinclude -fsyntax-only tests/compile_plan.cpp
```

Open `/benchmark.html` and click Run. It compares FFT with a same-precision
direct DFT using identical input/layout, reporting median and range across
seven samples after warmup. Timing uses queue-completion wall time (including
encoding/submission), **not GPU timestamp queries**; compilation, allocation,
uploads, and readbacks are excluded. Do not run other GPU work concurrently.
URL parameters choose length, batches, and repetitions. Results are hardware
and browser dependent; correctness tests deliberately make no timing claims.

The numerical kernels in `shaders/` are shared by the two language frontends.
After editing them run `npm run generate`; generated C++/JS embeddings are
checked in, so consumers need no generator. Stage emission is intentionally
small in each language and is tested independently against the same CPU oracle.

## Limits and roadmap

No real-packed transforms, arbitrary strides, in-place operation, lengths
above 256, convolution helpers, or automatic batching planner yet. No claim
of exhaustive adapter/backend conformance. Strict paired arithmetic is
deliberately more expensive than ordinary f32. Start with f32 if your error
budget allows it. Contributions should include correctness cases and
separate, reproducible benchmarks.

MIT licensed. Originally developed as a standalone dependency for cuMES.
