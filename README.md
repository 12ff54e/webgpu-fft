# webgpu-fft

Reusable **batched complex and real FFTs on WebGPU**, for JavaScript/TypeScript
and Emscripten C++ using Emdawnwebgpu (also compatible with Dawn's C++ API).
No framework, runtime JS dependencies, readbacks, or hidden queue submissions.
Bind caller-owned input/output buffers once and record into an existing compute
pass or command encoder. Large and real transforms allocate reusable scratch
at bind time, never during execution.

This is an experimental implementation, not an official WebGPU/Dawn library.
It includes a compact path for small transforms and a multipass path for larger
and prime lengths. It does not yet tune algorithms to individual adapters.

## Contract

- Plan transform lengths: every integer **2–1,048,576**, subject to device
  buffer limits. The standalone single-shader API remains limited to 2–256.
- Lengths up to 256 use mixed-radix workgroup-local FFTs. Larger powers of two
  use multipass radix-2 FFTs; other larger lengths use Bluestein convolution
  with `M = nextPowerOfTwo(2*N-1)`. Large prime transforms are O(M log M),
  not direct quadratic DFTs. Bluestein pads internally without changing the
  requested transform grid or normalization.
- Contiguous batches, out of place. Transform kinds: `c2c`, `r2c`, `c2r`.
  No arbitrary strides.
- One complex sample is four `f32`s: `[real_hi, imag_hi, real_lo, imag_lo]`.
  Both precision modes use the same 16-byte layout. In `f32`, low inputs are
  ignored and low outputs are zero. In `paired-f32`, real/imaginary values are
  the sums of their high and low parts.
- One real sample is two `f32`s: `[hi, lo]`, **8 bytes in either precision**.
  The half-spectrum contains `floor(N/2)+1` complex samples in the same vec4
  layout as C2C. This is Hermitian spectrum packing, not a single-float buffer.
  R2C forces the imaginary DC/Nyquist components to zero. C2R ignores those
  endpoint imaginary inputs, reconstructs negative frequencies by conjugation,
  and outputs real samples. An odd-length last bin is not a Nyquist endpoint.
- Forward: `Y[k] = sum_j X[j] exp(-2πijk/N)`. Inverse uses the positive sign
  and divides by N. Batches do not mix.
- `inverse` selects direction for C2C only. Real transforms infer direction
  from their kind (`r2c` forward, `c2r` normalized inverse), ignoring `inverse`.
- Buffers require STORAGE usage; input and output must be distinct. Binding
  offsets must satisfy the device's storage-buffer alignment. Each binding
  covers exactly its packed byte count, with no inter-batch padding.
- Small-transform batch count must fit `maxComputeWorkgroupsPerDimension`.
  Larger dispatches can span two workgroup dimensions. Input/output, scratch,
  and table bindings must fit `maxStorageBufferBindingSize` and allocations
  must fit `maxBufferSize`. Split larger batches into aligned ranges.
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

The declaration file uses ambient WebGPU types: recent TypeScript DOM
libraries provide them directly; older toolchains can add `@webgpu/types` to
their `tsconfig.json` `types` list. Do not load both incompatible definitions.

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

`fft.dispatch(existingComputePass)` records the complete transform and sets
pipeline and bind group 0. `shader({length, precision, inverse})` exposes a
single-workgroup WGSL shader for N≤256. `program(options)` exposes the
GPU-independent multipass description, tables, and shader sources.
`FFTPlan.create` compiles asynchronously; do this outside hot loops.

Real-packed example (including a large prime transform):

```js
const n = 1009;
const real = device.createBuffer({size: 8*n,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
const spectrum = device.createBuffer({size: 16*(Math.floor(n/2)+1),
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
const forward = await FFTPlan.create(device,
  {length: n, transform: 'r2c', precision: 'paired-f32'});
const inverse = await FFTPlan.create(device,
  {length: n, transform: 'c2r', precision: 'paired-f32'});
const r2c = forward.bind(real, spectrum);
const c2r = inverse.bind(spectrum, real);
const commands = device.createCommandEncoder();
r2c.encode(commands);
c2r.encode(commands); // GPU-only round trip; no intermediate readback
device.queue.submit([commands.finish()]);
await device.queue.onSubmittedWorkDone();
r2c.destroy(); c2r.destroy(); forward.destroy(); inverse.destroy();
```

Keep plans and bindings for reuse. `binding.destroy()` releases its scratch
and uniforms; `plan.destroy()` releases immutable tables. Neither destroys
caller buffers. Only destroy resources after their work completes. Binding
execution after either object is destroyed is rejected. Recreate plans after
device loss. WebGPU validation errors remain the application's responsibility.

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

webgpu_fft::Plan real_plan(device, webgpu_fft::Options{
    .length = 1009, .paired = true, .transform = webgpu_fft::Transform::R2C});
// Bind real vec2 input and floor(N/2)+1 vec4 output as above.
```

Requires C++20 and modern `webgpu/webgpu_cpp.h`. Device setup and browser
lifecycle belong to the application. The synchronous C++ plan constructor
uses Dawn's normal validation/error callbacks. `shader(int length, bool paired,
bool inverse = false)` is independent of WebGPU headers and can also be used
from native tooling. CMake target `webgpu_fft` remains available without the
namespace alias. No Emscripten toolchain is needed for the native generator.
`program(Options)` in `webgpu_fft/program.hpp` exposes the same pure-host
description as JavaScript. C++ plan/binding resources use WebGPU RAII handles;
bind groups retain the resources they reference. Caller buffers are never
destroyed by the library.

## Setup, storage, and execution

Small C2C transforms retain the original zero-scratch path. Other bindings
own two `16*M*batchCount`-byte complex scratch buffers and one 32-byte uniform
buffer per generic stage. For small real transforms M=N; for large powers of
two M=N; for Bluestein M is the padded convolution length above. A binding can
be encoded repeatedly in a single command stream, but must not be used for
overlapping execution on independent queues/streams because its scratch is
shared. Create separate bindings for independent in-flight work.

Plans own immutable root/chirp/convolution tables, stored in GPU buffers, not
giant WGSL arrays. The table uses approximately `16*(M/2+N+M+1)` bytes for
Bluestein, `16*(M/2+1)` for large powers of two, and 16 bytes for small real
transforms. Setup computes an O(M log M) **CPU double FFT of the immutable
chirp kernel only**; all transforms of caller data run on GPU. Table generation
reduces integer chirp phases modulo 2N before trigonometry. Table and uniform
uploads happen at plan/bind time. No allocations, uploads, readbacks, or queue
submissions happen inside `encode`/`dispatch`.

## Correctness and benchmarking

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
npm ci
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
Open `/extended.html` for independent packed-real/large/prime tests and GPU-only
round trips. These also exercise the C++ multipass generator through a separate
browser executor. Numerical gates are test bounds, not universal guarantees.

For C++ wrapper header validation with Emscripten:

```sh
em++ -std=c++20 --use-port=emdawnwebgpu -Iinclude -fsyntax-only tests/compile_plan.cpp
```

The actual C++ `Plan::bind`/`Binding::encode` browser smoke test also covers
large prime and power-of-two transforms, real packing, both precisions, and
GPU-only round trips. After activating Emscripten:

```sh
bash scripts/build_cpp_runtime.sh build
```

Open `/cpp_runtime.html` on the same local server. All browser interaction is
in separate JavaScript files; the C++ test contains no embedded JavaScript.

Open `/benchmark.html` and click Run. It compares FFT with a same-precision
direct DFT using identical input/layout, reporting median and range across
seven samples after warmup. Timing uses queue-completion wall time (including
encoding/submission), **not GPU timestamp queries**; compilation, allocation,
uploads, and readbacks are excluded. Do not run other GPU work concurrently.
URL parameters choose length, batches, and repetitions. Results are hardware
and browser dependent; correctness tests make no timing claims.

The numerical kernels in `shaders/` are shared by the two language frontends.
After editing them run `npm run generate`; generated C++/JS embeddings are
checked in, so consumers need no generator. Stage emission is intentionally
small in each language and is tested independently against the same CPU oracle.
Measured adapter-specific results, including a case where paired-f32 FFT is
slower than direct DFT, are recorded in [docs/qualification.md](docs/qualification.md).

## Limits and roadmap

No arbitrary strides, in-place operation, lengths above 1,048,576, convolution
helpers, or automatic batching planner yet. Real transforms currently use
full-complex internal arithmetic; real packing reduces external storage, but
does not yet halve the internal FFT work. No claim of exhaustive adapter/backend
conformance. Paired precision needs additional arithmetic compared with f32.
Start with f32 if your error
budget allows it. Contributions should include correctness cases and
separate, reproducible benchmarks.

MIT licensed. Originally developed as a standalone dependency for cuMES.
