import assert from 'node:assert/strict';
import {FFTPlan, program, shader} from '../src/index.js';
for (const length of [0, 1, 257, 4.5, NaN])
  assert.throws(() => shader({length}), RangeError);
assert.throws(() => shader({length: 36, precision: 'double'}), TypeError);
assert.throws(() => shader({length: 36, inverse: 1}), TypeError);
assert.throws(() => shader({length: 36, optimized: 1}), TypeError);
assert.throws(() => program({length: 1024, optimized: 1}), TypeError);
const generic=shader({length:36,precision:'paired-f32',optimized:false});
const optimized=shader({length:36,precision:'paired-f32'});
assert.match(generic,/for\(var p=0u/);
assert.doesNotMatch(generic,/radix_three/);
assert.match(optimized,/radix_three/);
assert.equal(program({length:36,precision:'paired-f32',optimized:false}).small_code,generic);
assert.equal((generic.match(/workgroupBarrier\(\)/g)||[]).length-(optimized.match(/workgroupBarrier\(\)/g)||[]).length,5);
assert.match(optimized,/if\(lane</);
for(const length of [16,27,32,34,243])assert.doesNotMatch(shader({length,precision:'paired-f32'}),/if\(lane</);
for(let n=2;n<=256;n++)for(const inverse of [false,true])
  assert.equal(shader({length:n,inverse,optimized:true}),shader({length:n,inverse,optimized:false}));
assert.throws(() => shader({length: 36, transform: 'r2c'}), TypeError);
for (const length of [0, 1, 1048577, 2.5, NaN])
  assert.throws(() => program({length}), RangeError);
assert.throws(() => program({length: 257, transform: 'bad'}), TypeError);
for (const n of [3, 36, 257, 1024]) {
  const real = program({length: n, transform: 'r2c', inverse: true});
  assert.equal(real.input_bytes, 8 * n);
  assert.equal(real.output_bytes, 16 * (Math.floor(n / 2) + 1));
  assert.equal(real.inverse, false);
  const inverse = program({length: n, transform: 'c2r', inverse: false});
  assert.equal(inverse.input_bytes, real.output_bytes);
  assert.equal(inverse.output_bytes, real.input_bytes);
  assert.equal(inverse.inverse, true);
  if (real.bluestein) {
    assert.ok(real.fft_length >= 2 * n - 1);
    assert.equal(real.fft_length & (real.fft_length - 1), 0);
  }
}
for (const length of [2, 7, 36, 256])
  assert.match(shader({length}), new RegExp(`@workgroup_size\\(${length}\\)`));
const device = {
  limits: {maxComputeWorkgroupsPerDimension: 65535, minStorageBufferOffsetAlignment: 256,
    maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456},
  createShaderModule: ({code}) => ({code}),
  createComputePipelineAsync: async () => ({getBindGroupLayout: () => ({})}),
  createBindGroup: descriptor => descriptor,
  createBindGroupLayout: descriptor => descriptor,
  createPipelineLayout: descriptor => descriptor,
  createBuffer: descriptor => ({...descriptor, destroy() {this.destroyed = true;}}),
  queue: {writeBuffer() {}},
};
const plan = await FFTPlan.create(device, {length: 36});
const input = {size: 4096, usage: 128}, output = {size: 4096, usage: 128};
assert.throws(() => plan.bind(input, input), TypeError);
assert.throws(() => plan.bind(input, output, {batchCount: 0}), RangeError);
assert.throws(() => plan.bind(input, output, {inputOffset: 1}), RangeError);
assert.throws(() => plan.bind(input, output, {batchCount: 8}), RangeError);
assert.throws(() => plan.bind({...input, usage: 0}, output), TypeError);
let dispatched = 0;
plan.bind(input, output, {batchCount: 4, inputOffset: 256}).encode({beginComputePass: () => ({
  setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(n) {dispatched = n;}, end() {},
})});
assert.equal(dispatched, 4);
const real = await FFTPlan.create(device, {length: 257, transform: 'r2c'});
assert.throws(() => real.bind({size: 8 * 257 - 4, usage: 128}, output), RangeError);
assert.throws(() => real.bind(input, {size: 16 * 129 - 4, usage: 128}), RangeError);
const binding = real.bind(input, output);
binding.destroy();
assert.throws(() => binding.encode({beginComputePass() {return {};}}), /destroyed/);
real.destroy();
assert.throws(() => real.bind(input, output), /destroyed/);
const smallLimits = {...device, limits: {...device.limits, maxStorageBufferBindingSize: 8192}};
await assert.rejects(() => FFTPlan.create(smallLimits, {length: 4093}), RangeError);
const grid = {...device, limits: {...device.limits, maxComputeWorkgroupsPerDimension: 128}};
const large = await FFTPlan.create(grid, {length: 4096});
const largeBinding = large.bind({size: 1 << 20, usage: 128}, {size: 1 << 20, usage: 128}, {batchCount: 3});
const dispatches = [];
largeBinding.dispatch({setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(x, y) {dispatches.push([x, y]);}});
assert.ok(dispatches.length > 1);
assert.ok(dispatches.every(([x, y]) => x === 128 && y === 2));
assert.equal(large.description.stages.filter(s=>s.entry_point==='block_butterfly').length,1);
assert.equal(large.description.stages.length,9);
const reference=program({length:4096,optimized:false});
assert.equal(reference.stages.length,14);
assert.equal(reference.stages.filter(s=>s.entry_point==='butterfly').length,12);
largeBinding.destroy(); large.destroy();
console.log('JS API validation PASS');
