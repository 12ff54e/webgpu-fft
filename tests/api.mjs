import assert from 'node:assert/strict';
import {FFTPlan, shader} from '../src/index.js';
for (const length of [0, 1, 257, 4.5, NaN])
  assert.throws(() => shader({length}), RangeError);
assert.throws(() => shader({length: 36, precision: 'double'}), TypeError);
assert.throws(() => shader({length: 36, inverse: 1}), TypeError);
for (const length of [2, 7, 36, 256])
  assert.match(shader({length}), new RegExp(`@workgroup_size\\(${length}\\)`));
const device = {
  limits: {maxComputeWorkgroupsPerDimension: 65535, minStorageBufferOffsetAlignment: 256,
    maxStorageBufferBindingSize: 134217728},
  createShaderModule: ({code}) => ({code}),
  createComputePipelineAsync: async () => ({getBindGroupLayout: () => ({})}),
  createBindGroup: descriptor => descriptor,
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
console.log('JS API validation PASS');
