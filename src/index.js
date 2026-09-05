import {arithmetic, paired, scalar} from './source.js';

function options({length, precision = 'f32', inverse = false}) {
  if (!Number.isInteger(length) || length < 2 || length > 256)
    throw new RangeError('FFT length must be an integer in [2,256]');
  if (precision !== 'f32' && precision !== 'paired-f32')
    throw new TypeError('FFT precision must be f32 or paired-f32');
  if (typeof inverse !== 'boolean') throw new TypeError('FFT inverse must be boolean');
  return {length, precision, inverse};
}

/** Emit WGSL independently of the browser; no WebGPU globals are accessed. */
export function shader(config) {
  const {length: n, precision, inverse} = options(config);
  const radices = [];
  for (let r = 2, remaining = n; remaining > 1; ++r)
    while (remaining % r === 0) {radices.push(r); remaining /= r;}
  const f = value => `${Math.fround(value).toPrecision(9)}f`;
  let code = `const N = ${n}u;
@group(0) @binding(0) var<storage,read> input: array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output: array<vec4f>;
var<workgroup> values: array<vec4f,${n}>;
var<workgroup> rounding: array<atomic<u32>,${n}>;
${arithmetic}
const roots = array<vec4f,${n}>(\n`;
  for (let k = 0; k < n; ++k) {
    const angle = (inverse ? 2 : -2) * Math.PI * k / n;
    const r = Math.cos(angle), i = Math.sin(angle);
    code += `vec4f(${f(r)},${f(i)},${f(r - Math.fround(r))},${f(i - Math.fround(i))}),\n`;
  }
  code += `);\n${precision === 'paired-f32' ? paired : scalar}
@compute @workgroup_size(${n})
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
let base=group.x*N;values[lane]=input[base+lane];workgroupBarrier();\n`;
  let span = n;
  for (const radix of radices) {
    const step = span / radix;
    code += `{let block=lane/${span}u;let j=lane%${step}u;let q=(lane%${span}u)/${step}u;
var sum=vec4f(0.0);for(var p=0u;p<${radix}u;p++){
let v=values[block*${span}u+j+p*${step}u];
sum=cadd(sum,cmul(v,roots[(p*q*${n / radix}u)%N],lane),lane);}
let value=cmul(sum,roots[(q*j*${n / span}u)%N],lane);workgroupBarrier();values[lane]=value;workgroupBarrier();}\n`;
    span = step;
  }
  code += 'var k=lane;var index=0u;\n';
  span = n;
  for (const radix of radices) {
    span /= radix;
    code += `index+=(k%${radix}u)*${span}u;k/=${radix}u;\n`;
  }
  code += inverse
    ? `output[base+lane]=cmul(values[index],vec4f(${f(1 / n)},0.0,${f(1 / n - Math.fround(1 / n))},0.0),lane);\n`
    : 'output[base+lane]=values[index];\n';
  return code + '}\n';
}

/** Reusable compiled FFT. No buffer allocation, submission, or readback. */
export class FFTPlan {
  static async create(device, config) {
    const normalized = options(config);
    const module = device.createShaderModule({label: 'webgpu-fft', code: shader(normalized)});
    const pipeline = await device.createComputePipelineAsync({
      label: 'webgpu-fft', layout: 'auto', compute: {module, entryPoint: 'main'},
    });
    return new FFTPlan(device, pipeline, normalized);
  }
  constructor(device, pipeline, config) {
    this.device = device;
    this.pipeline = pipeline;
    this.length = config.length;
    this.precision = config.precision;
    this.inverse = config.inverse;
  }
  /** Bind once, encode many times. Offsets are bytes; transforms are contiguous. */
  bind(input, output, {batchCount = 1, inputOffset = 0, outputOffset = 0} = {}) {
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (!Number.isInteger(batchCount) || batchCount < 1 || batchCount > limit)
      throw new RangeError(`FFT batchCount must be an integer in [1,${limit}]`);
    if (input === output) throw new TypeError('FFT requires distinct input and output buffers');
    const size = batchCount * this.length * 16;
    for (const [buffer, offset] of [[input, inputOffset], [output, outputOffset]]) {
      if (!Number.isSafeInteger(offset) || offset < 0 ||
          offset % this.device.limits.minStorageBufferOffsetAlignment !== 0)
        throw new RangeError('FFT buffer offset must be nonnegative and storage-aligned');
      if (offset + size > buffer.size || size > this.device.limits.maxStorageBufferBindingSize)
        throw new RangeError('FFT binding exceeds buffer size or device binding limit');
      if ((buffer.usage & 128) === 0) throw new TypeError('FFT buffers require STORAGE usage');
    }
    const group = this.device.createBindGroup({layout: this.pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: input, offset: inputOffset, size}},
      {binding: 1, resource: {buffer: output, offset: outputOffset, size}},
    ]});
    const pipeline = this.pipeline;
    return Object.freeze({
      dispatch(pass) {pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(batchCount);},
      encode(encoder) {const pass = encoder.beginComputePass(); this.dispatch(pass); pass.end();},
    });
  }
}
