import {arithmetic, paired, scalar, optimized as optimizedMath} from './source.js';

function options({length, precision = 'f32', inverse = false, transform = 'c2c', optimized = true}) {
  if (!Number.isInteger(length) || length < 2 || length > 256)
    throw new RangeError('FFT length must be an integer in [2,256]');
  if (precision !== 'f32' && precision !== 'paired-f32')
    throw new TypeError('FFT precision must be f32 or paired-f32');
  if (typeof inverse !== 'boolean') throw new TypeError('FFT inverse must be boolean');
  if (typeof optimized !== 'boolean') throw new TypeError('FFT optimized must be boolean');
  if (transform !== 'c2c') throw new TypeError('Single-shader API supports c2c only; use FFTPlan for real transforms');
  // Scalar specialization regressed large batches; keep its exact reference.
  return {length, precision, inverse, optimized: optimized && precision === 'paired-f32'};
}

/** Emit WGSL independently of the browser; no WebGPU globals are accessed. */
export function shader(config) {
  const {length: n, precision, inverse, optimized} = options(config);
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
  code += `);\n${precision === 'paired-f32' ? paired : scalar}`;
  if(optimized){
    const root=Math.sqrt(3)/2;
    code+=`const PAIRED = ${precision==='paired-f32'};\nconst INVERSE = ${inverse};\nconst SQRT_THREE_OVER_TWO = vec2f(${f(root)},${f(root-Math.fround(root))});\n${optimizedMath}`;
  }
  code += `
@compute @workgroup_size(${n})
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
let base=group.x*N;values[lane]=input[base+lane];workgroupBarrier();\n`;
  let span = n;
  for (const radix of radices) {
    const step = span / radix;
    if(optimized){
      code+=`{let block=lane/${span}u;let j=lane%${step}u;let q=(lane%${span}u)/${step}u;\nlet start=block*${span}u+j;\n`;
      if(radix===2)code+=`let sum=cadd(values[start],select(values[start+${step}u],-values[start+${step}u],q!=0u),lane);\n`;
      else if(radix===3)code+=`let sum=radix_three(values[start],values[start+${step}u],values[start+${2*step}u],q,lane);\n`;
      else code+=`var sum=values[start];for(var p=1u;p<${radix}u;p++){sum=cadd(sum,twiddle(values[start+p*${step}u],(p*q*${n/radix}u)%N,lane),lane);}\n`;
      code+=`let value=twiddle(sum,(q*j*${n/span}u)%N,lane);\n`;
      if(step===1){
        code+='var index=0u;var k=lane;\n';let digitSpan=n,weight=1;
        for(const r of radices){digitSpan/=r;code+=`index+=(k/${digitSpan}u)*${weight}u;k%=${digitSpan}u;\n`;weight*=r;}
        code+=inverse?`output[base+index]=cscale(value,vec2f(${f(1/n)},${f(1/n-Math.fround(1/n))}),lane);}\n`:'output[base+index]=value;}\n';
      }else code+='workgroupBarrier();values[lane]=value;workgroupBarrier();}\n';
      span=step;continue;
    }
    code += `{let block=lane/${span}u;let j=lane%${step}u;let q=(lane%${span}u)/${step}u;
var sum=vec4f(0.0);for(var p=0u;p<${radix}u;p++){
let v=values[block*${span}u+j+p*${step}u];
sum=cadd(sum,cmul(v,roots[(p*q*${n / radix}u)%N],lane),lane);}
let value=cmul(sum,roots[(q*j*${n / span}u)%N],lane);workgroupBarrier();values[lane]=value;workgroupBarrier();}\n`;
    span = step;
  }
  if(optimized)return code+'}\n';
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
