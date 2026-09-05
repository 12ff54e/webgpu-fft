import {FFTPlan, shader, type FFTBinding} from '../src/index.js';

export async function record(device: GPUDevice, input: GPUBuffer, output: GPUBuffer,
                             encoder: GPUCommandEncoder): Promise<FFTBinding> {
  const code: string = shader({length: 36, precision: 'paired-f32', inverse: true});
  void code;
  const plan = await FFTPlan.create(device, {length: 36});
  const binding = plan.bind(input, output, {batchCount: 8, inputOffset: 256});
  binding.encode(encoder);
  return binding;
}
