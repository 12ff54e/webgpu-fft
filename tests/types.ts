import {FFTPlan, program, shader, type FFTBinding} from '../src/index.js';

export async function record(device: GPUDevice, input: GPUBuffer, output: GPUBuffer,
                             encoder: GPUCommandEncoder): Promise<FFTBinding> {
  const code: string = shader({length: 36, precision: 'paired-f32', inverse: true});
  void code;
  const plan = await FFTPlan.create(device, {length: 36});
  const binding = plan.bind(input, output, {batchCount: 8, inputOffset: 256});
  binding.encode(encoder);
  return binding;
}

export async function real(device: GPUDevice, input: GPUBuffer, output: GPUBuffer,
                           encoder: GPUCommandEncoder): Promise<void> {
  const description = program({length: 65537, transform: 'r2c', precision: 'paired-f32'});
  const bytes: number = description.output_bytes;
  void bytes;
  const plan = await FFTPlan.create(device, {length: 65537, transform: 'r2c'});
  const binding = plan.bind(input, output);
  binding.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  binding.destroy(); plan.destroy();
}
