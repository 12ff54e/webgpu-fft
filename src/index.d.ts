/// <reference types="@webgpu/types" />
export interface FFTOptions {
  length: number;
  precision?: 'f32' | 'paired-f32';
  inverse?: boolean;
}
export interface FFTBindingOptions {
  batchCount?: number;
  inputOffset?: number;
  outputOffset?: number;
}
export interface FFTBinding {
  dispatch(pass: GPUComputePassEncoder): void;
  encode(encoder: GPUCommandEncoder): void;
}
export function shader(options: FFTOptions): string;
export class FFTPlan {
  private constructor();
  static create(device: GPUDevice, options: FFTOptions): Promise<FFTPlan>;
  readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly length: number;
  readonly precision: 'f32' | 'paired-f32';
  readonly inverse: boolean;
  bind(input: GPUBuffer, output: GPUBuffer, options?: FFTBindingOptions): FFTBinding;
}
