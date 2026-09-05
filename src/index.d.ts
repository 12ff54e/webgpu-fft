// Uses WebGPU globals from the consumer's DOM library or @webgpu/types.
export interface FFTOptions {
  length: number;
  precision?: 'f32' | 'paired-f32';
  inverse?: boolean;
  transform?: 'c2c' | 'r2c' | 'c2r';
}
export interface FFTBindingOptions {
  batchCount?: number;
  inputOffset?: number;
  outputOffset?: number;
}
export interface FFTBinding {
  dispatch(pass: GPUComputePassEncoder): void;
  encode(encoder: GPUCommandEncoder): void;
  destroy(): void;
}
export interface FFTStage {entry_point: string; span: number; flags: number;}
export interface FFTProgram {
  length: number; fft_length: number; transform: 'c2c' | 'r2c' | 'c2r';
  paired: boolean; inverse: boolean; small: boolean; bluestein: boolean;
  table: Float32Array; code: string; small_code: string; stages: FFTStage[];
  input_bytes: number; output_bytes: number;
}
export function program(options: FFTOptions): FFTProgram;
export function shader(options: FFTOptions): string;
export class FFTPlan {
  private constructor();
  static create(device: GPUDevice, options: FFTOptions): Promise<FFTPlan>;
  readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline | undefined;
  readonly length: number;
  readonly precision: 'f32' | 'paired-f32';
  readonly inverse: boolean;
  readonly transform: 'c2c' | 'r2c' | 'c2r';
  destroy(): void;
  bind(input: GPUBuffer, output: GPUBuffer, options?: FFTBindingOptions): FFTBinding;
}
