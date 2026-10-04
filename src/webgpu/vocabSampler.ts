import { getNNWebGPUContext } from './webgpuContext';
import { WebGPUTensorBuffer } from './webgpuTensorBuffer';

// The worker sends only buffer handles and scalar sampling parameters. All
// vocabulary arithmetic, GPU dispatch and readback stay in JavaScript/GPU.
// WebGL implements the same scalar sampling contract in its JS worker because
// a fragment shader cannot synchronize a vocabulary-wide reduction.
export const VOCAB_SAMPLE_FULL_WGSL = `
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> out_idx: array<i32>;
struct SM { n: u32, temperature: f32, random: f32, pad: u32, }
@group(0) @binding(2) var<storage,read> sm: SM;
var<workgroup> maxima: array<f32, 256>;
var<workgroup> masses: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let width = (sm.n + 255u) / 256u;
  let lo = t * width;
  let hi = min(lo + width, sm.n);
  var high = -3.402823466e+38;
  for (var i = lo; i < hi; i = i + 1u) { high = max(high, x[i]); }
  maxima[t] = high;
  workgroupBarrier();
  if (t == 0u) {
    var highest = -3.402823466e+38;
    for (var j = 0u; j < 256u; j = j + 1u) { highest = max(highest, maxima[j]); }
    maxima[0] = highest;
  }
  workgroupBarrier();
  let peak = maxima[0] / sm.temperature;
  var mass = 0.0;
  for (var i = lo; i < hi; i = i + 1u) {
    mass = mass + exp(x[i] / sm.temperature - peak);
  }
  masses[t] = mass;
  workgroupBarrier();
  if (t == 0u) {
    var total = 0.0;
    for (var j = 0u; j < 256u; j = j + 1u) { total = total + masses[j]; }
    if (!(total > 0.0)) { out_idx[0] = -1; return; }
    let threshold = clamp(sm.random, 0.0, 0.99999994) * total;
    var before = 0.0;
    var selected = 0u;
    for (var j = 0u; j < 256u; j = j + 1u) {
      selected = j;
      if (before + masses[j] >= threshold) { break; }
      before = before + masses[j];
    }
    let begin = selected * width;
    let end = min(begin + width, sm.n);
    for (var i = begin; i < end; i = i + 1u) {
      before = before + exp(x[i] / sm.temperature - peak);
      if (before >= threshold) { out_idx[0] = i32(i); return; }
    }
    out_idx[0] = i32(sm.n - 1u);
  }
}
`;

export class GPUVocabSampler {
  private meta: WebGPUTensorBuffer | null = null;
  private output: WebGPUTensorBuffer | null = null;
  private ready = false;

  private async prepare() {
    if (this.ready) return;
    const ctx = getNNWebGPUContext();
    const module = ctx.device.createShaderModule({code: VOCAB_SAMPLE_FULL_WGSL});
    // Bundled @webgpu/types predates Chrome's getCompilationInfo method.
    const info = await ((module as any).getCompilationInfo?.()
      ?? (module as any).compilationInfo);
    const messages: Array<{type: string; message: string}> = info?.messages || [];
    const errors = messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error('WebGPU vocabulary sampler failed to compile: '
      + errors.map(message => message.message).join('; '));
    ctx.createPipeline('vocab_sample_full_js', VOCAB_SAMPLE_FULL_WGSL,
      ['read-only-storage', 'storage', 'read-only-storage']);
    this.meta = new WebGPUTensorBuffer({byteLength: 16}, false);
    this.output = new WebGPUTensorBuffer({byteLength: 4}, false);
    this.ready = true;
  }

  async sample(logits: WebGPUTensorBuffer, count: number, temperature: number,
               random: number, target: Uint8Array): Promise<number> {
    if (!Number.isSafeInteger(count) || count < 1 || count * 4 > logits.bufferShape.byteLength)
      throw new Error('invalid vocabulary size for GPU sampler');
    if (!(temperature > 0) || !Number.isFinite(temperature))
      throw new Error('invalid GPU sampler temperature');
    if (target.byteLength !== 4) throw new Error('GPU sampler result must be four bytes');
    await this.prepare();
    const meta = new ArrayBuffer(16);
    const view = new DataView(meta);
    view.setUint32(0, count, true);
    view.setFloat32(4, Math.max(temperature, 1e-5), true);
    view.setFloat32(8, Number.isFinite(random) ? random : 0, true);
    this.meta!.setDataRaw(new Uint8Array(meta));
    const ctx = getNNWebGPUContext();
    ctx.runKernel({pipelineName: 'vocab_sample_full_js',
      tensorBuffers: [logits, this.output!, this.meta!],
      workGroups: {x: 1, y: 1, z: 1}});
    await this.output!.getDataInto(target);
    const token = new DataView(target.buffer, target.byteOffset, 4).getInt32(0, true);
    if (token < 0 || token >= count)
      throw new Error('sampled logits have no probability mass or returned an invalid token');
    return token;
  }

  dispose() {
    this.meta?.dispose();
    this.output?.dispose();
    this.meta = null;
    this.output = null;
    this.ready = false;
  }
}
