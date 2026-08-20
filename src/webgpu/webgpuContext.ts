import { WebGPUTensorBuffer } from './webgpuTensorBuffer';

interface WebGPURunnerPipeline {
  bindGroupLayout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}

type WorkGroupDim = 'x' | 'y' | 'z';

export interface WebGPUMetaBufferContentElement {
  value: number;
  type: 'int32' | 'uint32' | 'float32';
}

export interface WebGPUMetaBufferContent {
  elements: WebGPUMetaBufferContentElement[];
}

export interface WebGPURunnerRequest {
  pipelineName: string;
  tensorBuffers: WebGPUTensorBuffer[];
  workGroups: { [key in WorkGroupDim]: number };
}

export class NNWebGPUContext {
  initialized: boolean;

  isSupported: boolean;

  device!: GPUDevice;

  private pipelines: Map<string, WebGPURunnerPipeline>;

  // Batched submission: accumulate many compute dispatches into ONE command
  // encoder and submit once (at flush), instead of one queue.submit per kernel.
  private commandEncoder: GPUCommandEncoder | null = null;
  // One compute pass held open across dispatches. WebGPU orders dispatches within a pass
  // and makes each one's writes visible to the next, so a pass per dispatch buys nothing
  // and costs a begin/end pair every time.
  private passEncoder: GPUComputePassEncoder | null = null;
  // Bind groups keyed by pipeline + the exact buffers bound. A captured decode step replays
  // the same kernels over the same buffers every token, so rebuilding them each dispatch is
  // pure overhead -- and creating a bind group is one of the more expensive WebGPU calls.
  private bindGroupCache: Map<string, GPUBindGroup> = new Map();
  private pendingCount = 0;
  private pendingDisposes: GPUBuffer[] = [];
  private readonly flushThreshold = 1024;

  constructor() {
    if (
      typeof navigator.gpu !== 'object' ||
      typeof navigator.gpu.requestAdapter !== 'function'
    ) {
      throw new Error('WebGPU is not supported on this browser');
    }
    this.initialized = false;
    this.isSupported = false;
    this.pipelines = new Map();
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const adapter = await navigator.gpu!.requestAdapter();
    // Default device limits cap a storage buffer binding at 128MB, which is far
    // too small for LLM weight tensors. Ask for whatever the adapter allows.
    const requiredLimits: Record<string, number> = {};
    const wanted = [
      'maxBufferSize',
      'maxStorageBufferBindingSize',
      'maxStorageBuffersPerShaderStage',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupStorageSize',
    ];
    for (const key of wanted) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = (adapter as any)?.limits?.[key];
      if (typeof v === 'number' || typeof v === 'bigint') {
        requiredLimits[key] = Number(v);
      }
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    this.device = (await adapter!.requestDevice({ requiredLimits })) as GPUDevice;
    if (!this.device) {
      throw new Error('GPUAdapter.requestDevice() returned null');
    }
    this.isSupported = true;
    this.initialized = true;
  }

  hasPipeline(name: string): boolean {
    return this.pipelines.has(name);
  }

  createPipeline(name: string, source: string, bindingTypes: GPUBufferBindingType[]): void {
    if (this.hasPipeline(name)) {
      return;
    }
    const { device } = this,
      bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < bindingTypes.length; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindingTypes[i] },
      });
    }
    const bindGroupLayout = device.createBindGroupLayout({
        entries: bindings,
      }),
      pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      shaderModule = device.createShaderModule({ code: source }),
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

    this.pipelines.set(name, { bindGroupLayout, pipeline });
  }

  // Stable per-buffer id; GPUBuffer has no identity we can key a Map on directly.
  private bufferIds: WeakMap<GPUBuffer, number> = new WeakMap();
  private nextBufferId = 1;

  private bufferKey(b: GPUBuffer): number {
    let id = this.bufferIds.get(b);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(b, id);
    }
    return id;
  }

  runKernel(request: WebGPURunnerRequest): void {
    const pipeline = this.pipelines.get(request.pipelineName);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipeline} not found`);
    }
    const { device } = this;
    let key = request.pipelineName;
    for (let i = 0; i < request.tensorBuffers.length; i++) {
      const t = request.tensorBuffers[i];
      key += '|' + this.bufferKey(t.gpuBuffer) + ':' + t.bufferShape.byteLength;
    }
    let bindGroup = this.bindGroupCache.get(key);
    if (!bindGroup) {
      const entries: GPUBindGroupEntry[] = request.tensorBuffers.map((t, i) => ({
        binding: i,
        resource: {
          buffer: t.gpuBuffer,
          size: t.bufferShape.byteLength,
        },
      }));
      bindGroup = device.createBindGroup({
        layout: pipeline.bindGroupLayout,
        entries,
      });
      this.bindGroupCache.set(key, bindGroup);
    }
    if (!this.commandEncoder) {
      this.commandEncoder = device.createCommandEncoder();
    }
    if (!this.passEncoder) {
      this.passEncoder = this.commandEncoder.beginComputePass();
    }
    const passEncoder = this.passEncoder;
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(pipeline.pipeline);
    passEncoder.dispatchWorkgroups(
      request.workGroups.x,
      request.workGroups.y,
      request.workGroups.z
    );
    this.pendingCount++;
    if (this.pendingCount >= this.flushThreshold) {
      this.flush();
    }
  }

  // Submit all accumulated dispatches in one queue.submit, then safely destroy
  // any buffers whose disposal was deferred while they might still be referenced.
  flush(): void {
    if (this.passEncoder) {
      // The open pass has to be closed before the encoder can be finished.
      if (this.passEncoder.end) {
        this.passEncoder.end();
      } else {
        // deprecated (Firefox Nightly 111)
        (this.passEncoder as any).endPass();
      }
      this.passEncoder = null;
    }
    if (this.commandEncoder) {
      this.device.queue.submit([this.commandEncoder.finish()]);
      this.commandEncoder = null;
      this.pendingCount = 0;
    }
    if (this.pendingDisposes.length > 0) {
      for (const buf of this.pendingDisposes) {
        buf.destroy();
      }
      this.pendingDisposes.length = 0;
      // A destroyed buffer must not stay referenced by a cached bind group.
      this.bindGroupCache.clear();
    }
  }

  // Defer a buffer destroy until the next flush: a dispatch already encoded in
  // the pending command buffer may still reference it.
  deferDispose(buffer: GPUBuffer): void {
    this.pendingDisposes.push(buffer);
  }
}

let context: NNWebGPUContext | null = null;
export async function initializeNNWebGPUContext(): Promise<void> {
  context = new NNWebGPUContext();
  try {
    await context.initialize();
  } catch (error) {
    context = null;
    throw error;
  }
}

export function getNNWebGPUContext(): NNWebGPUContext {
  if (!context) {
    throw new Error('WebGPU Context does not exist');
  }
  return context;
}
