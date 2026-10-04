import { WebGPUTensorBuffer } from './webgpuTensorBuffer';

// Opt-in browser diagnosis.  Shader compilation happens synchronously on the main thread;
// a slow first decode can therefore be compilation rather than GPU arithmetic.  Keep this
// disabled for ordinary runs so instrumentation cannot change route selection or latency.
const PROFILE_COMPILE = typeof location !== 'undefined'
  && new URLSearchParams(location.search).get('profile_compile') === '1';
// Explicit cold-start diagnosis only. The profiler arms one compute submission from the
// Python worker and records actual GPU timestamps around it, separate from host wait time.
const PROFILE_GPU = typeof location !== 'undefined'
  && new URLSearchParams(location.search).get('profile_gpu') === '1';

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

  private deviceLostReason: string | null = null;

  private pipelines: Map<string, WebGPURunnerPipeline>;

  // WebGPU reports WGSL and pipeline validation errors asynchronously. A pipeline
  // object is not proof that its shader compiled: without checking these results,
  // an invalid kernel can yield zero/stale tensors and look like a model answer.
  private pendingPipelineChecks: Promise<void>[] = [];
  private pipelineError: Error | null = null;

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
  // One reusable staging buffer per readback shape. Kept behind an opt-in switch
  // until an end-to-end A/B establishes a positive gain on the actual device.
  private readbackPool: Map<number, GPUBuffer> = new Map();
  private diagnosticQuery: GPUQuerySet | null = null;
  private diagnosticPassIndex = 0;
  // Selected-kernel diagnosis keeps the entire graph in one queue submission.
  // It splits only the selected dispatches into timestamped passes, avoiding
  // the hundreds of submissions/readbacks of the per-kernel flush mode.
  private selectedQuery: GPUQuerySet | null = null;
  private selectedNames: string[] = [];
  private readonly flushThreshold: number;

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
    if (PROFILE_COMPILE) {
      // The worker does not expose its main-thread WebGPU context to the page. Keep
      // an opt-in aggregate where browser diagnostics can read it after a load;
      // logging only single >100 ms pipelines misses dozens of smaller compiles.
      (globalThis as any).__wgpyCompile = {
        count: 0, setupMs: 0, pipelineMs: 0, maxPipelineMs: 0,
        bindGroups: 0, bindGroupMs: 0, slowest: [] as Array<{ name: string; ms: number }>,
      };
    }
    // Diagnostic override for end-to-end graph measurements.  The production default is
    // unchanged until paired browser runs establish a positive device-level winner.
    const selected = typeof location !== 'undefined'
      ? Number(new URLSearchParams(location.search).get('dispatch_flush')) : 0;
    this.flushThreshold = [512, 1024, 2048, 4096].includes(selected) ? selected : 1024;
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
    const requiredFeatures: GPUFeatureName[] = (PROFILE_GPU && adapter!.features.has('timestamp-query'))
      ? ['timestamp-query'] : [];
    this.device = (await adapter!.requestDevice({ requiredLimits, requiredFeatures })) as GPUDevice;
    if (!this.device) {
      throw new Error('GPUAdapter.requestDevice() returned null');
    }
    // Match WebGL's lost-context contract at the public tensor/API boundary.
    // A lost device can otherwise leave reads looking like valid zero logits.
    this.device.lost?.then((info) => {
      this.deviceLostReason = info.message || info.reason || 'device lost';
    });
    this.isSupported = true;
    this.initialized = true;
  }

  assertAlive(): void {
    if (this.pipelineError !== null) throw this.pipelineError;
    if (this.deviceLostReason !== null) {
      throw new Error('WebGPU device lost: ' + this.deviceLostReason
                      + '; release and reload the model');
    }
  }

  private trackPipelineCheck(name: string, check: Promise<void>): void {
    this.pendingPipelineChecks.push(check.catch((reason) => {
      if (this.pipelineError === null) {
        this.pipelineError = new Error(`WebGPU pipeline ${name} failed: ${String((reason as any)?.message || reason)}`);
      }
    }));
  }

  async assertPipelinesReady(): Promise<void> {
    this.assertAlive();
    const checks = this.pendingPipelineChecks.splice(0);
    if (checks.length) await Promise.all(checks);
    this.assertAlive();
  }

  hasPipeline(name: string): boolean {
    return this.pipelines.has(name);
  }


  createPipeline(name: string, source: string, bindingTypes: GPUBufferBindingType[]): void {
    this.assertAlive();
    if (this.hasPipeline(name)) {
      return;
    }
    const setupStarted = PROFILE_COMPILE ? performance.now() : 0;
    const { device } = this,
      bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < bindingTypes.length; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindingTypes[i] },
      });
    }
    device.pushErrorScope('validation');
    let shaderModule: GPUShaderModule;
    let bindGroupLayout: GPUBindGroupLayout;
    let pipeline: GPUComputePipeline;
    try {
      bindGroupLayout = device.createBindGroupLayout({
        entries: bindings,
      });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      });
      shaderModule = device.createShaderModule({ code: source });
      const started = PROFILE_COMPILE ? performance.now() : 0;
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });
      if (PROFILE_COMPILE) {
        const ms = performance.now() - started;
        const stats = (globalThis as any).__wgpyCompile;
        stats.count++;
        stats.setupMs += performance.now() - setupStarted;
        stats.pipelineMs += ms;
        stats.maxPipelineMs = Math.max(stats.maxPipelineMs, ms);
        stats.slowest.push({ name, ms });
        stats.slowest.sort((a: { ms: number }, b: { ms: number }) => b.ms - a.ms);
        if (stats.slowest.length > 10) stats.slowest.length = 10;
        if (ms >= 100) console.info(`webgpu pipeline ${name}: ${ms.toFixed(1)} ms`);
      }
      const info = (shaderModule as any).getCompilationInfo?.();
      if (!info) throw new Error('WebGPU shader compilation diagnostics are unavailable');
      this.trackPipelineCheck(name, Promise.resolve(info).then((result: any) => {
        const errors = result.messages?.filter((message: any) => message.type === 'error') || [];
        if (errors.length) throw new Error(errors.map((message: any) => message.message).join('; '));
      }));
    } finally {
      this.trackPipelineCheck(name, device.popErrorScope().then((error) => {
        if (error) throw new Error(error.message || String(error));
      }));
    }
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
    this.assertAlive();
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
      const bindStarted = PROFILE_COMPILE ? performance.now() : 0;
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
      if (PROFILE_COMPILE) {
        const stats = (globalThis as any).__wgpyCompile;
        stats.bindGroups++;
        stats.bindGroupMs += performance.now() - bindStarted;
      }
    }
    if (!this.commandEncoder) {
      this.commandEncoder = device.createCommandEncoder();
    }
    const selected = PROFILE_GPU && device.features.has('timestamp-query')
      && Array.isArray((globalThis as any).__wgpyProfileKernelNames)
      && (globalThis as any).__wgpyProfileKernelNames.includes(request.pipelineName);
    if (selected) {
      // An existing non-timestamped pass must end before a selected dispatch
      // can get its own timestamp interval. No submit or CPU wait happens here.
      this.passEncoder?.end();
      this.passEncoder = null;
      if (!this.selectedQuery) {
        this.selectedQuery = device.createQuerySet({ type: 'timestamp', count: 1024 });
      }
      if (this.selectedNames.length < 512) {
        const q = this.selectedNames.length * 2;
        this.passEncoder = this.commandEncoder.beginComputePass({ timestampWrites: {
          querySet: this.selectedQuery, beginningOfPassWriteIndex: q,
          endOfPassWriteIndex: q + 1,
        } } as any);
        const byShape = (globalThis as any).__wgpyProfileKernelWorkgroups === true;
        this.selectedNames.push(byShape
          ? `${request.pipelineName}@${request.workGroups.x}x${request.workGroups.y}x${request.workGroups.z}`
          : request.pipelineName);
      }
    }
    if (!this.passEncoder) {
      if (!selected && PROFILE_GPU && ((globalThis as any).__wgpyProfileNextPass === true
          || (globalThis as any).__wgpyProfileAllPasses === true)
          && device.features.has('timestamp-query')) {
        const skip = (globalThis as any).__wgpyProfileAllPasses === true ? 0
          : Number((globalThis as any).__wgpyProfileSkipPasses || 0);
        if (skip > 0) {
          (globalThis as any).__wgpyProfileSkipPasses = skip - 1;
          this.passEncoder = this.commandEncoder.beginComputePass();
        } else {
          (globalThis as any).__wgpyProfileNextPass = false;
          let query: GPUQuerySet | null = null;
          try {
            query = device.createQuerySet({ type: 'timestamp', count: 2 });
            // Chrome's current WebGPU API takes one timestampWrites object. The bundled
            // @webgpu/types declaration still describes an older iterable form.
            this.passEncoder = this.commandEncoder.beginComputePass({ timestampWrites: {
              querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
            } } as any);
            this.diagnosticQuery = query;
          } catch (error) {
            query?.destroy();
            (globalThis as any).__wgpyGpuTiming = { error: String(error) };
            this.passEncoder = this.commandEncoder.beginComputePass();
          }
        }
      } else {
        this.passEncoder = this.commandEncoder.beginComputePass();
      }
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
    if (selected) {
      this.passEncoder.end();
      this.passEncoder = null;
    }
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
      const query = this.diagnosticQuery;
      const selectedQuery = this.selectedQuery;
      const selectedNames = this.selectedNames;
      const diagnosticDispatches = this.pendingCount;
      const diagnosticPassIndex = query ? ++this.diagnosticPassIndex : 0;
      let timingRead: GPUBuffer | null = null;
      let timingResolve: GPUBuffer | null = null;
      if (query) {
        timingResolve = this.device.createBuffer({
          size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        timingRead = this.device.createBuffer({
          size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.commandEncoder.resolveQuerySet(query, 0, 2, timingResolve, 0);
        this.commandEncoder.copyBufferToBuffer(timingResolve, 0, timingRead, 0, 16);
      }
      let selectedRead: GPUBuffer | null = null;
      let selectedResolve: GPUBuffer | null = null;
      if (selectedQuery && selectedNames.length > 0) {
        const bytes = selectedNames.length * 16;
        selectedResolve = this.device.createBuffer({
          size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        selectedRead = this.device.createBuffer({
          size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.commandEncoder.resolveQuerySet(selectedQuery, 0, selectedNames.length * 2,
                                            selectedResolve, 0);
        this.commandEncoder.copyBufferToBuffer(selectedResolve, 0, selectedRead, 0, bytes);
      }
      this.device.queue.submit([this.commandEncoder.finish()]);
      if (selectedQuery && selectedRead && selectedResolve) {
        const read = selectedRead, resolve = selectedResolve, names = selectedNames;
        read.mapAsync(GPUMapMode.READ).then(() => {
          const ticks = new BigUint64Array(read.getMappedRange());
          const byName: Record<string, { count: number; gpuMs: number }> = {};
          for (let i = 0; i < names.length; i++) {
            const row = byName[names[i]] || (byName[names[i]] = { count: 0, gpuMs: 0 });
            row.count++;
            row.gpuMs += Number(ticks[i * 2 + 1] - ticks[i * 2]) / 1e6;
          }
          const all = (globalThis as any).__wgpySelectedKernelPasses;
          if (Array.isArray(all)) all.push({ dispatches: diagnosticDispatches, byName });
          read.unmap();
        }).catch(error => {
          (globalThis as any).__wgpySelectedKernelError = String(error);
        }).finally(() => {
          read.destroy(); resolve.destroy(); selectedQuery.destroy();
        });
      } else if (selectedQuery) {
        selectedQuery.destroy();
      }
      if (query && timingRead && timingResolve) {
        const read = timingRead, resolve = timingResolve;
        read.mapAsync(GPUMapMode.READ).then(() => {
          const ticks = new BigUint64Array(read.getMappedRange());
          const ms = Number(ticks[1] - ticks[0]) / 1e6;
          const reading = { index: diagnosticPassIndex, gpuMs: ms,
                            dispatches: diagnosticDispatches };
          (globalThis as any).__wgpyGpuTiming = reading;
          if (Array.isArray((globalThis as any).__wgpyGpuPasses)) {
            (globalThis as any).__wgpyGpuPasses.push(reading);
          }
          if ((globalThis as any).__wgpyProfileAllPasses !== true) {
            console.info(`webgpu profiled compute pass: ${ms.toFixed(3)} ms on GPU, ${diagnosticDispatches} dispatches`);
          }
          read.unmap();
        }).catch(error => {
          (globalThis as any).__wgpyGpuTiming = { error: String(error) };
        }).finally(() => {
          read.destroy(); resolve.destroy(); query.destroy();
        });
        this.diagnosticQuery = null;
      }
      this.selectedQuery = null;
      this.selectedNames = [];
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

  rentReadback(byteLength: number): GPUBuffer {
    const cached = this.readbackPool.get(byteLength);
    if (cached) {
      this.readbackPool.delete(byteLength);
      return cached;
    }
    return this.device.createBuffer({
      size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  returnReadback(buffer: GPUBuffer, byteLength: number, reusable: boolean): void {
    if (reusable && this.initialized && this.deviceLostReason === null &&
        !this.readbackPool.has(byteLength)) {
      this.readbackPool.set(byteLength, buffer);
    } else {
      buffer.destroy();
    }
  }

  dispose(): void {
    if (!this.initialized) return;
    // A released model can have a zero buffer ledger while Dawn/Metal retains its
    // physical allocation. Destroying the device is the final resource boundary.
    try {
      this.flush();
    } finally {
      this.bindGroupCache.clear();
      this.pipelines.clear();
      this.pendingPipelineChecks.length = 0;
      for (const buffer of this.readbackPool.values()) buffer.destroy();
      this.readbackPool.clear();
      this.device.destroy();
      this.initialized = false;
      this.isSupported = false;
    }
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

export function disposeNNWebGPUContext(): void {
  const old = context;
  context = null;
  if (old) old.dispose();
}
