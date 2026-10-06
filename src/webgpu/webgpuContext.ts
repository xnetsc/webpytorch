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

  // What the adapter said about itself, for `features()`. Diagnostic: no route keys on it.
  adapterFacts: { vendor: string; architecture: string; subgroupMinSize: number;
                  subgroupMaxSize: number } = { vendor: '', architecture: '',
                                                subgroupMinSize: 0, subgroupMaxSize: 0 };

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
  // Submissions the GPU has not finished. While it is working, dispatches accumulate into one
  // submit; once it has nothing, what is pending goes at once (`kick`) instead of waiting for
  // the threshold or the next readback. Without this a prefill whose Python issued its 452
  // dispatches in 47 ms left the GPU idle for those 47 ms and then ran 71 ms of work behind
  // the final readback -- 118 ms where the two could overlap.
  private inflight = 0;
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
  // A timed span (`spanBegin`/`spanEnd`): every pass begun inside it writes its own begin
  // and end timestamps into `spanQuery`, and nothing is submitted early, so the span's
  // dispatches run back to back. The sum of the pass lengths is the GPU's own time for
  // that work -- not the browser's clock, which is coarsened and jittered on purpose and
  // which also counts the host issuing and the readback around it.
  private spanQuery: GPUQuerySet | null = null;
  private span: { pairs: number; overflow: boolean } | null = null;
  private static readonly SPAN_PAIRS = 512;
  // The step the device's timestamps come in, in ns, as the readings themselves show it: the
  // largest power of two dividing every one of them. A browser may coarsen them on purpose --
  // Chrome without developer flags reports multiples of 65536 ns -- and a caller timing
  // short work needs to know, to time enough of it.
  timestampStepNs = 0;

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
    // Timestamps whenever the adapter has them: route races time candidates with the GPU's
    // own clock (`spanBegin`). Where it has not, they fall back to the host's clock.
    const requiredFeatures: GPUFeatureName[] = adapter!.features.has('timestamp-query')
      ? ['timestamp-query'] : [];
    // Standard optional features, asked for whenever THIS adapter has them, whatever GPU it
    // is: half-precision arithmetic (`shader-f16`) and subgroup operations (`subgroups`).
    // Nothing is assumed from a vendor name. Python is told what the device ended up with
    // (`features()`) and only then offers the kernels that need a feature as candidates;
    // which of those is fastest is measured on the device itself.
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    for (const f of ['shader-f16', 'subgroups'] as GPUFeatureName[]) {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      if (adapter!.features.has(f)) requiredFeatures.push(f);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const info: any = (adapter as any)?.info || {};
    this.adapterFacts = {
      vendor: String(info.vendor || ''), architecture: String(info.architecture || ''),
      subgroupMinSize: Number(info.subgroupMinSize || 0),
      subgroupMaxSize: Number(info.subgroupMaxSize || 0),
    };
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
    if (!this.passEncoder && this.span && !selected) {
      if (this.span.pairs < NNWebGPUContext.SPAN_PAIRS) {
        const q = this.span.pairs++ * 2;
        this.passEncoder = this.commandEncoder.beginComputePass({ timestampWrites: {
          querySet: this.spanQuery!, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1,
        } } as any);
      } else {
        this.span.overflow = true;
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

  /** Submit what is pending if the GPU has nothing to do; called after each batch of
   * commands the producer sends. A diagnostic pass being timed is left whole. */
  kick(): void {
    if (this.inflight === 0 && this.pendingCount > 0 && !this.diagnosticQuery && !this.span) {
      this.flush();
    }
  }

  /** Whether `spanBegin` can time anything here: the device has timestamp queries. */
  hasTimestamps(): boolean {
    return !!this.device && this.device.features.has('timestamp-query');
  }

  /** Start timing the GPU work issued from now to `spanEnd`. False where the device cannot
   * (no timestamp queries) or a span is already open; the caller then times another way. */
  spanBegin(): boolean {
    this.assertAlive();
    if (this.span || !this.hasTimestamps()) return false;
    this.flush();                      // what came before is not this span's work
    if (!this.spanQuery) {
      this.spanQuery = this.device.createQuerySet({
        type: 'timestamp', count: 2 * NNWebGPUContext.SPAN_PAIRS });
    }
    this.span = { pairs: 0, overflow: false };
    return true;
  }

  /** Milliseconds of GPU time the span's passes took, summed; -1 when it could not be
   * timed whole (more passes than it has room to time). Waits for that work to finish. */
  async spanEnd(): Promise<number> {
    const span = this.span;
    this.span = null;
    if (!span) return -1;
    if (this.passEncoder) { this.passEncoder.end(); this.passEncoder = null; }
    if (span.pairs === 0 || span.overflow) {
      this.flush();
      return span.overflow ? -1 : 0;
    }
    if (!this.commandEncoder) this.commandEncoder = this.device.createCommandEncoder();
    const bytes = span.pairs * 16;
    const resolve = this.device.createBuffer({
      size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.device.createBuffer({
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      this.commandEncoder.resolveQuerySet(this.spanQuery!, 0, span.pairs * 2, resolve, 0);
      this.commandEncoder.copyBufferToBuffer(resolve, 0, read, 0, bytes);
      this.flush();
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange());
      let ns = 0;
      let zeros = 24;
      const low = BigInt(0xffffffff);
      for (let i = 0; i < span.pairs; i++) {
        // Each pass's length fits a double exactly; the absolute ticks may not.
        if (t[2 * i + 1] > t[2 * i]) ns += Number(t[2 * i + 1] - t[2 * i]);
        // The step: the fewest trailing zero bits of any reading (a step past 2^24 ns is not
        // a clock anyone times with; the low 32 bits decide everything below it).
        for (let j = 2 * i; j < 2 * i + 2; j++) {
          const v = Number(t[j] & low) >>> 0;
          if (v !== 0) zeros = Math.min(zeros, 31 - Math.clz32(v & -v));
        }
      }
      read.unmap();
      this.timestampStepNs = 2 ** zeros;
      return ns / 1e6;
    } finally {
      read.destroy();
      resolve.destroy();
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const done = (this.device.queue as any).onSubmittedWorkDone?.();
      if (done && typeof done.then === 'function') {
        this.inflight++;
        done.then(() => {
          this.inflight--;
          this.kick();                   // what accumulated while it worked goes now
        }, () => { this.inflight--; });
      }
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

  /** Copy `byteLength` bytes of `src` into `dst` behind every dispatch encoded so far, in
   * the same command buffer, and submit it. The copy sees those dispatches' results with no
   * separate submission; `dst` may be mapped as soon as this returns. */
  copyAndSubmit(src: GPUBuffer, dst: GPUBuffer, byteLength: number): void {
    this.assertAlive();
    if (this.passEncoder) {
      this.passEncoder.end();
      this.passEncoder = null;
    }
    if (!this.commandEncoder) this.commandEncoder = this.device.createCommandEncoder();
    this.commandEncoder.copyBufferToBuffer(src, 0, dst, 0, byteLength);
    this.pendingCount++;
    this.flush();
  }

  /** Zero `buffer` behind every dispatch encoded so far, in the same command buffer: the
   * device's own fill, with no data from the host and no submission of its own. */
  clearBuffer(buffer: GPUBuffer): void {
    this.assertAlive();
    if (this.passEncoder) {
      this.passEncoder.end();
      this.passEncoder = null;
    }
    if (!this.commandEncoder) this.commandEncoder = this.device.createCommandEncoder();
    this.commandEncoder.clearBuffer(buffer);
    this.pendingCount++;
    if (this.pendingCount >= this.flushThreshold) this.flush();
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
