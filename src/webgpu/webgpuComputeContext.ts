import { nonNull } from '../util';
import { disposeNNWebGPUContext, getNNWebGPUContext, initializeNNWebGPUContext } from './webgpuContext';
import {
  WebGPUTensorBuffer,
} from './webgpuTensorBuffer';
import { GPUVocabSampler } from './vocabSampler';
import { writeSharedReadbackError } from '../sharedReadback';

export type WorkGroupDim = 'x' | 'y' | 'z';

export interface GPUKernelRunDescriptor {
  name: string;
  tensors: number[];
  workGroups: { [key in WorkGroupDim]: number };
}

export interface ComputeContextGPUMessageCreateBuffer {
  method: 'gpu.createBuffer';
  id: number;
  byteLength: number;
}

export interface ComputeContextGPUMessageCreateMetaBuffer {
  method: 'gpu.createMetaBuffer';
  id: number;
  byteLength: number;
  data: Uint8Array;
}

export interface ComputeContextGPUMessageDisposeBuffer {
  method: 'gpu.disposeBuffer';
  id: number;
}

export interface ComputeContextGPUMessageSetData {
  method: 'gpu.setData';
  id: number;
  data: Uint8Array;
  notify: SharedArrayBuffer;
}

export interface ComputeContextGPUMessageUploadMemory {
  method: 'gpu.uploadMemory';
  memory: SharedArrayBuffer;
  notify: SharedArrayBuffer;
}

export interface ComputeContextGPUMessageSharedUpload {
  method: 'gpu.sharedUpload';
  id: number;
  byteOffset?: number;
  byteLength: number;
}

export interface ComputeContextGPUMessageSharedMetaBuffer {
  method: 'gpu.sharedMetaBuffer';
  id: number;
  byteOffset?: number;
  byteLength: number;
}

export interface ComputeContextGPUMessageReleaseUploadMemory {
  method: 'gpu.releaseUploadMemory';
}

export interface ComputeContextGPUMessageGetData {
  method: 'gpu.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
  error?: SharedArrayBuffer;
}

export interface ComputeContextGPUMessageSampleLogitsDevice {
  method: 'gpu.sampleLogitsDevice';
  id: number;
  count: number;
  temperature: number;
  random: number;
  data?: SharedArrayBuffer;
  notify?: SharedArrayBuffer;
  error?: SharedArrayBuffer;
}

export interface ComputeContextGPUMessageAddKernel {
  method: 'gpu.addKernel';
  name: string;
  descriptor: { source: string; bindingTypes: GPUBufferBindingType[] };
}

export interface ComputeContextGPUMessageRunKernel {
  method: 'gpu.runKernel';
  descriptor: GPUKernelRunDescriptor;
}

export interface ComputeContextGPUMessageBeginCapture {
  method: 'gpu.beginCapture';
  name: string;
}

export interface ComputeContextGPUMessageEndCapture {
  method: 'gpu.endCapture';
}

export interface ComputeContextGPUMessageReplay {
  method: 'gpu.replay';
  name: string;
}

export interface ComputeContextGPUMessageResetCaptures {
  method: 'gpu.resetCaptures';
}

export interface ComputeContextGPUMessageReleaseCapture {
  method: 'gpu.releaseCapture';
  name: string;
}

export type ComputeContextGPUMessage =
  | ComputeContextGPUMessageAddKernel
  | ComputeContextGPUMessageCreateBuffer
  | ComputeContextGPUMessageCreateMetaBuffer
  | ComputeContextGPUMessageDisposeBuffer
  | ComputeContextGPUMessageGetData
  | ComputeContextGPUMessageSampleLogitsDevice
  | ComputeContextGPUMessageRunKernel
  | ComputeContextGPUMessageSetData
  | ComputeContextGPUMessageUploadMemory
  | ComputeContextGPUMessageSharedUpload
  | ComputeContextGPUMessageSharedMetaBuffer
  | ComputeContextGPUMessageReleaseUploadMemory
  | ComputeContextGPUMessageBeginCapture
  | ComputeContextGPUMessageEndCapture
  | ComputeContextGPUMessageReplay
  | ComputeContextGPUMessageResetCaptures
  | ComputeContextGPUMessageReleaseCapture;

export class ComputeContextGPU {
  tensorBuffers: Map<number, WebGPUTensorBuffer> = new Map();
  private vocabSampler: GPUVocabSampler | null = null;
  commandError: unknown = null;
  // Graph capture/replay: record the kernel-dispatch sequence of one step so it
  // can be re-issued from JS with a single call, eliminating per-op Python cost.
  private capturing: string | null = null;
  private captures: Map<string, GPUKernelRunDescriptor[]> = new Map();
  private capturePins: Map<string, Set<number>> = new Map();
  private pinned: Set<number> = new Set();
  /** What the device was created with: the optional features it has and the limits that
   * decide which kernels can run on it. Detected on the device itself, for any GPU. */
  features(): Record<string, unknown> {
    try {
      const ctx = getNNWebGPUContext();
      const dev = ctx.device;
      return {
        f16: !!dev?.features?.has('shader-f16'),
        subgroups: !!dev?.features?.has('subgroups'),
        subgroupMinSize: ctx.adapterFacts.subgroupMinSize,
        subgroupMaxSize: ctx.adapterFacts.subgroupMaxSize,
        maxWorkgroupStorage: Number(dev?.limits?.maxComputeWorkgroupStorageSize || 0),
        maxInvocations: Number(dev?.limits?.maxComputeInvocationsPerWorkgroup || 0),
        vendor: ctx.adapterFacts.vendor,
        architecture: ctx.adapterFacts.architecture,
      };
    } catch (_) {
      return { f16: false, subgroups: false };
    }
  }

  async init() {
    await initializeNNWebGPUContext();
  }

  dispose() {
    this.resetCaptures();
    try {
      this.vocabSampler?.dispose();
      this.vocabSampler = null;
      for (const tb of this.tensorBuffers.values()) tb.dispose();
    } finally {
      this.tensorBuffers.clear();
      disposeNNWebGPUContext();
    }
  }

  createBuffer(
    id: number,
    byteLength: number,
  ) {
    const tensorBuffer = new WebGPUTensorBuffer({
      byteLength,
    }, false);
    this.tensorBuffers.set(id, tensorBuffer);
  }

  createMetaBuffer(
    id: number,
    byteLength: number,
    data: Uint8Array,
  ) {
    const tensorBuffer = new WebGPUTensorBuffer({
      byteLength,
    }, true);
    try {
      tensorBuffer.setMetaBufferContent(data);
      this.tensorBuffers.set(id, tensorBuffer);
    } catch (error) {
      tensorBuffer.dispose();
      throw error;
    }
  }

  disposeBuffer(id: number) {
    // Buffers referenced by a captured graph must stay alive for replay.
    if (this.pinned.has(id)) {
      return;
    }
    const tb = this.tensorBuffers.get(id);
    if (tb) {
      tb.dispose();
      this.tensorBuffers.delete(id);
    }
  }

  beginCapture(name: string) {
    this.capturing = name;
    this.captures.set(name, []);
    // A recording with this name replaces the old one.  Keep ids shared with other
    // recordings pinned, but allow the old recording's orphaned buffers to be disposed.
    this.capturePins.set(name, new Set());
    this.pinned = new Set(Array.from(this.capturePins.values()).flatMap(ids => [...ids]));
  }

  endCapture() {
    this.capturing = null;
  }

  releaseCapture(name: string) {
    if (this.capturing === name) throw new Error(`cannot release active capture '${name}'`);
    if (!this.captures.delete(name)) throw new Error(`capture '${name}' not found`);
    this.capturePins.delete(name);
    this.pinned = new Set(Array.from(this.capturePins.values()).flatMap(ids => [...ids]));
  }

  // Drop every recorded graph and unpin all of their buffers. Sent when a model is
  // released: without it the pins live forever, disposeBuffer keeps refusing every
  // buffer the captured decode ever touched, and the freed model's GPU memory is
  // never returned — so the next model allocates on top of it until the device
  // dies. A live model may reuse a graph between compatible turns; its owner
  // invalidates that reference before release or profiling. The disposeBuffer
  // messages that follow this one (same FIFO channel) then reach the buffers.
  resetCaptures() {
    this.capturing = null;
    this.captures.clear();
    this.capturePins.clear();
    this.pinned.clear();
  }

  replay(name: string) {
    const seq = this.captures.get(name);
    if (!seq) {
      throw new Error(`capture '${name}' not found`);
    }
    for (let i = 0; i < seq.length; i++) {
      this.runKernel(seq[i]);
    }
  }

  setData(id: number, data: Uint8Array): void | Promise<void> {
    if (this.commandError) throw this.commandError;
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      throw new Error(`WebGPU upload target ${id} was not created`);
    }
    return tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint8Array> {
    if (this.commandError) return Promise.reject(this.commandError);
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject(new Error(`WebGPU readback target ${id} was not created`));
    }
    return tb.getDataRaw() as Promise<Uint8Array>;
  }

  getDataInto(id: number, target: SharedArrayBuffer): Promise<void> {
    if (this.commandError) return Promise.reject(this.commandError);
    const tb = this.tensorBuffers.get(id);
    if (!tb) return Promise.reject(new Error(`WebGPU readback target ${id} was not created`));
    return tb.getDataInto(new Uint8Array(target, 0, tb.bufferShape.byteLength));
  }

  sampleLogitsDevice(id: number, count: number, temperature: number,
                     random: number, target: SharedArrayBuffer): Promise<number> {
    if (this.commandError) return Promise.reject(this.commandError);
    const logits = this.tensorBuffers.get(id);
    if (!logits) return Promise.reject(new Error(`WebGPU sampler logits ${id} were not found`));
    if (!this.vocabSampler) this.vocabSampler = new GPUVocabSampler();
    return this.vocabSampler.sample(logits, count, temperature, random,
      new Uint8Array(target, 0, 4));
  }

  addKernel(
    name: string,
    descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
  ) {
    const ctx = getNNWebGPUContext();
    ctx.createPipeline(name, descriptor.source, descriptor.bindingTypes);
  }

  /** After a batch of commands from the producer: start the GPU on them if it is idle. */
  afterBatch(): void {
    try { getNNWebGPUContext().kick(); } catch (_) { /* no device yet: nothing to submit */ }
  }

  runKernel(descriptor: GPUKernelRunDescriptor) {
    if (this.capturing) {
      // record the dispatch and pin its buffers so they survive across replays
      this.captures.get(this.capturing)!.push(descriptor);
      for (const id of descriptor.tensors) {
        this.capturePins.get(this.capturing)!.add(id);
        this.pinned.add(id);
      }
    }
    const ctx = getNNWebGPUContext();
    const tensor = descriptor.tensors.map((id) =>
      nonNull(this.tensorBuffers.get(id))
    );
    ctx.runKernel({
      pipelineName: descriptor.name,
      tensorBuffers: tensor,
      workGroups: descriptor.workGroups,
    });
  }

  mdata: SharedArrayBuffer | null = null;
  mnotify: Int32Array | null = null;
  merror: SharedArrayBuffer | null = null;
  uploadMemory: SharedArrayBuffer | null = null;
  uploadNotify: Int32Array | null = null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  handleMessage(message: ComputeContextGPUMessage, worker: Worker) {
    switch (message.method) {
      case 'gpu.addKernel':
        this.addKernel(message.name, message.descriptor);
        break;
      case 'gpu.createBuffer':
        this.createBuffer(
          message.id,
          message.byteLength,
        );
        break;
      case 'gpu.createMetaBuffer':
        this.createMetaBuffer(message.id, message.byteLength, message.data);
        break;
      case 'gpu.disposeBuffer':
        this.disposeBuffer(message.id);
        break;
      case 'gpu.getData':
        if (message.data) {
          this.mdata = message.data;
        }
        if (message.notify) {
          this.mnotify = new Int32Array(message.notify);
        }
        if (message.error) this.merror = message.error;
        // A device loss or an invalid target can throw before getDataInto
        // returns a Promise.  Wake the blocked worker for both sync and async
        // failures, exactly as the WebGL readback path does.
        void Promise.resolve().then(() => this.getDataInto(message.id, this.mdata!))
          .then(() => {
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
            writeSharedReadbackError(this.merror, reason);
            this.mnotify![0] = -1;
            Atomics.notify(this.mnotify!, 0);
          });
        break;
      case 'gpu.sampleLogitsDevice': {
        if (message.data) this.mdata = message.data;
        if (message.notify) this.mnotify = new Int32Array(message.notify);
        if (message.error) this.merror = message.error;
        const notify = this.mnotify!;
        void Promise.resolve().then(() => this.sampleLogitsDevice(
          message.id, message.count, message.temperature, message.random, this.mdata!))
          .then(() => {
            notify[0] = 1;
            Atomics.notify(notify, 0);
          })
          .catch(reason => {
            console.error(reason);
            writeSharedReadbackError(this.merror, reason);
            notify[0] = -1;
            Atomics.notify(notify, 0);
          });
        break;
      }
      case 'gpu.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gpu.setData':
        // setData is initiated by synchronous Python code in the worker.  Always wake that
        // worker, including on allocation/device errors; otherwise Python waits forever and
        // the product remains stuck on "loading" after the actual error was printed only to
        // the main-thread console.
        {
          const notify = new Int32Array(message.notify);
          const finish = (status: number, reason?: unknown) => {
            if (reason) console.error(reason);
            notify[0] = status;
            Atomics.notify(notify, 0);
          };
          try {
            const pending = this.setData(message.id, message.data);
            if (pending) {
              void pending.then(() => finish(1), reason => finish(-1, reason));
            } else {
              finish(1);
            }
          } catch (reason) {
            finish(-1, reason);
          }
        }
        break;
      case 'gpu.uploadMemory':
        this.uploadMemory = message.memory;
        this.uploadNotify = new Int32Array(message.notify);
        break;
      case 'gpu.releaseUploadMemory':
        this.uploadMemory = null;
        this.uploadNotify = null;
        break;
      case 'gpu.sharedUpload': {
        const notify = this.uploadNotify;
        if (!notify) throw new Error('WebGPU shared upload was not initialized');
        const finish = (status: number, reason?: unknown) => {
          if (reason) { this.commandError = reason; console.error(reason); }
          notify[0] = status;
          Atomics.notify(notify, 0);
        };
        try {
          const offset = message.byteOffset || 0;
          if (!this.uploadMemory || offset < 0 ||
              offset + message.byteLength > this.uploadMemory.byteLength) {
            throw new Error('WebGPU shared upload size exceeds staging memory');
          }
          const data = new Uint8Array(this.uploadMemory, offset, message.byteLength);
          const pending = this.setData(message.id, data);
          if (pending) void pending.then(() => finish(1), reason => finish(-1, reason));
          else finish(1);
        } catch (reason) {
          finish(-1, reason);
        }
        break;
      }
      case 'gpu.sharedMetaBuffer': {
        const notify = this.uploadNotify;
        if (!notify) throw new Error('WebGPU shared upload was not initialized');
        try {
          const offset = message.byteOffset || 0;
          if (!this.uploadMemory || offset < 0 ||
              offset + message.byteLength > this.uploadMemory.byteLength) {
            throw new Error('WebGPU shared meta-buffer size exceeds staging memory');
          }
          this.createMetaBuffer(message.id, message.byteLength,
            new Uint8Array(this.uploadMemory, offset, message.byteLength));
          notify[0] = 1;
        } catch (reason) {
          this.commandError = reason;
          console.error(reason);
          notify[0] = -1;
        } finally {
          Atomics.notify(notify, 0);
        }
        break;
      }
      case 'gpu.beginCapture':
        this.beginCapture((message as any).name);
        break;
      case 'gpu.endCapture':
        this.endCapture();
        break;
      case 'gpu.replay':
        this.replay((message as any).name);
        break;
      case 'gpu.resetCaptures':
        this.resetCaptures();
        break;
      case 'gpu.releaseCapture':
        this.releaseCapture(message.name);
        break;
    }
  }
}
