import { nonNull } from '../util';
import { getNNWebGPUContext, initializeNNWebGPUContext } from './webgpuContext';
import {
  WebGPUTensorBuffer,
} from './webgpuTensorBuffer';

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
}

export interface ComputeContextGPUMessageGetData {
  method: 'gpu.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
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

export type ComputeContextGPUMessage =
  | ComputeContextGPUMessageAddKernel
  | ComputeContextGPUMessageCreateBuffer
  | ComputeContextGPUMessageCreateMetaBuffer
  | ComputeContextGPUMessageDisposeBuffer
  | ComputeContextGPUMessageGetData
  | ComputeContextGPUMessageRunKernel
  | ComputeContextGPUMessageSetData
  | ComputeContextGPUMessageBeginCapture
  | ComputeContextGPUMessageEndCapture
  | ComputeContextGPUMessageReplay;

export class ComputeContextGPU {
  tensorBuffers: Map<number, WebGPUTensorBuffer> = new Map();
  // Graph capture/replay: record the kernel-dispatch sequence of one step so it
  // can be re-issued from JS with a single call, eliminating per-op Python cost.
  private capturing: string | null = null;
  private captures: Map<string, GPUKernelRunDescriptor[]> = new Map();
  private pinned: Set<number> = new Set();
  async init() {
    await initializeNNWebGPUContext();
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
    tensorBuffer.setMetaBufferContent(data);
    this.tensorBuffers.set(id, tensorBuffer);
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
  }

  endCapture() {
    this.capturing = null;
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

  setData(id: number, data: Uint8Array): void {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return;
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint8Array> {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject();
    }
    return tb.getDataRaw() as Promise<Uint8Array>;
  }

  addKernel(
    name: string,
    descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
  ) {
    const ctx = getNNWebGPUContext();
    ctx.createPipeline(name, descriptor.source, descriptor.bindingTypes);
  }

  runKernel(descriptor: GPUKernelRunDescriptor) {
    if (this.capturing) {
      // record the dispatch and pin its buffers so they survive across replays
      this.captures.get(this.capturing)!.push(descriptor);
      for (const id of descriptor.tensors) {
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
        this.getData(message.id)
          .then((data) => {
            (new Uint8Array(this.mdata!)).set(data);
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
          });
        break;
      case 'gpu.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gpu.setData':
        this.setData(message.id, message.data);
        break;
      case 'gpu.beginCapture':
        this.beginCapture((message as any).name);
        break;
      case 'gpu.endCapture':
        this.endCapture();
        break;
      case 'gpu.replay':
        this.replay((message as any).name);
        break;
    }
  }
}
