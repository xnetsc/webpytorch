import { nonNull } from '../util';
import { WorkGroupDim } from '../webgpu/webgpuComputeContext';
import {
  getNNWebGLContext,
  initializeNNWebGLContext,
  TensorTextureShape,
  WebGLTensorBuffer,
  WebGLUniformItem,
} from './webglContext';

export interface GLKernelRunDescriptor {
  name: string;
  inputs: { name: string; id: number }[];
  output: number;
  uniforms: WebGLUniformItem[];
}

export interface GPUKernelRunDescriptor {
  name: string;
  tensors: number[];
  uniforms: WebGLUniformItem[];
  workGroups: { [key in WorkGroupDim]: number };
}

export interface ComputeContextGLMessageCreateBuffer {
  method: 'gl.createBuffer';
  id: number;
  textureShape: TensorTextureShape;
}

export interface ComputeContextGLMessageDisposeBuffer {
  method: 'gl.disposeBuffer';
  id: number;
}

export interface ComputeContextGLMessageSetData {
  method: 'gl.setData';
  id: number;
  data: Float32Array;
}

export interface ComputeContextGLMessageGetData {
  method: 'gl.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
  ctorType: string;
}

export interface ComputeContextGLMessageAddKernel {
  method: 'gl.addKernel';
  name: string;
  descriptor: { source: string };
}

export interface ComputeContextGLMessageRunKernel {
  method: 'gl.runKernel';
  descriptor: GLKernelRunDescriptor;
}

export interface ComputeContextGLMessageBeginCapture {
  method: 'gl.beginCapture';
  name: string;
}

export interface ComputeContextGLMessageEndCapture {
  method: 'gl.endCapture';
}

export interface ComputeContextGLMessageReplay {
  method: 'gl.replay';
  name: string;
}

export interface ComputeContextGLMessageResetCaptures {
  method: 'gl.resetCaptures';
}

export type ComputeContextGLMessage =
  | ComputeContextGLMessageAddKernel
  | ComputeContextGLMessageCreateBuffer
  | ComputeContextGLMessageDisposeBuffer
  | ComputeContextGLMessageGetData
  | ComputeContextGLMessageRunKernel
  | ComputeContextGLMessageSetData
  | ComputeContextGLMessageBeginCapture
  | ComputeContextGLMessageEndCapture
  | ComputeContextGLMessageReplay
  | ComputeContextGLMessageResetCaptures;

export class ComputeContextGL {
  tensorBuffers: Map<number, WebGLTensorBuffer> = new Map();
  // Graph capture/replay: record the kernel-dispatch sequence of one step so it
  // can be re-issued from JS in a single call (same idea as the WebGPU backend).
  private capturing: string | null = null;
  private captures: Map<string, GLKernelRunDescriptor[]> = new Map();
  private pinned: Set<number> = new Set();
  async init() {
    await initializeNNWebGLContext();
  }

  getDeviceInfo() {
    const ctx = getNNWebGLContext();
    return {
      maxTextureSize: ctx.maxTextureSize,
      supportsTexture32bit: ctx.supportsTexture32bit,
      supportsTexture16bit: ctx.supportsTexture16bit,
      canReadRedTexture: ctx.canReadRedTexture,
      canReadNon32bitTexture: ctx.canReadNon32bitTexture,
    };
  }

  createBuffer(id: number, textureShape: TensorTextureShape) {
    const tensorBuffer = new WebGLTensorBuffer(textureShape);
    this.tensorBuffers.set(id, tensorBuffer);
  }

  disposeBuffer(id: number) {
    if (this.pinned.has(id)) {
      return; // referenced by a captured graph — keep alive for replay
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

  // Drop every recorded graph and unpin all of their buffers. Sent when a model is
  // released: without it the pins live forever, disposeBuffer keeps refusing every
  // buffer the captured step ever touched, and the freed model's memory is never
  // returned — so the next model allocates on top of it. Safe because a capture is
  // re-recorded on every generate() call; the disposeBuffer messages that follow
  // this one (same FIFO channel) then actually reach the buffers.
  resetCaptures() {
    this.capturing = null;
    this.captures.clear();
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

  setData(id: number, data: ArrayBufferView): void {
    // no pack
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return;
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint16Array> {
    // no pack
    // not necessarily async, but matching WebGPU API
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject();
    }
    // TODO consider data format
    // const data = tb.getDataRawFloat32();
    const data = tb.getDataRaw();
    return Promise.resolve(data.buffer as Uint16Array);
  }

  addKernel(name: string, descriptor: { source: string }) {
    const ctx = getNNWebGLContext();
    ctx.addKernel(name, descriptor.source);
  }

  runKernel(descriptor: GLKernelRunDescriptor) {
    if (this.capturing) {
      this.captures.get(this.capturing)!.push(descriptor);
      for (const inp of descriptor.inputs) {
        this.pinned.add(inp.id);
      }
      this.pinned.add(descriptor.output);
    }
    const ctx = getNNWebGLContext();
    const inputs = descriptor.inputs.map(({ name, id }) => ({
      name,
      buffer: nonNull(this.tensorBuffers.get(id)),
    }));
    const output = nonNull(this.tensorBuffers.get(descriptor.output));
    ctx.runKernel(descriptor.name, inputs, output, descriptor.uniforms);
  }

  mdata: SharedArrayBuffer | null = null;
  mnotify: Int32Array | null = null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  handleMessage(message: ComputeContextGLMessage, worker: Worker) {
    switch (message.method) {
      case 'gl.addKernel':
        this.addKernel(message.name, message.descriptor);
        break;
      case 'gl.createBuffer':
        this.createBuffer(message.id, message.textureShape);
        break;
      case 'gl.disposeBuffer':
        this.disposeBuffer(message.id);
        break;
      case 'gl.getData':
        if (message.data) {
          this.mdata = message.data;
        }
        if (message.notify) {
          this.mnotify = new Int32Array(message.notify);
        }
        this.getData(message.id)
          .then((data) => {
            const ctor = {
              Float32Array: Float32Array,
              Int32Array: Int32Array,
              Uint16Array: Uint16Array,
              Uint8Array: Uint8Array,
            }[message.ctorType];
            if (!ctor) {
              throw new Error('unknown ctor type');
            }
            new ctor(this.mdata!).set(data);
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
          });
        break;
      case 'gl.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gl.setData':
        this.setData(message.id, message.data);
        break;
      case 'gl.beginCapture':
        this.beginCapture((message as ComputeContextGLMessageBeginCapture).name);
        break;
      case 'gl.endCapture':
        this.endCapture();
        break;
      case 'gl.replay':
        this.replay((message as ComputeContextGLMessageReplay).name);
        break;
      case 'gl.resetCaptures':
        this.resetCaptures();
        break;
    }
  }
}
