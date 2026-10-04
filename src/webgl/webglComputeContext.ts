import { nonNull } from '../util';
import { WorkGroupDim } from '../webgpu/webgpuComputeContext';
import {
  getNNWebGLContext,
  initializeNNWebGLContext,
  disposeNNWebGLContext,
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

export interface ComputeContextGLMessageUploadMemory {
  method: 'gl.uploadMemory';
  memory: SharedArrayBuffer;
  notify: SharedArrayBuffer;
}

export interface ComputeContextGLMessageSharedUpload {
  method: 'gl.sharedUpload';
  id: number;
  byteOffset?: number;
  byteLength: number;
  ctorType: string;
}

export interface ComputeContextGLMessageReleaseUploadMemory {
  method: 'gl.releaseUploadMemory';
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
  | ComputeContextGLMessageUploadMemory
  | ComputeContextGLMessageSharedUpload
  | ComputeContextGLMessageReleaseUploadMemory
  | ComputeContextGLMessageBeginCapture
  | ComputeContextGLMessageEndCapture
  | ComputeContextGLMessageReplay
  | ComputeContextGLMessageResetCaptures;

export class ComputeContextGL {
  tensorBuffers: Map<number, WebGLTensorBuffer> = new Map();
  commandError: unknown = null;
  private resourceStats: Float64Array | null = null;
  private textureBytes: Map<number, number> = new Map();
  private heldTextureBytes = 0;
  private peakTextureBytes = 0;
  // Graph capture/replay: record the kernel-dispatch sequence of one step so it
  // can be re-issued from JS in a single call (same idea as the WebGPU backend).
  private capturing: string | null = null;
  private captures: Map<string, GLKernelRunDescriptor[]> = new Map();
  private capturePins: Map<string, Set<number>> = new Map();
  private pinned: Set<number> = new Set();
  async init() {
    await initializeNNWebGLContext();
  }

  // The SDK's resource panel reads this same SharedArrayBuffer. Keep GPU accounting next
  // to the actual JS texture lifecycle: no Python ledger query or tensor-size RPC in the
  // hot path, and a lost context can be diagnosed while its worker is blocked.
  setResourceStats(memory: SharedArrayBuffer | null): void {
    this.resourceStats = memory ? new Float64Array(memory) : null;
    this.writeResourceStats();
  }

  private writeResourceStats(): void {
    if (!this.resourceStats) return;
    this.resourceStats[0] = this.heldTextureBytes;
    this.resourceStats[1] = this.peakTextureBytes;
    this.resourceStats[2] = this.tensorBuffers.size;
    this.resourceStats[4] = Date.now();
  }

  private textureStorageBytes(buffer: WebGLTensorBuffer): number {
    const type = buffer.textureShape.type;
    const componentBytes = type === WebGL2RenderingContext.UNSIGNED_BYTE ? 1
      : type === WebGL2RenderingContext.HALF_FLOAT ? 2 : 4;
    return buffer.textureLength * componentBytes;
  }

  dispose() {
    this.resetCaptures();
    try {
      for (const tb of this.tensorBuffers.values()) tb.dispose();
    } finally {
      this.tensorBuffers.clear();
      this.textureBytes.clear();
      this.heldTextureBytes = 0;
      this.writeResourceStats();
      disposeNNWebGLContext();
    }
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
    if (this.tensorBuffers.has(id)) throw new Error(`WebGL buffer ${id} already exists`);
    const tensorBuffer = new WebGLTensorBuffer(textureShape);
    this.tensorBuffers.set(id, tensorBuffer);
    const bytes = this.textureStorageBytes(tensorBuffer);
    this.textureBytes.set(id, bytes);
    this.heldTextureBytes += bytes;
    this.peakTextureBytes = Math.max(this.peakTextureBytes, this.heldTextureBytes);
    this.writeResourceStats();
  }

  disposeBuffer(id: number) {
    if (this.pinned.has(id)) {
      return; // referenced by a captured graph — keep alive for replay
    }
    const tb = this.tensorBuffers.get(id);
    if (tb) {
      tb.dispose();
      this.tensorBuffers.delete(id);
      this.heldTextureBytes -= this.textureBytes.get(id) || 0;
      this.textureBytes.delete(id);
      this.writeResourceStats();
    }
  }

  beginCapture(name: string) {
    this.capturing = name;
    this.captures.set(name, []);
    this.capturePins.set(name, new Set());
    this.pinned = new Set(Array.from(this.capturePins.values()).flatMap(ids => [...ids]));
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

  setData(id: number, data: ArrayBufferView): void {
    // no pack
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      throw new Error(`WebGL upload target ${id} was not created`);
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint16Array> {
    if (this.commandError) return Promise.reject(this.commandError);
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

  getDataInto(id: number, target: SharedArrayBuffer, ctorType: string): void {
    if (this.commandError) throw this.commandError;
    const tb = this.tensorBuffers.get(id);
    if (!tb) throw new Error(`WebGL readback target ${id} was not created`);
    const ctor = {
      Float32Array, Int32Array, Uint16Array, Uint8Array,
    }[ctorType];
    if (!ctor) throw new Error(`unknown WebGL readback ctor ${ctorType}`);
    const targetView = new ctor(target);
    const result = tb.getDataRaw(targetView);
    // Matching texture/worker formats let readPixels write straight into shared
    // memory. Preserve the older conversion only for a mismatched format.
    if (result.buffer.buffer !== target) targetView.set(result.buffer);
  }

  addKernel(name: string, descriptor: { source: string }) {
    const ctx = getNNWebGLContext();
    ctx.addKernel(name, descriptor.source);
  }

  runKernel(descriptor: GLKernelRunDescriptor) {
    if (this.capturing) {
      this.captures.get(this.capturing)!.push(descriptor);
      const pins = this.capturePins.get(this.capturing)!;
      for (const inp of descriptor.inputs) {
        pins.add(inp.id);
        this.pinned.add(inp.id);
      }
      pins.add(descriptor.output);
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
  uploadMemory: SharedArrayBuffer | null = null;
  uploadNotify: Int32Array | null = null;
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
        // getData can throw synchronously (e.g. context loss in readPixels), so
        // enter the Promise chain before calling it.  Both sync and async errors
        // must wake the worker waiting in Atomics.wait.
        void Promise.resolve().then(() => this.getDataInto(
          message.id, this.mdata!, message.ctorType
        ))
          .then(() => {
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
            if (String(reason).includes('WebGL context lost')) {
              console.error('WebGL texture ledger at loss:', this.tensorBuffers.size,
                'textures,', (this.heldTextureBytes / 1073741824).toFixed(2),
                'GiB declared storage');
            }
            // The worker is synchronously blocked in Atomics.wait.  A failed
            // readback (notably after context loss) must wake it with an error,
            // otherwise Python hangs forever and can mistake stale bytes for data.
            this.mnotify![0] = -1;
            Atomics.notify(this.mnotify!, 0);
          });
        break;
      case 'gl.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gl.setData':
        this.setData(message.id, message.data);
        break;
      case 'gl.uploadMemory':
        this.uploadMemory = message.memory;
        this.uploadNotify = new Int32Array(message.notify);
        break;
      case 'gl.releaseUploadMemory':
        this.uploadMemory = null;
        this.uploadNotify = null;
        break;
      case 'gl.sharedUpload': {
        const notify = this.uploadNotify;
        if (!notify) throw new Error('WebGL shared upload was not initialized');
        try {
          const offset = message.byteOffset || 0;
          const ctor = {
            Float32Array, Int32Array, Uint16Array, Uint8Array,
          }[message.ctorType];
          if (!ctor) throw new Error(`WebGL upload ctor ${message.ctorType} is unknown`);
          if (!this.uploadMemory || offset < 0 ||
              offset + message.byteLength > this.uploadMemory.byteLength ||
              offset % ctor.BYTES_PER_ELEMENT !== 0 ||
              message.byteLength % ctor.BYTES_PER_ELEMENT !== 0) {
            throw new Error('WebGL shared upload size exceeds staging memory');
          }
          const data = new ctor(this.uploadMemory, offset,
            message.byteLength / ctor.BYTES_PER_ELEMENT);
          this.setData(message.id, data);
          notify[0] = 1;
        } catch (error) {
          this.commandError = error;
          console.error(error);
          notify[0] = -1;
        } finally {
          Atomics.notify(notify, 0);
        }
        break;
      }
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
