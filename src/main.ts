import { WgpyBackend } from './backend';
import { ComputeContextGL } from './webgl/webglComputeContext';
import { ComputeContextGPU } from './webgpu/webgpuComputeContext';

export interface WgpyInitOptions {
  // specify the order of backend to try. default: ['webgpu', 'webgl']
  // if ['webgpu', 'webgl'] is specified, webgpu will be tried first, and if it fails, webgl will be tried.
  backendOrder?: WgpyBackend[];
}

export interface WgpyInitResult {
  backend: WgpyBackend;
  dispose: () => void;
}

interface SharedQueue {
  memory: SharedArrayBuffer;
  slotCount: number;
  slotBytes: number;
}

function readShared(queue: SharedQueue | null, slot: number): any[] {
  if (!queue || slot < 0 || slot >= queue.slotCount) throw new Error('invalid shared GPU command slot');
  const header = new Int32Array(queue.memory, 0, queue.slotCount * 2);
  const length = Atomics.load(header, slot * 2 + 1);
  if (Atomics.load(header, slot * 2) !== 1 || length < 0 || length > queue.slotBytes) {
    throw new Error('invalid shared GPU command length');
  }
  const offset = queue.slotCount * 2 * Int32Array.BYTES_PER_ELEMENT + slot * queue.slotBytes;
  // TextDecoder rejects SharedArrayBuffer-backed views in this browser.  Copy
  // only the small command metadata, never tensor payloads.
  return JSON.parse(new TextDecoder().decode(
    new Uint8Array(queue.memory, offset, length).slice()));
}

function releaseShared(queue: SharedQueue | null, slot: number): void {
  if (!queue || slot < 0 || slot >= queue.slotCount) return;
  const header = new Int32Array(queue.memory, 0, queue.slotCount * 2);
  Atomics.store(header, slot * 2, 0);
  Atomics.notify(header, slot * 2);
}

export async function initMain(worker: Worker, options: WgpyInitOptions): Promise<WgpyInitResult> {
  let contextGL: ComputeContextGL | null = null;
  let contextGPU: ComputeContextGPU | null = null;
  let initializedBackend: WgpyBackend | null = null;
  let sharedGL: SharedQueue | null = null;
  let sharedGPU: SharedQueue | null = null;
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error('wgpy: SharedArrayBuffer is not supported');
  }
  for (const backend of options.backendOrder ?? ['webgpu', 'webgl']) {
    if (backend === 'webgl') {
      contextGL = new ComputeContextGL();
      try {
        await contextGL.init();
        initializedBackend = backend;
      } catch (error) {
        console.error(
          `wgpy: failed to initialize WebGL context: ${(error as any)?.message}`
        );
        contextGL = null;
      }
    } else if (backend === 'webgpu') {
      contextGPU = new ComputeContextGPU();
      try {
        await contextGPU.init();
        initializedBackend = backend;
      } catch (error) {
        console.error(
          `wgpy: failed to initialize WebGPU context: ${(error as any)?.message}`
        );
        contextGPU = null;
      }
    } else {
      throw new Error(`wgpy: unknown backend: ${backend}`);
    }
    if (initializedBackend) {
      break;
    }
  }

  const onMessage = (e: MessageEvent) => {
    if (e.data?.__webtorch === 'channels') {
      contextGL?.setResourceStats(e.data.channels?.stat || null);
      return;
    }
    if (e.data?.namespace !== 'wgpy') {
      return;
    }
    if (e.data.method === 'init') {
      // if no backend is initialized, send initComplete with gl=gpu=null, which causes initPromiseReject
      worker.postMessage({
        namespace: 'wgpy',
        method: 'initComplete',
        gl: contextGL ? contextGL.getDeviceInfo() : null, // TODO: send device features
        gpu: contextGPU ? {} : null,
      });
    } else if (e.data.method.startsWith('gl.')) {
      if (contextGL) {
        if (e.data.method === 'gl.sharedQueue') {
          sharedGL = e.data;
        } else {
          try {
            const commands = e.data.method === 'gl.signal'
              ? readShared(sharedGL, e.data.slot) : [e.data];
            for (const command of commands) {
              try { contextGL.handleMessage(command, worker); }
              catch (error) { contextGL.commandError = error; console.error(error); }
            }
          } catch (error) {
            contextGL.commandError = error;
            console.error(error);
          } finally {
            if (e.data.method === 'gl.signal') releaseShared(sharedGL, e.data.slot);
          }
        }
      } else {
        console.error('WebGL context is not initialized. You may have loaded wrong wgpy python package.');
      }
    } else if (e.data.method.startsWith('gpu.')) {
      if (contextGPU) {
        if (e.data.method === 'gpu.sharedQueue') {
          sharedGPU = e.data;
        } else {
          try {
            const commands = e.data.method === 'gpu.signal'
              ? readShared(sharedGPU, e.data.slot) : [e.data];
            for (const command of commands) {
              try { contextGPU.handleMessage(command, worker); }
              catch (error) { contextGPU.commandError = error; console.error(error); }
            }
          } catch (error) {
            contextGPU.commandError = error;
            console.error(error);
          } finally {
            if (e.data.method === 'gpu.signal') releaseShared(sharedGPU, e.data.slot);
          }
        }
      } else {
        console.error('WebGPU context is not initialized. You may have loaded wrong wgpy python package.');
      }
    }
  };
  worker.addEventListener('message', onMessage);

  if (!initializedBackend) {
    throw new Error('wgpy: failed to initialize any backend');
  }

  let disposed = false;
  return {backend: initializedBackend, dispose: () => {
    if (disposed) return;
    disposed = true;
    worker.removeEventListener('message', onMessage);
    try {
      contextGPU?.dispose();
    } finally {
      contextGL?.dispose();
    }
  }};
}
