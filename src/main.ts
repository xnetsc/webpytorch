import { WgpyBackend } from './backend';
import { ComputeContextGPU } from './webgpu/webgpuComputeContext';

// Capture the bundle URL while its script is executing. The GPU worker inherits
// its cache-busting query, so a page never mixes two builds of the backend.
const mainScriptUrl = typeof document === 'undefined' ? ''
  : (document.currentScript as HTMLScriptElement | null)?.src
    || Array.from(document.scripts).find(s => /wgpy-main\.js(?:\?|$)/.test(s.src))?.src || '';

async function startGLWorker(): Promise<{ worker: Worker; deviceInfo: any }> {
  if (!mainScriptUrl) throw new Error('wgpy-main.js script URL is unavailable');
  const url = new URL('wgpy-gl-worker.js', mainScriptUrl);
  url.search = new URL(mainScriptUrl).search;
  const worker = new Worker(url.href);
  try {
    const deviceInfo = await new Promise<any>((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error('WebGL worker initialization exceeded 10 seconds'));
      }, 10_000);
      const onMessage = (event: MessageEvent) => {
        if (event.data?.method === 'ready') {
          cleanup();
          resolve(event.data.deviceInfo);
        } else if (event.data?.method === 'error') {
          cleanup();
          reject(new Error(event.data.error));
        }
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || 'WebGL worker failed to start'));
      };
      const cleanup = () => {
        clearTimeout(timeout);
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      worker.postMessage({ method: 'init' });
    });
    return { worker, deviceInfo };
  } catch (error) {
    worker.terminate();
    throw error;
  }
}

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
  let glWorker: Worker | null = null;
  let glDeviceInfo: any = null;
  let contextGPU: ComputeContextGPU | null = null;
  let initializedBackend: WgpyBackend | null = null;
  let sharedGPU: SharedQueue | null = null;
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error('wgpy: SharedArrayBuffer is not supported');
  }
  for (const backend of options.backendOrder ?? ['webgpu', 'webgl']) {
    if (backend === 'webgl') {
      try {
        const started = await startGLWorker();
        glWorker = started.worker;
        glDeviceInfo = started.deviceInfo;
        initializedBackend = backend;
      } catch (error) {
        throw new Error(`wgpy: failed to initialize WebGL context: ${(error as any)?.message}`);
      }
    } else if (backend === 'webgpu') {
      contextGPU = new ComputeContextGPU();
      try {
        await contextGPU.init();
        initializedBackend = backend;
      } catch (error) {
        try { contextGPU.dispose(); } catch (_) { /* preserve initialization failure */ }
        contextGPU = null;
        throw new Error(`wgpy: failed to initialize WebGPU context: ${(error as any)?.message}`);
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
      glWorker?.postMessage({ __webtorch: 'channels', stat: e.data.channels?.stat || null });
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
        gl: glWorker ? glDeviceInfo : null,
        gpu: contextGPU ? contextGPU.features() : null,
      });
    } else if (e.data.method.startsWith('gl.')) {
      if (glWorker) {
        glWorker.postMessage(e.data);
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
            contextGPU.afterBatch();
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
      glWorker?.postMessage({ method: 'dispose' });
      glWorker = null;
    }
  }};
}
