import { ComputeContextGL, ComputeContextGLMessage } from './webglComputeContext';

interface SharedQueue {
  memory: SharedArrayBuffer;
  slotCount: number;
  slotBytes: number;
}

let context: ComputeContextGL | null = null;
let queue: SharedQueue | null = null;

function readShared(shared: SharedQueue | null, slot: number): ComputeContextGLMessage[] {
  if (!shared || slot < 0 || slot >= shared.slotCount) {
    throw new Error('invalid shared WebGL command slot');
  }
  const header = new Int32Array(shared.memory, 0, shared.slotCount * 2);
  const length = Atomics.load(header, slot * 2 + 1);
  if (Atomics.load(header, slot * 2) !== 1 || length < 0 || length > shared.slotBytes) {
    throw new Error('invalid shared WebGL command length');
  }
  const offset = shared.slotCount * 2 * Int32Array.BYTES_PER_ELEMENT + slot * shared.slotBytes;
  // Commands are small metadata. Tensor payloads stay in shared memory.
  return JSON.parse(new TextDecoder().decode(
    new Uint8Array(shared.memory, offset, length).slice()));
}

function releaseShared(shared: SharedQueue | null, slot: number): void {
  if (!shared || slot < 0 || slot >= shared.slotCount) return;
  const header = new Int32Array(shared.memory, 0, shared.slotCount * 2);
  Atomics.store(header, slot * 2, 0);
  Atomics.notify(header, slot * 2);
}

self.addEventListener('message', (event: MessageEvent) => {
  const message = event.data;
  if (message.method === 'init') {
    void (async () => {
      try {
        const next = new ComputeContextGL();
        await next.init();
        context = next;
        self.postMessage({ method: 'ready', deviceInfo: next.getDeviceInfo() });
      } catch (error) {
        self.postMessage({ method: 'error', error: String(error) });
      }
    })();
    return;
  }
  if (message.method === 'dispose') {
    try { context?.dispose(); }
    finally { context = null; self.close(); }
    return;
  }
  if (!context) return;
  if (message.__webtorch === 'channels') {
    context.setResourceStats(message.stat || null);
    return;
  }
  if (message.namespace !== 'wgpy' || !message.method.startsWith('gl.')) return;
  if (message.method === 'gl.sharedQueue') {
    queue = message;
    return;
  }
  try {
    const commands = message.method === 'gl.signal'
      ? readShared(queue, message.slot) : [message];
    for (const command of commands) {
      try { context.handleMessage(command, null as unknown as Worker); }
      catch (error) { context.commandError = error; console.error(error); }
    }
  } catch (error) {
    context.commandError = error;
    console.error(error);
  } finally {
    if (message.method === 'gl.signal') releaseShared(queue, message.slot);
  }
});
