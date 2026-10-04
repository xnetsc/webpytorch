/** Ordered, bounded bridge between the Pyodide worker and the GPU-owning thread.
 * GPU payloads stay in JS/GPU buffers; only operation descriptors use this queue.
 * A readback or upload must call flush() first to preserve the command order.
 */
export function commandQueue(
  backend: 'gl' | 'gpu',
  send: (message: Record<string, unknown>) => void,
  maxCommands = 128,
) {
  let pending: Record<string, unknown>[] = [];
  let scheduled = false;
  const slotCount = 8;
  const slotBytes = 512 * 1024;
  const headerBytes = slotCount * 2 * Int32Array.BYTES_PER_ELEMENT;
  let memory: SharedArrayBuffer | null = null;
  let header: Int32Array | null = null;
  let slotCursor = 0;
  const encoder = new TextEncoder();

  const shared = () => {
    if (!memory) {
      memory = new SharedArrayBuffer(headerBytes + slotCount * slotBytes);
      header = new Int32Array(memory, 0, slotCount * 2);
      // One-time shared-memory setup; all hot-path RPCs below are signals only.
      send({ method: `${backend}.sharedQueue`, memory, slotCount, slotBytes });
    }
    return memory;
  };

  const sendShared = (commands: Record<string, unknown>[]) => {
    const bytes = encoder.encode(JSON.stringify(commands));
    if (bytes.byteLength > slotBytes) {
      if (commands.length === 1) {
        // Shader source registration is low-frequency and may exceed a slot.
        send(commands[0]);
      } else {
        const middle = Math.floor(commands.length / 2);
        sendShared(commands.slice(0, middle));
        sendShared(commands.slice(middle));
      }
      return;
    }
    const buffer = shared();
    const slot = slotCursor;
    const stateIndex = slot * 2;
    while (Atomics.load(header!, stateIndex) !== 0) {
      Atomics.wait(header!, stateIndex, 1);
    }
    new Uint8Array(buffer, headerBytes + slot * slotBytes, bytes.length).set(bytes);
    Atomics.store(header!, stateIndex + 1, bytes.length);
    Atomics.store(header!, stateIndex, 1);
    send({ method: `${backend}.signal`, slot });
    slotCursor = (slot + 1) % slotCount;
  };

  const flush = () => {
    scheduled = false;
    if (!pending.length) return;
    const commands = pending;
    pending = [];
    sendShared(commands);
  };
  const enqueue = (command: Record<string, unknown>) => {
    pending.push(command);
    if (pending.length >= maxCommands) {
      flush();
    } else if (!scheduled) {
      scheduled = true;
      queueMicrotask(flush);
    }
  };
  return { enqueue, flush };
}
