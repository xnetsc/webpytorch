/** Readbacks queued now and collected later, for a worker that pipelines GPU work.
 *
 * `getData` waits for everything queued before it, so a worker that must see one result
 * before it queues the next piece of work leaves the GPU idle while it looks. A staged read
 * is recorded behind the work queued so far and lands in a shared slot when the GPU gets
 * there; the worker queues the next piece first and collects afterwards.
 *
 * Each slot carries the sequence number of the last read the GPU thread finished into it.
 * A read the worker never collected (a loop that stopped early) may still finish after the
 * slot has been staged again; the GPU thread drops a completion older than one it already
 * wrote, and the worker waits for its own number, so the late one is harmless.
 */
export function stagedReadArena(slots = 4, slotBytes = 4096) {
  const header = slots * Int32Array.BYTES_PER_ELEMENT;
  const memory = new SharedArrayBuffer(header + slots * slotBytes);
  const status = new Int32Array(memory, 0, slots);
  // Failure text: the blocked worker cannot receive a message until its wait ends.
  const error = new SharedArrayBuffer(4096);
  const errorBytes = new Uint8Array(error);
  const issued = new Int32Array(slots);
  let bound = false;

  /** The GPU thread's half of the arena, the first time only; null afterwards. */
  function binding() {
    if (bound) return null;
    bound = true;
    return { memory, error, slots, slotBytes };
  }

  /** Claim the next sequence number of `slot` for a read of `byteLength` bytes. */
  function stage(slot: number, byteLength: number): number {
    if (!(Number.isInteger(slot) && slot >= 0 && slot < slots)) {
      throw new Error(`staged read slot ${slot} is outside 0..${slots - 1}`);
    }
    if (!(Number.isInteger(byteLength) && byteLength > 0 && byteLength <= slotBytes
          && (byteLength & 3) === 0)) {
      throw new Error(`staged read of ${byteLength} bytes does not fit a ${slotBytes}-byte slot`);
    }
    issued[slot] += 1;
    return issued[slot];
  }

  function errorMessage() {
    let length = 0;
    while (length < errorBytes.length && errorBytes[length]) length++;
    return new TextDecoder().decode(errorBytes.slice(0, length));
  }

  /** Wait for the last read staged in `slot` and return a copy of its bytes. */
  function collect(slot: number, byteLength: number): Uint8Array {
    if (!(Number.isInteger(slot) && slot >= 0 && slot < slots) || issued[slot] === 0) {
      throw new Error(`nothing was staged in slot ${slot}`);
    }
    const want = issued[slot];
    for (;;) {
      const now = Atomics.load(status, slot);
      if (now === want) break;
      if (now < 0) {
        throw new Error('WebGPU staged readback failed: ' + (errorMessage() || 'unknown GPU error'));
      }
      Atomics.wait(status, slot, now);
    }
    return new Uint8Array(memory, header + slot * slotBytes, byteLength).slice();
  }

  return { binding, stage, collect };
}

/** The GPU thread's side: where finished reads go, and which of them still count. */
export function stagedReadTarget(memory: SharedArrayBuffer, error: SharedArrayBuffer,
                                 slots: number, slotBytes: number) {
  const header = slots * Int32Array.BYTES_PER_ELEMENT;
  const status = new Int32Array(memory, 0, slots);
  const done = new Array<number>(slots).fill(0);

  function finish(slot: number, seq: number, data: Uint8Array | null, reason?: unknown) {
    if (!(slot >= 0 && slot < slots) || seq <= done[slot]) return;
    done[slot] = seq;
    if (data) {
      new Uint8Array(memory, header + slot * slotBytes, data.byteLength).set(data);
      Atomics.store(status, slot, seq);
    } else {
      const target = new Uint8Array(error);
      const text = new TextEncoder().encode(String((reason as any)?.message || reason));
      const length = Math.min(text.byteLength, target.byteLength - 1);
      target.set(text.subarray(0, length));
      target[length] = 0;
      Atomics.store(status, slot, -1);
    }
    Atomics.notify(status, slot);
  }

  return { finish, slotBytes };
}
