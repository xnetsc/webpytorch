/** A readback arena sized to the largest tensor actually requested.
 * The old unconditional 64 MiB allocation remained resident beside large
 * models even when the only readback was a few logits or a token id.
 */
export function writeSharedReadbackError(memory: SharedArrayBuffer | null, reason: unknown): void {
  if (!memory) return;
  const target = new Uint8Array(memory);
  const message = String((reason as any)?.message || reason);
  const encoded = new TextEncoder().encode(message);
  const length = Math.min(encoded.byteLength, target.byteLength - 1);
  target.set(encoded.subarray(0, length));
  target[length] = 0;
}

export function sharedReadbackArena() {
  const notify = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
  const status = new Int32Array(notify);
  // Failure text uses shared memory too: the blocked worker cannot receive an RPC
  // until Atomics.wait ends. Keep this tiny and transfer it only on first binding.
  const error = new SharedArrayBuffer(4096);
  const errorBytes = new Uint8Array(error);
  let memory: SharedArrayBuffer | null = null;
  let capacity = 0;
  let bound = false;

  function begin(byteLength: number) {
    if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
      throw new Error('invalid shared readback length');
    }
    if (!memory || capacity < byteLength) {
      capacity = Math.max(65536, 2 ** Math.ceil(Math.log2(Math.max(1, byteLength))));
      memory = new SharedArrayBuffer(capacity);
      bound = false;
    }
    Atomics.store(status, 0, 0);
    errorBytes[0] = 0;
    const binding = bound ? {} : { data: memory, notify, error };
    bound = true;
    return { memory, status, binding };
  }

  function errorMessage() {
    let length = 0;
    while (length < errorBytes.length && errorBytes[length]) length++;
    return new TextDecoder().decode(errorBytes.slice(0, length));
  }

  return { begin, errorMessage };
}
