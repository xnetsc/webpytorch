/** A readback arena sized to the largest tensor actually requested.
 * The old unconditional 64 MiB allocation remained resident beside large
 * models even when the only readback was a few logits or a token id.
 */
export function sharedReadbackArena() {
  const notify = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
  const status = new Int32Array(notify);
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
    const binding = bound ? {} : { data: memory, notify };
    bound = true;
    return { memory, status, binding };
  }

  return { begin };
}
