/** Reusable worker/main upload staging.  The model data is copied once from
 * Pyodide's WASM heap into shared JS memory, never structured-cloned into an
 * RPC.  The GPU thread acknowledges after consuming that shared view.
 */
export function sharedUploader(
  backend: 'gl' | 'gpu',
  send: (message: Record<string, unknown>) => void,
  retainedLimit = 32 * 1024 * 1024,
) {
  let memory: SharedArrayBuffer | null = null;
  let notify: SharedArrayBuffer | null = null;
  let status: Int32Array | null = null;
  let capacity = 0;
  // A batch sent by `uploadPreparedMany` and not yet acknowledged. Its bytes are still in
  // the staging memory, so nothing may write there until the GPU thread has taken them.
  let pending = false;

  /** Wait for the unacknowledged batch, if any; throw if the GPU thread could not apply it. */
  function settle() {
    if (!pending) return;
    pending = false;
    Atomics.wait(status!, 0, 0);
    if (Atomics.load(status!, 0) < 0) throw new Error('a staged upload batch failed');
  }

  function ensure(byteLength: number) {
    if (memory && capacity >= byteLength) return;
    capacity = byteLength > retainedLimit
      ? byteLength : Math.min(retainedLimit, Math.max(65536, 2 ** Math.ceil(Math.log2(Math.max(1, byteLength)))));
    memory = new SharedArrayBuffer(capacity);
    notify = new SharedArrayBuffer(4);
    status = new Int32Array(notify);
    send({ method: `${backend}.uploadMemory`, memory, notify });
  }

  function releaseOversized() {
    if (capacity <= retainedLimit) return;
    settle();
    // A single giant weight must not leave a giant staging arena resident.
    send({ method: `${backend}.releaseUploadMemory` });
    memory = null;
    notify = null;
    status = null;
    capacity = 0;
  }

  function uploadPrepared(id: number, byteOffset: number, byteLength: number,
                          ctorType?: string, operation = 'sharedUpload'): number {
    settle();
    if (!memory || byteOffset < 0 || byteLength < 0 ||
        byteOffset + byteLength > capacity) {
      throw new Error('shared upload region is outside staging memory');
    }
    Atomics.store(status!, 0, 0);
    send({ method: `${backend}.${operation}`, id, byteOffset, byteLength, ctorType });
    Atomics.wait(status!, 0, 0);
    return Atomics.load(status!, 0);
  }

  /** Several regions of the prepared staging memory into several buffers, in one message
   * and without waiting: the next use of the staging memory waits for the acknowledgement
   * instead. Commands sent after this run after it on the GPU thread (one FIFO channel). */
  function uploadPreparedMany(parts: Array<[number, number, number]>, ctorType?: string): void {
    settle();
    for (const [, byteOffset, byteLength] of parts) {
      if (!memory || byteOffset < 0 || byteLength < 0 || byteOffset + byteLength > capacity) {
        throw new Error('shared upload region is outside staging memory');
      }
    }
    Atomics.store(status!, 0, 0);
    send({ method: `${backend}.sharedUploadMany`, parts, ctorType });
    pending = true;
  }

  function prepare(byteLength: number): Uint8Array {
    settle();
    ensure(byteLength);
    return new Uint8Array(memory!, 0, byteLength);
  }

  function upload(id: number, source: Uint8Array, ctorType?: string,
                  operation = 'sharedUpload'): number {
    const target = prepare(source.byteLength);
    target.set(source);
    try {
      return uploadPrepared(id, 0, source.byteLength, ctorType, operation);
    } finally {
      releaseOversized();
    }
  }

  return { upload, prepare, uploadPrepared, uploadPreparedMany, settle,
           releasePrepared: releaseOversized };
}
