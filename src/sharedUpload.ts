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
    // A single giant weight must not leave a giant staging arena resident.
    send({ method: `${backend}.releaseUploadMemory` });
    memory = null;
    notify = null;
    status = null;
    capacity = 0;
  }

  function uploadPrepared(id: number, byteOffset: number, byteLength: number,
                          ctorType?: string, operation = 'sharedUpload'): number {
    if (!memory || byteOffset < 0 || byteLength < 0 ||
        byteOffset + byteLength > capacity) {
      throw new Error('shared upload region is outside staging memory');
    }
    Atomics.store(status!, 0, 0);
    send({ method: `${backend}.${operation}`, id, byteOffset, byteLength, ctorType });
    Atomics.wait(status!, 0, 0);
    return Atomics.load(status!, 0);
  }

  function prepare(byteLength: number): Uint8Array {
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

  return { upload, prepare, uploadPrepared, releasePrepared: releaseOversized };
}
