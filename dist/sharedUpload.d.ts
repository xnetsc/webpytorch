/** Reusable worker/main upload staging.  The model data is copied once from
 * Pyodide's WASM heap into shared JS memory, never structured-cloned into an
 * RPC.  The GPU thread acknowledges after consuming that shared view.
 */
export declare function sharedUploader(backend: 'gl' | 'gpu', send: (message: Record<string, unknown>) => void, retainedLimit?: number): {
    upload: (id: number, source: Uint8Array, ctorType?: string, operation?: string) => number;
    prepare: (byteLength: number) => Uint8Array;
    uploadPrepared: (id: number, byteOffset: number, byteLength: number, ctorType?: string, operation?: string) => number;
    releasePrepared: () => void;
};
