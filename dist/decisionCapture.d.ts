/** Prepare a captured encoder's mutable inputs in the worker, not in Pyodide.
 *
 * The destination is the shared upload arena.  Embedding rows and attention masks are
 * written there directly, then the GPU backend consumes the same bytes.  A new question
 * must rewrite every row: graph replay keeps buffer identities, never their old contents.
 */
export type DecisionSource = DataView;
export declare function fillDecisionEmbeddings(out: Float32Array, table: DecisionSource, tableType: 'f16' | 'f32', ids: DecisionSource, indexBytes: 4 | 8, batch: number, length: number, padded: number, hidden: number, vocab: number, padId: number): void;
export declare function fillDecisionMask(out: Float32Array, valid: DecisionSource, indexBytes: 4 | 8, batch: number, length: number, padded: number, heads: number, kind: string, window: number): void;
/** One padded-key mask row per attention head; the query dimension broadcasts. */
export declare function fillDecisionKeyMask(out: Float32Array, lengths: DecisionSource, indexBytes: 4 | 8, batch: number, heads: number, padded: number): void;
type BufferProxy = {
    getBuffer(): {
        data: ArrayBufferView;
        release(): void;
    };
    destroy(): void;
};
type UploadArena = {
    prepare(bytes: number): Uint8Array;
    uploadPrepared(id: number, offset: number, bytes: number, ctor?: string): number;
    releasePrepared(): void;
};
/** Generate and upload the head mask in JS; Python only provides buffer/shape references. */
export declare function stageDecisionKeyMask(backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena, maskId: number, lengthsArg: BufferProxy, batch: number, heads: number, padded: number): void;
/** Browser CPU path writes the same mask directly into NumPy's existing WASM bytes. */
export declare function fillDecisionKeyMaskCpu(targetArg: BufferProxy, lengthsArg: BufferProxy, batch: number, heads: number, padded: number): void;
/** Stage all mutable inputs for one captured batch without a Python data round-trip. */
export declare function stageDecisionCapture(backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena, xId: number, maskIds: Record<string, number>, idsArg: BufferProxy, validArg: BufferProxy, tableArg: BufferProxy, tableType: 'f16' | 'f32', batch: number, length: number, padded: number, hidden: number, vocab: number, padId: number, heads: number, window: number): void;
export {};
