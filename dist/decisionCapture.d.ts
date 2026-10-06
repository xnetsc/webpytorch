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
    uploadPreparedMany?(parts: Array<[number, number, number] | [number, number, number, number]>, ctor?: string): void;
    releasePrepared(): void;
};
/** Generate and upload the head mask in JS; Python only provides buffer/shape references. */
export declare function stageDecisionKeyMask(backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena, maskId: number, lengthsArg: BufferProxy, batch: number, heads: number, padded: number): void;
/** Browser CPU path writes the same mask directly into NumPy's existing WASM bytes. */
export declare function fillDecisionKeyMaskCpu(targetArg: BufferProxy, lengthsArg: BufferProxy, batch: number, heads: number, padded: number): void;
/** Stage all mutable inputs for one captured batch without a Python data round-trip. */
export declare function stageDecisionCapture(backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena, xId: number, maskIds: Record<string, number>, idsArg: BufferProxy, validArg: BufferProxy, tableArg: BufferProxy, tableType: 'f16' | 'f32', batch: number, length: number, padded: number, hidden: number, vocab: number, padId: number, heads: number, window: number): void;
/** Packed-layout inputs for one captured batch, all from one staging area.
 *
 * The sequences go end to end -- no row is spent padding a short question up to the longest
 * one -- and the total is rounded to `rows` with the pad token. Per packed row: its token
 * as an f32 row number (`tok`, for a vocabulary table on the device) or its embedding row
 * (`embed`, for one kept on the host); its position inside its sequence (`pos`). Per
 * sequence: (first row, length) (`seg`). For every (sequence, position) of the head's
 * (batch, length) layout: the packed row it reads (`gather`); a position past a sequence's
 * end reads that sequence's first row, which the head masks. `ids` is (batch, length),
 * padded; `lengths` holds each real length.
 */
export declare function fillDecisionPacked(embed: Float32Array | null, tok: Float32Array | null, seg: Uint32Array, pos: Uint32Array, gather: Float32Array, table: DecisionSource | null, tableType: 'f16' | 'f32', ids: DecisionSource, lengths: DecisionSource, indexBytes: 4 | 8, batch: number, length: number, rows: number, hidden: number, vocab: number, padId: number): void;
/** `fillDecisionPacked` into the upload arena, then one upload per target. `xId` (embedding
 * rows; needs `table`) or `tokId` (token row numbers) may be -1 when not wanted. `prefix`:
 * the targets were allocated at a capacity and `rows`, `batch` and `gatherLen` are what this
 * call has -- each is written at its target's start and the rest of it left as it was. */
export declare function stageDecisionPacked(backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena, xId: number, tokId: number, segId: number, posId: number, gatherId: number, idsArg: BufferProxy, lengthsArg: BufferProxy, tableArg: BufferProxy | null, tableType: 'f16' | 'f32', batch: number, length: number, rows: number, hidden: number, vocab: number, padId: number, gatherLen?: number, prefix?: boolean): void;
export {};
