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
export declare function stagedReadArena(slots?: number, slotBytes?: number): {
    binding: () => {
        memory: SharedArrayBuffer;
        error: SharedArrayBuffer;
        slots: number;
        slotBytes: number;
    } | null;
    stage: (slot: number, byteLength: number) => number;
    collect: (slot: number, byteLength: number) => Uint8Array;
};
/** The GPU thread's side: where finished reads go, and which of them still count. */
export declare function stagedReadTarget(memory: SharedArrayBuffer, error: SharedArrayBuffer, slots: number, slotBytes: number): {
    finish: (slot: number, seq: number, data: Uint8Array | null, reason?: unknown) => void;
    slotBytes: number;
};
