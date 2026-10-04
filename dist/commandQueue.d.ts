/** Ordered, bounded bridge between the Pyodide worker and the GPU-owning thread.
 * GPU payloads stay in JS/GPU buffers; only operation descriptors use this queue.
 * A readback or upload must call flush() first to preserve the command order.
 */
export declare function commandQueue(backend: 'gl' | 'gpu', send: (message: Record<string, unknown>) => void, maxCommands?: number): {
    enqueue: (command: Record<string, unknown>) => void;
    flush: () => void;
};
