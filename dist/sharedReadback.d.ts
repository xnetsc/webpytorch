/** A readback arena sized to the largest tensor actually requested.
 * The old unconditional 64 MiB allocation remained resident beside large
 * models even when the only readback was a few logits or a token id.
 */
export declare function writeSharedReadbackError(memory: SharedArrayBuffer | null, reason: unknown): void;
export declare function sharedReadbackArena(): {
    begin: (byteLength: number) => {
        memory: SharedArrayBuffer;
        status: Int32Array;
        binding: {
            data?: undefined;
            notify?: undefined;
            error?: undefined;
        } | {
            data: SharedArrayBuffer;
            notify: SharedArrayBuffer;
            error: SharedArrayBuffer;
        };
    };
    errorMessage: () => string;
};
