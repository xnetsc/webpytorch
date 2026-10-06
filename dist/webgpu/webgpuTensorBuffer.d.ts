/// <reference types="dist" />
export declare const existingBuffers: Set<WebGPUTensorBuffer>;
export interface WebGPUBufferShape {
    byteLength: number;
}
export declare class WebGPUTensorBuffer {
    readonly bufferShape: WebGPUBufferShape;
    readonly forMetaBuffer: boolean;
    gpuBuffer: GPUBuffer;
    constructor(bufferShape: WebGPUBufferShape, forMetaBuffer: boolean);
    setMetaBufferContent(data: Uint8Array): void;
    /** `prefix`: `data` may be shorter than the buffer and lands at its start -- the live rows
     * of a buffer allocated at a capacity, the rest left as it was. */
    setDataRaw(data: Uint8Array, prefix?: boolean): void | Promise<void>;
    getDataRaw(): Promise<Uint8Array>;
    getDataInto(data: Uint8Array): Promise<void>;
    dispose(): void;
}
