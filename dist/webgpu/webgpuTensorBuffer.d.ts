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
    setDataRaw(data: Uint8Array): void;
    getDataRaw(): Promise<Uint8Array>;
    dispose(): void;
}
