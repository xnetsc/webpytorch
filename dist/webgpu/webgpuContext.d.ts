/// <reference types="dist" />
import { WebGPUTensorBuffer } from './webgpuTensorBuffer';
type WorkGroupDim = 'x' | 'y' | 'z';
export interface WebGPUMetaBufferContentElement {
    value: number;
    type: 'int32' | 'uint32' | 'float32';
}
export interface WebGPUMetaBufferContent {
    elements: WebGPUMetaBufferContentElement[];
}
export interface WebGPURunnerRequest {
    pipelineName: string;
    tensorBuffers: WebGPUTensorBuffer[];
    workGroups: {
        [key in WorkGroupDim]: number;
    };
}
export declare class NNWebGPUContext {
    initialized: boolean;
    isSupported: boolean;
    device: GPUDevice;
    private pipelines;
    private commandEncoder;
    private passEncoder;
    private bindGroupCache;
    private pendingCount;
    private pendingDisposes;
    private readonly flushThreshold;
    constructor();
    initialize(): Promise<void>;
    hasPipeline(name: string): boolean;
    createPipeline(name: string, source: string, bindingTypes: GPUBufferBindingType[]): void;
    private bufferIds;
    private nextBufferId;
    private bufferKey;
    runKernel(request: WebGPURunnerRequest): void;
    flush(): void;
    deferDispose(buffer: GPUBuffer): void;
}
export declare function initializeNNWebGPUContext(): Promise<void>;
export declare function getNNWebGPUContext(): NNWebGPUContext;
export {};
