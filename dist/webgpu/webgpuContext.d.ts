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
    adapterFacts: {
        vendor: string;
        architecture: string;
        subgroupMinSize: number;
        subgroupMaxSize: number;
    };
    private deviceLostReason;
    private pipelines;
    private pendingPipelineChecks;
    private pipelineError;
    private commandEncoder;
    private passEncoder;
    private bindGroupCache;
    private pendingCount;
    private inflight;
    private pendingDisposes;
    private readbackPool;
    private diagnosticQuery;
    private diagnosticPassIndex;
    private selectedQuery;
    private selectedNames;
    private readonly flushThreshold;
    constructor();
    initialize(): Promise<void>;
    assertAlive(): void;
    private trackPipelineCheck;
    assertPipelinesReady(): Promise<void>;
    hasPipeline(name: string): boolean;
    createPipeline(name: string, source: string, bindingTypes: GPUBufferBindingType[]): void;
    private bufferIds;
    private nextBufferId;
    private bufferKey;
    runKernel(request: WebGPURunnerRequest): void;
    /** Submit what is pending if the GPU has nothing to do; called after each batch of
     * commands the producer sends. A diagnostic pass being timed is left whole. */
    kick(): void;
    flush(): void;
    /** Copy `byteLength` bytes of `src` into `dst` behind every dispatch encoded so far, in
     * the same command buffer, and submit it. The copy sees those dispatches' results with no
     * separate submission; `dst` may be mapped as soon as this returns. */
    copyAndSubmit(src: GPUBuffer, dst: GPUBuffer, byteLength: number): void;
    deferDispose(buffer: GPUBuffer): void;
    rentReadback(byteLength: number): GPUBuffer;
    returnReadback(buffer: GPUBuffer, byteLength: number, reusable: boolean): void;
    dispose(): void;
}
export declare function initializeNNWebGPUContext(): Promise<void>;
export declare function getNNWebGPUContext(): NNWebGPUContext;
export declare function disposeNNWebGPUContext(): void;
export {};
