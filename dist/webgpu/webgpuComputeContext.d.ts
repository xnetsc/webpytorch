/// <reference types="dist" />
import { WebGPUTensorBuffer } from './webgpuTensorBuffer';
export type WorkGroupDim = 'x' | 'y' | 'z';
export interface GPUKernelRunDescriptor {
    name: string;
    tensors: number[];
    workGroups: {
        [key in WorkGroupDim]: number;
    };
}
export interface ComputeContextGPUMessageCreateBuffer {
    method: 'gpu.createBuffer';
    id: number;
    byteLength: number;
}
export interface ComputeContextGPUMessageCreateMetaBuffer {
    method: 'gpu.createMetaBuffer';
    id: number;
    byteLength: number;
    data: Uint8Array;
}
export interface ComputeContextGPUMessageDisposeBuffer {
    method: 'gpu.disposeBuffer';
    id: number;
}
export interface ComputeContextGPUMessageSetData {
    method: 'gpu.setData';
    id: number;
    data: Uint8Array;
    notify: SharedArrayBuffer;
}
export interface ComputeContextGPUMessageUploadMemory {
    method: 'gpu.uploadMemory';
    memory: SharedArrayBuffer;
    notify: SharedArrayBuffer;
}
export interface ComputeContextGPUMessageSharedUpload {
    method: 'gpu.sharedUpload';
    id: number;
    byteOffset?: number;
    byteLength: number;
}
export interface ComputeContextGPUMessageSharedMetaBuffer {
    method: 'gpu.sharedMetaBuffer';
    id: number;
    byteOffset?: number;
    byteLength: number;
}
export interface ComputeContextGPUMessageReleaseUploadMemory {
    method: 'gpu.releaseUploadMemory';
}
export interface ComputeContextGPUMessageGetData {
    method: 'gpu.getData';
    id: number;
    data: SharedArrayBuffer;
    notify: SharedArrayBuffer;
    error?: SharedArrayBuffer;
}
export interface ComputeContextGPUMessageSampleLogitsDevice {
    method: 'gpu.sampleLogitsDevice';
    id: number;
    count: number;
    temperature: number;
    random: number;
    data?: SharedArrayBuffer;
    notify?: SharedArrayBuffer;
    error?: SharedArrayBuffer;
}
export interface ComputeContextGPUMessageAddKernel {
    method: 'gpu.addKernel';
    name: string;
    descriptor: {
        source: string;
        bindingTypes: GPUBufferBindingType[];
    };
}
export interface ComputeContextGPUMessageRunKernel {
    method: 'gpu.runKernel';
    descriptor: GPUKernelRunDescriptor;
}
export interface ComputeContextGPUMessageBeginCapture {
    method: 'gpu.beginCapture';
    name: string;
}
export interface ComputeContextGPUMessageEndCapture {
    method: 'gpu.endCapture';
}
export interface ComputeContextGPUMessageReplay {
    method: 'gpu.replay';
    name: string;
}
export interface ComputeContextGPUMessageResetCaptures {
    method: 'gpu.resetCaptures';
}
export interface ComputeContextGPUMessageReleaseCapture {
    method: 'gpu.releaseCapture';
    name: string;
}
export type ComputeContextGPUMessage = ComputeContextGPUMessageAddKernel | ComputeContextGPUMessageCreateBuffer | ComputeContextGPUMessageCreateMetaBuffer | ComputeContextGPUMessageDisposeBuffer | ComputeContextGPUMessageGetData | ComputeContextGPUMessageSampleLogitsDevice | ComputeContextGPUMessageRunKernel | ComputeContextGPUMessageSetData | ComputeContextGPUMessageUploadMemory | ComputeContextGPUMessageSharedUpload | ComputeContextGPUMessageSharedMetaBuffer | ComputeContextGPUMessageReleaseUploadMemory | ComputeContextGPUMessageBeginCapture | ComputeContextGPUMessageEndCapture | ComputeContextGPUMessageReplay | ComputeContextGPUMessageResetCaptures | ComputeContextGPUMessageReleaseCapture;
export declare class ComputeContextGPU {
    tensorBuffers: Map<number, WebGPUTensorBuffer>;
    private vocabSampler;
    commandError: unknown;
    private capturing;
    private captures;
    private capturePins;
    private pinned;
    /** What the device was created with: the optional features it has and the limits that
     * decide which kernels can run on it. Detected on the device itself, for any GPU. */
    features(): Record<string, unknown>;
    init(): Promise<void>;
    dispose(): void;
    createBuffer(id: number, byteLength: number): void;
    createMetaBuffer(id: number, byteLength: number, data: Uint8Array): void;
    disposeBuffer(id: number): void;
    beginCapture(name: string): void;
    endCapture(): void;
    releaseCapture(name: string): void;
    resetCaptures(): void;
    replay(name: string): void;
    setData(id: number, data: Uint8Array): void | Promise<void>;
    getData(id: number): Promise<Uint8Array>;
    getDataInto(id: number, target: SharedArrayBuffer): Promise<void>;
    sampleLogitsDevice(id: number, count: number, temperature: number, random: number, target: SharedArrayBuffer): Promise<number>;
    addKernel(name: string, descriptor: {
        source: string;
        bindingTypes: GPUBufferBindingType[];
    }): void;
    runKernel(descriptor: GPUKernelRunDescriptor): void;
    mdata: SharedArrayBuffer | null;
    mnotify: Int32Array | null;
    merror: SharedArrayBuffer | null;
    uploadMemory: SharedArrayBuffer | null;
    uploadNotify: Int32Array | null;
    handleMessage(message: ComputeContextGPUMessage, worker: Worker): void;
}
