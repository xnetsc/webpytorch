/// <reference types="dist" />
import { WebGPUTensorBuffer } from './webgpuTensorBuffer';
export type WorkGroupDim = 'x' | 'y' | 'z';
export interface GPUKernelRunDescriptor {
    name: string;
    tensors: number[];
    workGroups: {
        [key in WorkGroupDim]: number;
    };
    /** Recorded at a capacity, reissued for what a replay has: axis -> (live quantity, mul,
     * div), the count being ceil(live * mul / div). See `replay`. */
    dyn?: {
        [key in WorkGroupDim]?: [string, number, number];
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
export interface ComputeContextGPUMessageTimingBegin {
    method: 'gpu.timingBegin';
}
export interface ComputeContextGPUMessageSubmission {
    method: 'gpu.beginSubmission' | 'gpu.endSubmission';
}
export interface ComputeContextGPUMessageTimingEnd {
    method: 'gpu.timingEnd';
    data?: SharedArrayBuffer;
    notify?: SharedArrayBuffer;
    error?: SharedArrayBuffer;
}
export interface ComputeContextGPUMessageReplay {
    method: 'gpu.replay';
    name: string;
    /** The live quantities of a recording made at a capacity (`GPUKernelRunDescriptor.dyn`). */
    live?: Record<string, number> | null;
}
export interface ComputeContextGPUMessageResetCaptures {
    method: 'gpu.resetCaptures';
}
export interface ComputeContextGPUMessageReleaseCapture {
    method: 'gpu.releaseCapture';
    name: string;
}
export interface ComputeContextGPUMessageSharedUploadMany {
    method: 'gpu.sharedUploadMany';
    parts: Array<[number, number, number] | [number, number, number, number]>;
    ctorType?: string;
}
export interface ComputeContextGPUMessageClearBuffer {
    method: 'gpu.clearBuffer';
    id: number;
}
export interface ComputeContextGPUMessageStageArena {
    method: 'gpu.stageArena';
    memory: SharedArrayBuffer;
    error: SharedArrayBuffer;
    slots: number;
    slotBytes: number;
}
export interface ComputeContextGPUMessageStageRead {
    method: 'gpu.stageRead';
    id: number;
    byteLength: number;
    slot: number;
    seq: number;
}
export type ComputeContextGPUMessage = ComputeContextGPUMessageAddKernel | ComputeContextGPUMessageCreateBuffer | ComputeContextGPUMessageCreateMetaBuffer | ComputeContextGPUMessageDisposeBuffer | ComputeContextGPUMessageGetData | ComputeContextGPUMessageSampleLogitsDevice | ComputeContextGPUMessageRunKernel | ComputeContextGPUMessageSetData | ComputeContextGPUMessageUploadMemory | ComputeContextGPUMessageSharedUpload | ComputeContextGPUMessageSharedMetaBuffer | ComputeContextGPUMessageReleaseUploadMemory | ComputeContextGPUMessageBeginCapture | ComputeContextGPUMessageEndCapture | ComputeContextGPUMessageReplay | ComputeContextGPUMessageTimingBegin | ComputeContextGPUMessageSubmission | ComputeContextGPUMessageTimingEnd | ComputeContextGPUMessageResetCaptures | ComputeContextGPUMessageReleaseCapture | ComputeContextGPUMessageStageArena | ComputeContextGPUMessageStageRead | ComputeContextGPUMessageClearBuffer | ComputeContextGPUMessageSharedUploadMany;
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
    /** Issue a recording again. With `live`, a dispatch recorded with `dyn` rules is issued
     * for the quantities this call has -- a pass recorded at a row capacity serves any row
     * count up to it -- and never with more workgroups than it was recorded with. */
    replay(name: string, live?: Record<string, number> | null): void;
    setData(id: number, data: Uint8Array, prefix?: boolean): void | Promise<void>;
    getData(id: number): Promise<Uint8Array>;
    private stageTarget;
    private stageFree;
    /** Copy `byteLength` bytes of buffer `id` behind everything queued so far, submit, and
     * deliver them to the worker's `slot` when the GPU gets there. Returns at once. */
    stageRead(id: number, byteLength: number, slot: number, seq: number): void;
    getDataInto(id: number, target: SharedArrayBuffer): Promise<void>;
    sampleLogitsDevice(id: number, count: number, temperature: number, random: number, target: SharedArrayBuffer): Promise<number>;
    addKernel(name: string, descriptor: {
        source: string;
        bindingTypes: GPUBufferBindingType[];
    }): void;
    /** After a batch of commands from the producer: start the GPU on them if it is idle. */
    afterBatch(): void;
    runKernel(descriptor: GPUKernelRunDescriptor): void;
    mdata: SharedArrayBuffer | null;
    mnotify: Int32Array | null;
    merror: SharedArrayBuffer | null;
    uploadMemory: SharedArrayBuffer | null;
    uploadNotify: Int32Array | null;
    handleMessage(message: ComputeContextGPUMessage, worker: Worker): void;
}
