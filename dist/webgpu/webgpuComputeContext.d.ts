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
}
export interface ComputeContextGPUMessageGetData {
    method: 'gpu.getData';
    id: number;
    data: SharedArrayBuffer;
    notify: SharedArrayBuffer;
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
export type ComputeContextGPUMessage = ComputeContextGPUMessageAddKernel | ComputeContextGPUMessageCreateBuffer | ComputeContextGPUMessageCreateMetaBuffer | ComputeContextGPUMessageDisposeBuffer | ComputeContextGPUMessageGetData | ComputeContextGPUMessageRunKernel | ComputeContextGPUMessageSetData | ComputeContextGPUMessageBeginCapture | ComputeContextGPUMessageEndCapture | ComputeContextGPUMessageReplay;
export declare class ComputeContextGPU {
    tensorBuffers: Map<number, WebGPUTensorBuffer>;
    private capturing;
    private captures;
    private pinned;
    init(): Promise<void>;
    createBuffer(id: number, byteLength: number): void;
    createMetaBuffer(id: number, byteLength: number, data: Uint8Array): void;
    disposeBuffer(id: number): void;
    beginCapture(name: string): void;
    endCapture(): void;
    replay(name: string): void;
    setData(id: number, data: Uint8Array): void;
    getData(id: number): Promise<Uint8Array>;
    addKernel(name: string, descriptor: {
        source: string;
        bindingTypes: GPUBufferBindingType[];
    }): void;
    runKernel(descriptor: GPUKernelRunDescriptor): void;
    mdata: SharedArrayBuffer | null;
    mnotify: Int32Array | null;
    handleMessage(message: ComputeContextGPUMessage, worker: Worker): void;
}
