import { WorkGroupDim } from '../webgpu/webgpuComputeContext';
import { TensorTextureShape, WebGLTensorBuffer, WebGLUniformItem } from './webglContext';
export interface GLKernelRunDescriptor {
    name: string;
    inputs: {
        name: string;
        id: number;
    }[];
    output: number;
    uniforms: WebGLUniformItem[];
}
export interface GPUKernelRunDescriptor {
    name: string;
    tensors: number[];
    uniforms: WebGLUniformItem[];
    workGroups: {
        [key in WorkGroupDim]: number;
    };
}
export interface ComputeContextGLMessageCreateBuffer {
    method: 'gl.createBuffer';
    id: number;
    textureShape: TensorTextureShape;
}
export interface ComputeContextGLMessageDisposeBuffer {
    method: 'gl.disposeBuffer';
    id: number;
}
export interface ComputeContextGLMessageSetData {
    method: 'gl.setData';
    id: number;
    data: Float32Array;
}
export interface ComputeContextGLMessageUploadMemory {
    method: 'gl.uploadMemory';
    memory: SharedArrayBuffer;
    notify: SharedArrayBuffer;
}
export interface ComputeContextGLMessageSharedUpload {
    method: 'gl.sharedUpload';
    id: number;
    byteOffset?: number;
    byteLength: number;
    ctorType: string;
}
export interface ComputeContextGLMessageReleaseUploadMemory {
    method: 'gl.releaseUploadMemory';
}
export interface ComputeContextGLMessageGetData {
    method: 'gl.getData';
    id: number;
    data: SharedArrayBuffer;
    notify: SharedArrayBuffer;
    error?: SharedArrayBuffer;
    ctorType: string;
}
export interface ComputeContextGLMessageAddKernel {
    method: 'gl.addKernel';
    name: string;
    descriptor: {
        source: string;
    };
}
export interface ComputeContextGLMessageRunKernel {
    method: 'gl.runKernel';
    descriptor: GLKernelRunDescriptor;
}
export interface ComputeContextGLMessageBeginCapture {
    method: 'gl.beginCapture';
    name: string;
}
export interface ComputeContextGLMessageEndCapture {
    method: 'gl.endCapture';
}
export interface ComputeContextGLMessageReplay {
    method: 'gl.replay';
    name: string;
}
export interface ComputeContextGLMessageResetCaptures {
    method: 'gl.resetCaptures';
}
export interface ComputeContextGLMessageReleaseCapture {
    method: 'gl.releaseCapture';
    name: string;
}
export interface ComputeContextGLMessageClearBuffer {
    method: 'gl.clearBuffer';
    id: number;
}
export type ComputeContextGLMessage = ComputeContextGLMessageClearBuffer | ComputeContextGLMessageAddKernel | ComputeContextGLMessageCreateBuffer | ComputeContextGLMessageDisposeBuffer | ComputeContextGLMessageGetData | ComputeContextGLMessageRunKernel | ComputeContextGLMessageSetData | ComputeContextGLMessageUploadMemory | ComputeContextGLMessageSharedUpload | ComputeContextGLMessageReleaseUploadMemory | ComputeContextGLMessageBeginCapture | ComputeContextGLMessageEndCapture | ComputeContextGLMessageReplay | ComputeContextGLMessageResetCaptures | ComputeContextGLMessageReleaseCapture;
export declare class ComputeContextGL {
    tensorBuffers: Map<number, WebGLTensorBuffer>;
    commandError: unknown;
    private resourceStats;
    private textureBytes;
    private heldTextureBytes;
    private peakTextureBytes;
    private capturing;
    private captures;
    private capturePins;
    private pinned;
    init(): Promise<void>;
    setResourceStats(memory: SharedArrayBuffer | null): void;
    private writeResourceStats;
    private textureStorageBytes;
    dispose(): void;
    getDeviceInfo(): {
        maxTextureSize: number;
        supportsTexture32bit: boolean;
        supportsTexture16bit: boolean;
        canReadRedTexture: boolean;
        canReadNon32bitTexture: boolean;
    };
    createBuffer(id: number, textureShape: TensorTextureShape): void;
    disposeBuffer(id: number): void;
    beginCapture(name: string): void;
    endCapture(): void;
    releaseCapture(name: string): void;
    resetCaptures(): void;
    replay(name: string): void;
    setData(id: number, data: ArrayBufferView): void;
    getData(id: number): Promise<Uint16Array>;
    getDataInto(id: number, target: SharedArrayBuffer, ctorType: string): void;
    addKernel(name: string, descriptor: {
        source: string;
    }): void;
    runKernel(descriptor: GLKernelRunDescriptor): void;
    mdata: SharedArrayBuffer | null;
    mnotify: Int32Array | null;
    merror: SharedArrayBuffer | null;
    uploadMemory: SharedArrayBuffer | null;
    uploadNotify: Int32Array | null;
    handleMessage(message: ComputeContextGLMessage, worker: Worker): void;
}
