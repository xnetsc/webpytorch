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
export interface ComputeContextGLMessageGetData {
    method: 'gl.getData';
    id: number;
    data: SharedArrayBuffer;
    notify: SharedArrayBuffer;
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
export type ComputeContextGLMessage = ComputeContextGLMessageAddKernel | ComputeContextGLMessageCreateBuffer | ComputeContextGLMessageDisposeBuffer | ComputeContextGLMessageGetData | ComputeContextGLMessageRunKernel | ComputeContextGLMessageSetData | ComputeContextGLMessageBeginCapture | ComputeContextGLMessageEndCapture | ComputeContextGLMessageReplay;
export declare class ComputeContextGL {
    tensorBuffers: Map<number, WebGLTensorBuffer>;
    private capturing;
    private captures;
    private pinned;
    init(): Promise<void>;
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
    replay(name: string): void;
    setData(id: number, data: ArrayBufferView): void;
    getData(id: number): Promise<Uint16Array>;
    addKernel(name: string, descriptor: {
        source: string;
    }): void;
    runKernel(descriptor: GLKernelRunDescriptor): void;
    mdata: SharedArrayBuffer | null;
    mnotify: Int32Array | null;
    handleMessage(message: ComputeContextGLMessage, worker: Worker): void;
}
