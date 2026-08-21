import { DType } from '../dtype';
export interface WebGLUniformItem {
    name: string;
    value: number;
    type: 'float' | 'int';
}
export interface TensorTextureShapeFormat {
    internalFormat: number;
    format: number;
    type: number;
}
export declare const tensorTextureShapeFormatR32F: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatR16F: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatR32I: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatR8UI: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatRGBA32F: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatRGBA16F: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatRGBA32I: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare const tensorTextureShapeFormatRGBA8UI: {
    internalFormat: number;
    format: number;
    type: number;
};
export declare function getTensorTextureShapeFormatForDType(dtype: DType, supportsTexture32bit?: boolean): TensorTextureShapeFormat;
export declare const tensorTextureShapeFormatDefault: {
    internalFormat: number;
    format: number;
    type: number;
};
export type TensorTextureShapeDim = '2D' | '2DArray';
export interface TensorTextureShape2D extends TensorTextureShapeFormat {
    dim: '2D';
    width: number;
    height: number;
}
export interface TensorTextureShape2DArray extends TensorTextureShapeFormat {
    dim: '2DArray';
    width: number;
    height: number;
    depth: number;
}
export type TensorTextureShape = TensorTextureShape2D | TensorTextureShape2DArray;
export declare class WebGLTensorBuffer {
    readonly textureShape: TensorTextureShape;
    readonly texture: WebGLTexture;
    ref: number;
    target: number;
    private isBoundToDrawFrameBuffer;
    private readTextureUnitIndices;
    dimPerPixel: number;
    textureLength: number;
    constructor(textureShape: TensorTextureShape);
    dispose(): void;
    bindToReadTexture(unit: number): void;
    unbindFromReadTexture(): void;
    bindToDrawTexture(layer?: number): void;
    unbindFromDrawTexture(): void;
    getDataRawFloat32(): Float32Array;
    getDataRaw(): {
        type: 'Float32Array';
        buffer: Float32Array;
    } | {
        type: 'Uint16Array';
        buffer: Uint16Array;
    } | {
        type: 'Int32Array';
        buffer: Int32Array;
    } | {
        type: 'Uint8Array';
        buffer: Uint8Array;
    };
    setDataRaw(data: ArrayBufferView): void;
    private readPixels2D;
    private readPixels2DArray;
}
export interface WebGLKernelInputBuffer {
    name: string;
    buffer: WebGLTensorBuffer;
}
export type WebGLKernelInput = WebGLKernelInputBuffer;
export declare class NNWebGLContext {
    gl: WebGL2RenderingContext;
    maxTextureSize: number;
    fb: WebGLFramebuffer;
    supportsTexture32bit: boolean;
    supportsTexture16bit: boolean;
    canReadRedTexture: boolean;
    canReadNon32bitTexture: boolean;
    private programs;
    private vshader;
    constructor();
    createArrayBuffer(vertexArray: Float32Array): WebGLBuffer;
    bindArrayBuffer(buffer: WebGLBuffer): void;
    createTexture(textureShape: TensorTextureShape): WebGLTexture;
    createShader(type: number, source: string, name?: string): WebGLShader;
    addKernel(name: string, sourceCode: string): void;
    hasKernel(name: string): boolean;
    compileKernel(sourceCode: string, name?: string): WebGLProgram;
    runKernel(name: string, inputs: WebGLKernelInput[], output: WebGLTensorBuffer, uniforms: WebGLUniformItem[], drawLayer?: number | null): void;
    private runKernelSingleDrawLayer;
}
export declare function initializeNNWebGLContext(): Promise<void>;
export declare function getNNWebGLContext(): NNWebGLContext;
