/**
 * Data type of tensor element.
 */
export type DType = 'float32' | 'int32' | 'uint8' | 'bool';
export declare const DTypeDefault: DType;
export type TypedArrayTypes = Float32Array | Int32Array | Uint8Array;
export type TypedArrayConstructor = Float32ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor;
export declare const TypedArrayForDType: {
    [dtype in DType]: TypedArrayConstructor;
};
