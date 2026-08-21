export declare function arrayEqual(a: readonly unknown[], b: readonly unknown[]): boolean;
export declare function arraySum(vec: ArrayLike<number>): number;
export declare function arrayProd(vec: ArrayLike<number>): number;
export declare function nonNull<T>(v: T | null | undefined): T;
export declare function arange(stop: number): number[];
export declare function arange(start: number, stop: number): number[];
export declare function arange(start: number, stop: number, step: number): number[];
export declare function base64ToUint8Array(encodedData: string): Uint8Array;
export declare function uint8ArrayToBase64(array: Uint8Array): string;
