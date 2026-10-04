/** CPU-side MoE routing on JS views, with only small selected-index/weight
 * buffers returned to the GPU. Python chooses the route but never touches the
 * router logits or assembles its output data.
 */
export declare function routeTopK(logits: Float32Array, rows: number, experts: number, k: number, renormalize: boolean): {
    indices: Int32Array;
    weights: Float32Array;
};
export declare function routeTopKInto(logits: Float32Array, rows: number, experts: number, k: number, renormalize: boolean, indices: Int32Array, weights: Float32Array): void;
