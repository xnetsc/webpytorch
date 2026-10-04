/** Select one token without moving a vocabulary-sized logit vector into Python.
 * The input is a mutable JS view of a GPU readback and is reused in place.
 */
export interface SamplingOptions {
    doSample: boolean;
    execution?: 'auto' | 'js' | 'gpu';
    temperature?: number;
    topP?: number;
    topK?: number;
    minP?: number;
    random?: number;
    repetitionPenalty?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    seen?: number[];
    seenCounts?: Map<number, number>;
    eosIds?: number[];
    blockEos?: boolean;
}
export declare function sampleLogits(logits: Float32Array, options: SamplingOptions): number;
