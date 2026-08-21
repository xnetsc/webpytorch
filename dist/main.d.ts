import { WgpyBackend } from './backend';
export interface WgpyInitOptions {
    backendOrder?: WgpyBackend[];
}
export interface WgpyInitResult {
    backend: WgpyBackend;
}
export declare function initMain(worker: Worker, options: WgpyInitOptions): Promise<WgpyInitResult>;
