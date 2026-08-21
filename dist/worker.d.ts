import { WgpyBackend } from './backend';
export interface WgpyInitWorkerResult {
    backend: WgpyBackend;
}
export declare function initWorker(): Promise<WgpyInitWorkerResult>;
