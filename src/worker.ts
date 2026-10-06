import { WgpyBackend } from './backend';
import { GLKernelRunDescriptor } from './webgl/webglComputeContext';
import { TensorTextureShape } from './webgl/webglContext';
import { GPUKernelRunDescriptor } from './webgpu/webgpuComputeContext';
import { commandQueue } from './commandQueue';
import { sharedUploader } from './sharedUpload';
import { sampleLogits, SamplingOptions } from './sampleLogits';
import { routeTopKInto } from './routeTopK';
import { sharedReadbackArena } from './sharedReadback';
import { stagedReadArena } from './stagedRead';
import { fillDecisionKeyMaskCpu, stageDecisionCapture, stageDecisionPacked,
         stageDecisionKeyMask } from './decisionCapture';

export interface WgpyInitWorkerResult {
  backend: WgpyBackend;
}

function dictToObj(dict: any) {
  // Convert python dict to JS object. func({"x":1,"y":[2,3]}) in python
  return dict.toJs({
    dict_converter: Object.fromEntries,
    create_proxies: false,
  });
}

function postToMain(obj: any, transfer: Transferable[] = []) {
  postMessage({ namespace: 'wgpy', ...obj }, transfer);
}

// Also present when the SDK selects CPU: the NumPy destination is a borrowed WASM
// byte view and JavaScript writes it in place, without a Python element loop.
(globalThis as any).decision = {
  fillKeyMask: fillDecisionKeyMaskCpu,
};

// Opt-in, worker-local diagnosis of the complete token-selection boundary. A Python
// caller may set `js.self.__wgpyProfileSample = true` and read the aggregate JSON;
// ordinary inference pays only the final branch and never moves logits to Python.
function recordSampleProfile(backend: string, bytes: number,
                             readMs: number, prepareMs: number, sampleMs: number) {
  const root = globalThis as any;
  if (!root.__wgpyProfileSample) return;
  const all = root.__wgpySampleProfile || (root.__wgpySampleProfile = {});
  const row = all[backend] || (all[backend] = {
    count: 0, bytes: 0, readMs: 0, prepareMs: 0, sampleMs: 0,
  });
  row.count++;
  row.bytes += bytes;
  row.readMs += readMs;
  row.prepareMs += prepareMs;
  row.sampleMs += sampleMs;
}

function initGLInterface(glAvailable: boolean, glDeviceInfo: any) {
  const commands = commandQueue('gl', postToMain);
  const uploader = sharedUploader('gl', postToMain);
  let samplerCounts = new Map<number, number>();
  const readback = sharedReadbackArena();
  let commBuf: any = undefined;
  let commBufUint8Array: Uint8Array | undefined = undefined;
  (globalThis as any).gl = {
    stageDecisionKeyMask: (maskId: number, lengths: any,
                           batch: number, heads: number, padded: number) =>
      stageDecisionKeyMask('gl', () => commands.flush(), uploader,
                           maskId, lengths, batch, heads, padded),
    stageDecisionCapture: (
      xId: number, maskIds: any, ids: any, valid: any, table: any,
      tableType: 'f16' | 'f32', batch: number, length: number, padded: number,
      hidden: number, vocab: number, padId: number, heads: number, window: number,
    ) => {
      try {
        stageDecisionCapture('gl', () => commands.flush(), uploader, xId,
                             typeof maskIds.toJs === 'function' ? dictToObj(maskIds) : maskIds,
                             ids, valid, table, tableType, batch, length, padded, hidden,
                             vocab, padId, heads, window);
      } finally {
        if (typeof maskIds.destroy === 'function') maskIds.destroy();
      }
    },
    isAvailable: () => {
      return glAvailable;
    },
    getDeviceInfo: () => {
      return glDeviceInfo;
    },
    createBuffer: (id: number, textureShape: TensorTextureShape) => {
      commands.enqueue({
        method: 'gl.createBuffer',
        id,
        textureShape: dictToObj(textureShape),
      });
    },
    disposeBuffer: (id: number) => {
      commands.enqueue({ method: 'gl.disposeBuffer', id });
    },
    setCommBuf: (data: any) => {
      if (commBuf) {
        commBuf.release();
      }
      commBuf = data.getBuffer();
      // data.destroy() takes relatively long time, so use the same buffer for every setData / getData.
      data.destroy();
      commBufUint8Array = commBuf.data;
    },
    releaseCommBuf: () => {
      if (commBuf) commBuf.release();
      commBuf = undefined;
      commBufUint8Array = undefined;
    },
    setData: (id: number, ctorType: string, size: number) => {
      const ctor = {
        Float32Array: Float32Array,
        Int32Array: Int32Array,
        Uint16Array: Uint16Array,
        Uint8Array: Uint8Array,
      }[ctorType];
      if (!ctor) {
        throw new Error('ctorType unknown ' + ctorType);
      }

      let dataSrc: Float32Array | Int32Array | Uint16Array | Uint8Array;
      try {
        // same as setData
        dataSrc = new ctor(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          size
        );
      } catch (e) {
        return false;
      }
      commands.flush();
      return uploader.upload(id, new Uint8Array(dataSrc.buffer,
        dataSrc.byteOffset, dataSrc.byteLength), ctorType);
    },
    setDataFromArray: (id: number, data: any, ctorType: string, byteLength: number) => {
      let view: any;
      try {
        view = data.getBuffer();
        if (view.data.byteLength !== byteLength) {
          throw new Error('WebGL direct upload size mismatch');
        }
        commands.flush();
        return uploader.upload(id, new Uint8Array(view.data.buffer,
          view.data.byteOffset, byteLength), ctorType);
      } finally {
        if (view) view.release();
        data.destroy();
      }
    },
    getData: (id: number, ctorType: string, size: number, copyToWasm = true) => {
      const ctor = {
        Float32Array: Float32Array,
        Int32Array: Int32Array,
        Uint16Array: Uint16Array,
        Uint8Array: Uint8Array,
      }[ctorType];
      if (!ctor) {
        throw new Error('ctorType unknown ' + ctorType);
      }
      let dataSrc: Float32Array | Int32Array | Uint16Array | Uint8Array | undefined;
      if (copyToWasm) {
        try {
          dataSrc = new ctor(commBufUint8Array!.buffer,
            commBufUint8Array!.byteOffset, size);
        } catch (e) {
          return false;
        }
      }
      const { memory, status, binding } = readback.begin(size * ctor.BYTES_PER_ELEMENT);
      commands.flush();
      postToMain({ method: 'gl.getData', id, ctorType, ...binding });
      // if buffer[0] = 1 is written before Atomics.wait, it does not wait.
      Atomics.wait(status, 0, 0);

      if (Atomics.load(status, 0) < 0) {
        throw new Error('WebGL readback failed: ' + (readback.errorMessage() || 'unknown GPU error'));
      }

      const placeholderData = new ctor(memory, 0, size);
      if (copyToWasm) {
        dataSrc!.set(placeholderData);
        return true;
      }
      return placeholderData;
    },
    sampleLogits: (id: number, size: number, options: SamplingOptions) => {
      const started = performance.now();
      const data = (globalThis as any).gl.getData(id, 'Float32Array', size, false);
      if (data === -1) throw new Error('WebGL logit readback failed; release and reload the model');
      if (!(data instanceof Float32Array)) throw new Error('WebGL logit readback is not float32');
      const readDone = performance.now();
      const opts = dictToObj(options) as SamplingOptions;
      if (Array.isArray(opts.seen)) {
        samplerCounts = new Map<number, number>();
        for (const seen of opts.seen) {
          if (Number.isInteger(seen) && seen >= 0 && seen < data.length)
            samplerCounts.set(seen, (samplerCounts.get(seen) || 0) + 1);
        }
      }
      opts.seenCounts = samplerCounts;
      const prepared = performance.now();
      const token = sampleLogits(data, opts);
      const sampled = performance.now();
      samplerCounts.set(token, (samplerCounts.get(token) || 0) + 1);
      recordSampleProfile('webgl', size * Float32Array.BYTES_PER_ELEMENT,
        readDone - started, prepared - readDone, sampled - prepared);
      return token;
    },
    routeHost: (logitsId: number, indexId: number, weightId: number,
                rows: number, experts: number, k: number, renormalize: boolean) => {
      const data = (globalThis as any).gl.getData(logitsId, 'Float32Array', rows * experts, false);
      if (!(data instanceof Float32Array)) throw new Error('WebGL MoE router readback failed');
      const count = rows * k;
      const halfBytes = count * 4;
      const staging = uploader.prepare(halfBytes * 2);
      try {
        const indices = new Int32Array(staging.buffer, staging.byteOffset, count);
        const weights = new Float32Array(staging.buffer, staging.byteOffset + halfBytes, count);
        routeTopKInto(data, rows, experts, k, renormalize, indices, weights);
        commands.flush();
        if (uploader.uploadPrepared(indexId, 0, halfBytes, 'Int32Array') < 0 ||
            uploader.uploadPrepared(weightId, halfBytes, halfBytes, 'Float32Array') < 0) {
          throw new Error('WebGL MoE router upload failed');
        }
      } finally {
        uploader.releasePrepared();
      }
    },
    addKernel: (name: string, descriptor: { source: string }) => {
      commands.enqueue({
        method: 'gl.addKernel',
        name,
        descriptor: dictToObj(descriptor),
      });
    },
    runKernel: (descriptor: GLKernelRunDescriptor) => {
      commands.enqueue({ method: 'gl.runKernel', descriptor: dictToObj(descriptor) });
    },
    beginCapture: (name: string) => {
      commands.enqueue({ method: 'gl.beginCapture', name });
    },
    endCapture: () => {
      commands.enqueue({ method: 'gl.endCapture' });
    },
    replay: (name: string) => {
      commands.enqueue({ method: 'gl.replay', name });
    },
    resetCaptures: () => {
      commands.enqueue({ method: 'gl.resetCaptures' });
    },
    releaseCapture: (name: string) => {
      commands.enqueue({ method: 'gl.releaseCapture', name });
    },
    /** Zero a buffer where it lives, in command order: no host data crosses. */
    clearBuffer: (id: number) => {
      commands.enqueue({ method: 'gl.clearBuffer', id });
    },
  };
}

function initGPUInterface(gpuAvailable: boolean, gpuDeviceInfo: any) {
  const commands = commandQueue('gpu', postToMain);
  const uploader = sharedUploader('gpu', postToMain);
  let samplerCounts = new Map<number, number>();
  const readback = sharedReadbackArena();
  const staged = stagedReadArena();
  let commBuf: any = undefined;
  let commBufUint8Array: Uint8Array | undefined = undefined;
  (globalThis as any).gpu = {
    stageDecisionKeyMask: (maskId: number, lengths: any,
                           batch: number, heads: number, padded: number) =>
      stageDecisionKeyMask('gpu', () => commands.flush(), uploader,
                           maskId, lengths, batch, heads, padded),
    stageDecisionCapture: (
      xId: number, maskIds: any, ids: any, valid: any, table: any,
      tableType: 'f16' | 'f32', batch: number, length: number, padded: number,
      hidden: number, vocab: number, padId: number, heads: number, window: number,
    ) => {
      try {
        stageDecisionCapture('gpu', () => commands.flush(), uploader, xId,
                             typeof maskIds.toJs === 'function' ? dictToObj(maskIds) : maskIds,
                             ids, valid, table, tableType, batch, length, padded, hidden,
                             vocab, padId, heads, window);
      } finally {
        if (typeof maskIds.destroy === 'function') maskIds.destroy();
      }
    },
    stageDecisionPacked: (
      xId: number, tokId: number, segId: number, posId: number, gatherId: number,
      ids: any, lengths: any, table: any, tableType: 'f16' | 'f32', batch: number,
      length: number, rows: number, hidden: number, vocab: number, padId: number,
      gatherLen: number, prefix?: boolean,
    ) => stageDecisionPacked('gpu', () => commands.flush(), uploader, xId, tokId, segId, posId,
                             gatherId, ids, lengths, table ?? null, tableType, batch, length,
                             rows, hidden, vocab, padId, gatherLen, !!prefix),
    isAvailable: () => {
      return gpuAvailable;
    },
    getDeviceInfo: () => {
      return gpuDeviceInfo;
    },
    createBuffer: (
      id: number,
      byteLength: number,
    ) => {
      commands.enqueue({
        method: 'gpu.createBuffer',
        id,
        byteLength,
      });
    },
    createMetaBuffer: (
      id: number,
      byteLength: number,
      data: any,
    ) => {
      let view: any;
      try {
        view = data.getBuffer();
        if (view.data.byteLength !== byteLength) {
          throw new Error('WebGPU meta-buffer size mismatch');
        }
        commands.flush();
        const result = uploader.upload(id, new Uint8Array(view.data.buffer,
          view.data.byteOffset, byteLength), undefined, 'sharedMetaBuffer');
        if (result < 0) throw new Error('WebGPU meta-buffer upload failed');
      } finally {
        if (view) view.release();
        data.destroy();
      }
    },
    disposeBuffer: (id: number) => {
      commands.enqueue({ method: 'gpu.disposeBuffer', id });
    },
    setCommBuf: (data: any) => {
      if (commBuf) {
        commBuf.release();
      }
      commBuf = data.getBuffer();
      // data.destroy() takes relatively long time, so use the same buffer for every setData / getData.
      data.destroy();
      commBufUint8Array = commBuf.data;
    },
    releaseCommBuf: () => {
      if (commBuf) commBuf.release();
      commBuf = undefined;
      commBufUint8Array = undefined;
    },
    setData: (id: number, byteLength: number) => {
      // When wasm buffer is reallocated, commBufUint8Array is detached.
      // 'TypeError: Cannot perform Construct on a detached ArrayBuffer' is thrown.
      let dataSrc: Uint8Array;
      try {
        dataSrc = new Uint8Array(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          byteLength
        );
      } catch (e) {
        return false;
      }
      commands.flush();
      // 1 is success, 0 is the detached-WASM-buffer retry above and -1 is a
      // GPU upload failure.  The shared arena is never overwritten before ack.
      return uploader.upload(id, dataSrc);
    },
    setDataFromArray: (id: number, data: any, byteLength: number) => {
      let view: any;
      try {
        view = data.getBuffer();
        if (view.data.byteLength !== byteLength) {
          throw new Error('WebGPU direct upload size mismatch');
        }
        commands.flush();
        return uploader.upload(id, new Uint8Array(view.data.buffer,
          view.data.byteOffset, byteLength));
      } finally {
        if (view) view.release();
        data.destroy();
      }
    },
    getData: (id: number, byteLength: number, copyToWasm = true) => {
      let dataSrc: Uint8Array | undefined;
      if (copyToWasm) {
        try {
          dataSrc = new Uint8Array(commBufUint8Array!.buffer,
            commBufUint8Array!.byteOffset, byteLength);
        } catch (e) {
          return false;
        }
      }
      const { memory, status, binding } = readback.begin(byteLength);
      commands.flush();
      postToMain({ method: 'gpu.getData', id, ...binding });

      // if buffer[0] = 1 is written before Atomics.wait, it does not wait.
      Atomics.wait(status, 0, 0);

      if (Atomics.load(status, 0) < 0) {
        throw new Error('WebGPU readback failed: ' + (readback.errorMessage() || 'unknown GPU error'));
      }

      const placeholderData = new Uint8Array(memory, 0, byteLength);
      if (copyToWasm) {
        dataSrc!.set(placeholderData);
        return true;
      }
      return placeholderData;
    },
    sampleLogits: (id: number, byteLength: number, count: number, options: SamplingOptions) => {
      const started = performance.now();
      const opts = dictToObj(options) as SamplingOptions;
      if (opts.execution === 'gpu' && opts.doSample && (opts.topP ?? 1) >= 1
          && (opts.topK ?? 0) <= 0 && (opts.minP ?? 0) <= 0
          && (opts.repetitionPenalty ?? 1) === 1
          && !(opts.presencePenalty || opts.frequencyPenalty || opts.blockEos)
          && (opts.temperature ?? 1) > 0) {
        const { memory, status, binding } = readback.begin(4);
        commands.flush();
        postToMain({ method: 'gpu.sampleLogitsDevice', id, count,
          temperature: opts.temperature ?? 1, random: opts.random ?? 0, ...binding });
        Atomics.wait(status, 0, 0);
        if (Atomics.load(status, 0) < 0)
          throw new Error('WebGPU device sampling failed: '
            + (readback.errorMessage() || 'unknown GPU error'));
        const token = new DataView(memory).getInt32(0, true);
        if (token < 0 || token >= count)
          throw new Error('WebGPU device sampler selected an invalid token');
        samplerCounts.set(token, (samplerCounts.get(token) || 0) + 1);
        recordSampleProfile('webgpu', 4, performance.now() - started, 0, 0);
        return token;
      }
      const data = (globalThis as any).gpu.getData(id, byteLength, false);
      if (data === -1) throw new Error('WebGPU logit readback failed; release and reload the model');
      if (!(data instanceof Uint8Array)) throw new Error('WebGPU logit readback is not bytes');
      const readDone = performance.now();
      if (Array.isArray(opts.seen)) {
        samplerCounts = new Map<number, number>();
        for (const seen of opts.seen) {
          if (Number.isInteger(seen) && seen >= 0 && seen < count)
            samplerCounts.set(seen, (samplerCounts.get(seen) || 0) + 1);
        }
      }
      opts.seenCounts = samplerCounts;
      const prepared = performance.now();
      const token = sampleLogits(new Float32Array(data.buffer, data.byteOffset, count), opts);
      const sampled = performance.now();
      samplerCounts.set(token, (samplerCounts.get(token) || 0) + 1);
      recordSampleProfile('webgpu', byteLength,
        readDone - started, prepared - readDone, sampled - prepared);
      return token;
    },
    routeHost: (logitsId: number, logitsBytes: number, indexId: number, weightId: number,
                rows: number, experts: number, k: number, renormalize: boolean) => {
      const data = (globalThis as any).gpu.getData(logitsId, logitsBytes, false);
      if (!(data instanceof Uint8Array)) throw new Error('WebGPU MoE router readback failed');
      const count = rows * k;
      const halfBytes = count * 4;
      const staging = uploader.prepare(halfBytes * 2);
      try {
        const indices = new Int32Array(staging.buffer, staging.byteOffset, count);
        const weights = new Float32Array(staging.buffer, staging.byteOffset + halfBytes, count);
        routeTopKInto(new Float32Array(data.buffer, data.byteOffset,
          rows * experts), rows, experts, k, renormalize, indices, weights);
        commands.flush();
        if (uploader.uploadPrepared(indexId, 0, halfBytes) < 0 ||
            uploader.uploadPrepared(weightId, halfBytes, halfBytes) < 0) {
          throw new Error('WebGPU MoE router upload failed');
        }
      } finally {
        uploader.releasePrepared();
      }
    },
    addKernel: (
      name: string,
      descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
    ) => {
      commands.enqueue({
        method: 'gpu.addKernel',
        name,
        descriptor: dictToObj(descriptor),
      });
    },
    runKernel: (descriptor: GPUKernelRunDescriptor) => {
      commands.enqueue({
        method: 'gpu.runKernel',
        descriptor: dictToObj(descriptor),
      });
    },
    beginCapture: (name: string) => {
      commands.enqueue({ method: 'gpu.beginCapture', name });
    },
    endCapture: () => {
      commands.enqueue({ method: 'gpu.endCapture' });
    },
    /** `live`: the quantities of a recording made at a capacity (a JSON object string). */
    replay: (name: string, live?: string | null) => {
      commands.enqueue({ method: 'gpu.replay', name, live: live ? JSON.parse(live) : null });
    },
    resetCaptures: () => {
      commands.enqueue({ method: 'gpu.resetCaptures' });
    },
    releaseCapture: (name: string) => {
      commands.enqueue({ method: 'gpu.releaseCapture', name });
    },
    /** Zero a buffer where it lives, in command order: no host data crosses. */
    clearBuffer: (id: number) => {
      commands.enqueue({ method: 'gpu.clearBuffer', id });
    },
    /** Time the GPU work issued from here to `timingEnd` with the device's own timestamps
     * (where `features().timestamps`). */
    timingBegin: () => {
      commands.enqueue({ method: 'gpu.timingBegin' });
    },
    /** [milliseconds of GPU time since `timingBegin`, its passes summed; the timestamps'
     * step in ns]. Waits for that work. -1 ms where it could not be timed (no timestamps, or
     * more passes than a span holds). */
    timingEnd: (): number[] => {
      const { memory, status, binding } = readback.begin(16);
      commands.flush();
      postToMain({ method: 'gpu.timingEnd', ...binding });
      Atomics.wait(status, 0, 0);
      if (Atomics.load(status, 0) < 0) {
        throw new Error('WebGPU timing failed: ' + (readback.errorMessage() || 'unknown GPU error'));
      }
      const view = new DataView(memory);
      return [view.getFloat64(0, true), view.getFloat64(8, true)];
    },
    /** One crossing per round of a pipelined loop. Queue a replay of `name` (if given) and
     * a staged read of the first `byteLength` bytes of buffer `id` into `stageSlot` (if
     * >= 0), submit, then wait for the read staged earlier in `collectSlot` (if >= 0) and
     * return its bytes. The GPU runs the new work while the caller handles the old. */
    replayStaged: (name: string | null | undefined, id: number, byteLength: number,
                   stageSlot: number, collectSlot: number) => {
      if (name) commands.enqueue({ method: 'gpu.replay', name });
      if (stageSlot >= 0) {
        const seq = staged.stage(stageSlot, byteLength);
        const bind = staged.binding();
        if (bind) {
          commands.flush();             // keep the order: the arena reaches the GPU thread first
          postToMain({ method: 'gpu.stageArena', ...bind });
        }
        commands.enqueue({ method: 'gpu.stageRead', id, byteLength, slot: stageSlot, seq });
      }
      commands.flush();
      return collectSlot >= 0 ? staged.collect(collectSlot, byteLength) : null;
    },
  };
}

export async function initWorker(): Promise<WgpyInitWorkerResult> {
  let initPromiseResolve: (initResult: WgpyInitWorkerResult) => void = () => {
    throw new Error('unexpected call of initPromiseResolve');
  };
  let initPromiseReject: (reason: any) => void = () => {
    throw new Error('unexpected call of initPromiseReject');
  };
  addEventListener('message', (e) => {
    if (e.data.namespace !== 'wgpy') {
      return;
    }
    switch (e.data.method) {
      case 'initComplete':
        let backend: WgpyBackend | null = null;
        if (e.data.gl != null) {
          backend = 'webgl';
          initGLInterface(true, e.data.gl);
        } else {
          initGLInterface(false, null);
        }
        if (e.data.gpu != null) {
          backend = 'webgpu';
          initGPUInterface(true, e.data.gpu);
        } else {
          initGPUInterface(false, null);
        }
        if (backend) {
          initPromiseResolve({ backend });
        } else {
          initPromiseReject(new Error('wgpy: failed to initialize any backend'));
        }
        break;
    }
  });

  postToMain({ method: 'init' });

  return new Promise<WgpyInitWorkerResult>((resolve, reject) => {
    initPromiseResolve = resolve;
    initPromiseReject = reject;
  });
}
