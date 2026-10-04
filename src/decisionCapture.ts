/** Prepare a captured encoder's mutable inputs in the worker, not in Pyodide.
 *
 * The destination is the shared upload arena.  Embedding rows and attention masks are
 * written there directly, then the GPU backend consumes the same bytes.  A new question
 * must rewrite every row: graph replay keeps buffer identities, never their old contents.
 */

export type DecisionSource = DataView;

function numberAt(view: DecisionSource, index: number, bytes: 4 | 8): number {
  return bytes === 8 ? Number(view.getBigInt64(index * 8, true))
    : view.getInt32(index * 4, true);
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const mantissa = bits & 1023;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (1024 + mantissa) * 2 ** (exponent - 25);
}

let halfLookup: Float32Array | undefined;

function halfValues(): Float32Array {
  if (halfLookup) return halfLookup;
  const values = new Float32Array(1 << 16);
  for (let bits = 0; bits < values.length; bits++) values[bits] = halfToFloat(bits);
  halfLookup = values;
  return values;
}

export function fillDecisionEmbeddings(
  out: Float32Array, table: DecisionSource, tableType: 'f16' | 'f32',
  ids: DecisionSource, indexBytes: 4 | 8, batch: number, length: number,
  padded: number, hidden: number, vocab: number, padId: number,
): void {
  if (out.length !== batch * padded * hidden) throw new Error('embedding target shape mismatch');
  if (table.byteLength < vocab * hidden * (tableType === 'f16' ? 2 : 4)) {
    throw new Error('embedding table is shorter than its declared shape');
  }
  for (let b = 0; b < batch; b++) {
    for (let t = 0; t < padded; t++) {
      const id = t < length ? numberAt(ids, b * length + t, indexBytes) : padId;
      if (!Number.isInteger(id) || id < 0 || id >= vocab) {
        throw new Error(`embedding token ${id} is outside vocabulary ${vocab}`);
      }
      const src = id * hidden;
      const dst = (b * padded + t) * hidden;
      if (tableType === 'f16') {
        // A byte view of the original NumPy FP16 table crosses the bridge without a
        // copy. Decode each 16-bit pattern with a worker-local lookup instead of
        // repeating exponent arithmetic for every token and every question.
        const values = halfValues();
        for (let d = 0; d < hidden; d++) out[dst + d] = values[table.getUint16((src + d) * 2, true)];
      } else {
        for (let d = 0; d < hidden; d++) out[dst + d] = table.getFloat32((src + d) * 4, true);
      }
    }
  }
}

export function fillDecisionMask(
  out: Float32Array, valid: DecisionSource, indexBytes: 4 | 8,
  batch: number, length: number, padded: number, heads: number,
  kind: string, window: number,
): void {
  if (out.length !== batch * heads * padded * padded) {
    throw new Error('attention mask target shape mismatch');
  }
  out.fill(-1e9);
  const plane = padded * padded;
  for (let b = 0; b < batch; b++) {
    // Decode validity once per token, not once for every query *and* attention head.
    // The first head is computed directly in the upload buffer; other heads have the
    // identical mask and receive a bulk in-place TypedArray copy from that first plane.
    const live = new Uint8Array(length);
    for (let k = 0; k < length; k++) {
      live[k] = numberAt(valid, b * length + k, indexBytes) !== 0 ? 1 : 0;
    }
    const base = b * heads * plane;
    for (let q = 0; q < padded; q++) {
      const start = kind === 'sliding_attention' && window ? Math.max(0, q - window) : 0;
      const end = kind === 'sliding_attention' && window
        ? Math.min(length, q + window + 1) : length;
      for (let k = start; k < end; k++) {
        if (live[k]) out[base + q * padded + k] = 0;
      }
    }
    const firstHead = out.subarray(base, base + plane);
    for (let h = 1; h < heads; h++) out.set(firstHead, base + h * plane);
  }
}

/** One padded-key mask row per attention head; the query dimension broadcasts. */
export function fillDecisionKeyMask(
  out: Float32Array, lengths: DecisionSource, indexBytes: 4 | 8,
  batch: number, heads: number, padded: number,
): void {
  if (out.length !== batch * heads * padded) throw new Error('decision key mask shape mismatch');
  if (lengths.byteLength < batch * indexBytes) throw new Error('decision lengths are incomplete');
  out.fill(0);
  for (let b = 0; b < batch; b++) {
    const length = numberAt(lengths, b, indexBytes);
    if (!Number.isInteger(length) || length < 1 || length > padded) {
      throw new Error(`decision length ${length} is outside 1..${padded}`);
    }
    for (let h = 0; h < heads; h++) {
      const base = (b * heads + h) * padded;
      out.fill(-1e9, base + length, base + padded);
    }
  }
}

type BufferProxy = {
  getBuffer(): { data: ArrayBufferView; release(): void };
  destroy(): void;
};
type UploadArena = {
  prepare(bytes: number): Uint8Array;
  uploadPrepared(id: number, offset: number, bytes: number, ctor?: string): number;
  releasePrepared(): void;
};

/** Generate and upload the head mask in JS; Python only provides buffer/shape references. */
export function stageDecisionKeyMask(
  backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena,
  maskId: number, lengthsArg: BufferProxy, batch: number, heads: number, padded: number,
): void {
  let lengths: ReturnType<BufferProxy['getBuffer']> | undefined;
  try {
    lengths = lengthsArg.getBuffer();
    const bytes = batch * heads * padded * 4;
    const staging = uploader.prepare(bytes);
    const lengthView = new DataView(lengths.data.buffer, lengths.data.byteOffset,
                                    lengths.data.byteLength);
    const indexBytes = lengths.data.byteLength / batch;
    if (indexBytes !== 4 && indexBytes !== 8) {
      throw new Error('decision lengths need int32 or int64 bytes');
    }
    fillDecisionKeyMask(new Float32Array(staging.buffer, staging.byteOffset, bytes / 4),
                        lengthView, indexBytes, batch, heads, padded);
    flush();
    if (uploader.uploadPrepared(maskId, 0, bytes,
                                backend === 'gl' ? 'Float32Array' : undefined) < 0) {
      throw new Error('decision key mask upload failed');
    }
  } finally {
    uploader.releasePrepared();
    lengths?.release();
    lengthsArg.destroy();
  }
}

/** Browser CPU path writes the same mask directly into NumPy's existing WASM bytes. */
export function fillDecisionKeyMaskCpu(
  targetArg: BufferProxy, lengthsArg: BufferProxy,
  batch: number, heads: number, padded: number,
): void {
  let target: ReturnType<BufferProxy['getBuffer']> | undefined;
  let lengths: ReturnType<BufferProxy['getBuffer']> | undefined;
  try {
    target = targetArg.getBuffer();
    lengths = lengthsArg.getBuffer();
    const lengthView = new DataView(lengths.data.buffer, lengths.data.byteOffset,
                                    lengths.data.byteLength);
    const indexBytes = lengths.data.byteLength / batch;
    if (indexBytes !== 4 && indexBytes !== 8) {
      throw new Error('decision lengths need int32 or int64 bytes');
    }
    if (target.data.byteLength !== batch * heads * padded * 4) {
      throw new Error('decision CPU key mask byte length mismatch');
    }
    fillDecisionKeyMask(new Float32Array(target.data.buffer, target.data.byteOffset,
                                         target.data.byteLength / 4),
                        lengthView, indexBytes, batch, heads, padded);
  } finally {
    target?.release(); lengths?.release();
    targetArg.destroy(); lengthsArg.destroy();
  }
}

/** Stage all mutable inputs for one captured batch without a Python data round-trip. */
export function stageDecisionCapture(
  backend: 'gl' | 'gpu', flush: () => void, uploader: UploadArena,
  xId: number, maskIds: Record<string, number>, idsArg: BufferProxy,
  validArg: BufferProxy, tableArg: BufferProxy, tableType: 'f16' | 'f32',
  batch: number, length: number, padded: number, hidden: number,
  vocab: number, padId: number, heads: number, window: number,
): void {
  let ids: ReturnType<BufferProxy['getBuffer']> | undefined;
  let valid: ReturnType<BufferProxy['getBuffer']> | undefined;
  let table: ReturnType<BufferProxy['getBuffer']> | undefined;
  try {
    ids = idsArg.getBuffer();
    valid = validArg.getBuffer();
    table = tableArg.getBuffer();
    const kinds = Object.keys(maskIds);
    const embedCount = batch * padded * hidden;
    const maskCount = batch * heads * padded * padded;
    const embedBytes = embedCount * 4;
    const maskBytes = maskCount * 4;
    const staging = uploader.prepare(embedBytes + kinds.length * maskBytes);
    const iv = new DataView(ids.data.buffer, ids.data.byteOffset, ids.data.byteLength);
    const vv = new DataView(valid.data.buffer, valid.data.byteOffset, valid.data.byteLength);
    const tv = new DataView(table.data.buffer, table.data.byteOffset, table.data.byteLength);
    const indexBytes = ids.data.byteLength / (batch * length);
    const validBytes = valid.data.byteLength / (batch * length);
    if ((indexBytes !== 4 && indexBytes !== 8) || validBytes !== indexBytes) {
      throw new Error('decision capture needs matching int32/int64 token and mask buffers');
    }
    fillDecisionEmbeddings(
      new Float32Array(staging.buffer, staging.byteOffset, embedCount), tv, tableType,
      iv, indexBytes, batch, length, padded, hidden, vocab, padId);
    for (let i = 0; i < kinds.length; i++) {
      fillDecisionMask(
        new Float32Array(staging.buffer, staging.byteOffset + embedBytes + i * maskBytes,
                         maskCount), vv, indexBytes, batch, length, padded, heads,
        kinds[i], window);
    }
    flush();
    const ctor = backend === 'gl' ? 'Float32Array' : undefined;
    if (uploader.uploadPrepared(xId, 0, embedBytes, ctor) < 0) {
      throw new Error('decision capture embedding upload failed');
    }
    for (let i = 0; i < kinds.length; i++) {
      if (uploader.uploadPrepared(maskIds[kinds[i]], embedBytes + i * maskBytes,
                                  maskBytes, ctor) < 0) {
        throw new Error(`decision capture ${kinds[i]} mask upload failed`);
      }
    }
  } finally {
    uploader.releasePrepared();
    ids?.release(); valid?.release(); table?.release();
    idsArg.destroy(); validArg.destroy(); tableArg.destroy();
  }
}
