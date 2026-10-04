import { getNNWebGPUContext } from './webgpuContext';

let webgpuAllocCount = 0;
export const existingBuffers: Set<WebGPUTensorBuffer> = new Set();

// Some WebGPU implementations reject a large mappedAtCreation allocation even when the
// final STORAGE buffer itself is legal.  Model weights can easily contain a single tensor
// larger than that mapping limit (Qwen3-30B has one at 130,744,320 bytes), so never mirror
// an entire destination buffer in one mapped staging allocation.  This only changes the
// transport: every byte is copied to the same destination offset and the stored dtype and
// layout are untouched.
const MAX_MAPPED_UPLOAD_BYTES = 32 * 1024 * 1024;
// Per-token inputs are a few bytes to a few kilobytes.  Mapping a staging buffer and
// submitting one copy command for each of them turns a captured decode step into several
// extra queue submissions per token.  queue.writeBuffer has the same byte-for-byte result
// and obeys queue order after the flush below, without an extra command encoder.
const DIRECT_UPLOAD_BYTES = 64 * 1024;

export interface WebGPUBufferShape {
  byteLength: number;
}

export class WebGPUTensorBuffer {
  gpuBuffer: GPUBuffer;

  // private mappedForWriteFromCPU: boolean;

  constructor(public readonly bufferShape: WebGPUBufferShape, public readonly forMetaBuffer: boolean) {
    const ctx = getNNWebGPUContext();
    ctx.assertAlive?.();
    const usage = forMetaBuffer ?  GPUBufferUsage.STORAGE : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    // if (bufferShape.forReadToCPU) {
    //   usage |= GPUBufferUsage.COPY_SRC;
    // }
    this.gpuBuffer = ctx.device.createBuffer({
      mappedAtCreation: forMetaBuffer, //bufferShape.forWriteFromCPU,
      size: bufferShape.byteLength,
      usage,
    });
    // this.mappedForWriteFromCPU = bufferShape.forWriteFromCPU;
    webgpuAllocCount++;
    existingBuffers.add(this);
  }

  setMetaBufferContent(data: Uint8Array): void {
    const ab = this.gpuBuffer.getMappedRange();
    try {
      new Uint8Array(ab).set(data);
    } finally {
      // A rejected copy must not leave a mapped allocation pinned on the GPU.
      this.gpuBuffer.unmap();
    }
  }

  setDataRaw(data: Uint8Array): void | Promise<void> {
    const ctx = getNNWebGPUContext();
    ctx.assertAlive?.();
    ctx.flush();  // submit pending dispatches before this upload reorders the queue
    if (data.byteLength !== this.gpuBuffer.size) {
      throw new Error(
        `WebGPU upload size mismatch: received ${data.byteLength}, expected ${this.gpuBuffer.size}`
      );
    }
    if ((data.byteLength & 3) !== 0) {
      throw new Error(`WebGPU upload size must be a multiple of four bytes: ${data.byteLength}`);
    }

    if (data.byteLength <= DIRECT_UPLOAD_BYTES) {
      ctx.device.queue.writeBuffer(this.gpuBuffer, 0, data);
      return;
    }

    const stagingBuffers: GPUBuffer[] = [];
    let uploadError: unknown;
    try {
      for (let offset = 0; offset < data.byteLength; offset += MAX_MAPPED_UPLOAD_BYTES) {
        const byteLength = Math.min(MAX_MAPPED_UPLOAD_BYTES, data.byteLength - offset);
        const staging = ctx.device.createBuffer({
          mappedAtCreation: true,
          size: byteLength,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
        });
        stagingBuffers.push(staging);
        try {
          new Uint8Array(staging.getMappedRange()).set(
            new Uint8Array(data.buffer, data.byteOffset + offset, byteLength)
          );
          staging.unmap();
          const commandEncoder = ctx.device.createCommandEncoder();
          commandEncoder.copyBufferToBuffer(staging, 0, this.gpuBuffer, offset, byteLength);
          ctx.device.queue.submit([commandEncoder.finish()]);
        } finally {
          try { staging.unmap(); } catch (_) { /* already unmapped */ }
        }
      }
    } catch (error) {
      uploadError = error;
    }
    // A submitted copy may still retain its source. Drain this tensor's copies before
    // acknowledging the worker or starting the next tensor, so multi-GB loads cannot
    // accumulate another model-sized allocation of in-flight staging buffers.
    return ctx.device.queue.onSubmittedWorkDone()
      .then(() => { ctx.assertAlive?.(); if (uploadError) throw uploadError; })
      .finally(() => {
        for (const staging of stagingBuffers) staging.destroy();
      });
  }

  async getDataRaw(): Promise<Uint8Array> {
    const data = new Uint8Array(this.bufferShape.byteLength);
    await this.getDataInto(data);
    return data;
  }

  async getDataInto(data: Uint8Array): Promise<void> {
    const ctx = getNNWebGPUContext();
    ctx.assertAlive?.();
    ctx.flush();  // submit pending dispatches so this readback sees their results

    if (data.byteLength !== this.bufferShape.byteLength) {
      throw new Error(`WebGPU readback target size mismatch: ${data.byteLength}`);
    }

    const reuse = (globalThis as any).__wgpyReadbackPool === true;
    const dst = reuse ? ctx.rentReadback(this.bufferShape.byteLength)
      : ctx.device.createBuffer({
          size: this.bufferShape.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    let mapped = false;
    try {
      const commandEncoder = ctx.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        this.gpuBuffer,
        0,
        dst,
        0,
        this.bufferShape.byteLength
      );
      ctx.device.queue.submit([commandEncoder.finish()]);
      await dst.mapAsync(GPUMapMode.READ);
      ctx.assertAlive?.();
      mapped = true;
      const arrayBuffer = dst.getMappedRange(),
        buffer_mapped_array = new Uint8Array(arrayBuffer, 0, this.bufferShape.byteLength);
      data.set(buffer_mapped_array);
    } finally {
      if (mapped) dst.unmap();
      // A rejected map (e.g. device loss) must not strand this readback allocation.
      if (reuse) ctx.returnReadback(dst, this.bufferShape.byteLength, mapped);
      else dst.destroy();
    }
  }

  dispose() {
    // Defer the destroy until the next flush — a dispatch already encoded in the
    // pending (unsubmitted) command buffer may still reference this buffer.
    getNNWebGPUContext().deferDispose(this.gpuBuffer);
    webgpuAllocCount--;
    existingBuffers.delete(this);
    (this as { gpuBuffer: GPUBuffer | null }).gpuBuffer = null;
  }
}
