import assert from 'node:assert/strict';
import test from 'node:test';
import { sharedUploader } from '../src/sharedUpload.ts';

for (const backend of ['gl', 'gpu']) {
  test(`${backend} uploads bytes through shared memory with signal-only hot RPC`, () => {
    const calls = [], payloads = [];
    let memory, notify;
    const uploader = sharedUploader(backend, message => {
      calls.push(message);
      if (message.method === `${backend}.uploadMemory`) {
        memory = message.memory;
        notify = new Int32Array(message.notify);
      } else if (message.method === `${backend}.sharedUpload`) {
        assert.equal('data' in message, false);
        payloads.push([...new Uint8Array(memory, 0, message.byteLength)]);
        Atomics.store(notify, 0, 1);
        Atomics.notify(notify, 0);
      }
    }, 16);
    assert.equal(uploader.upload(4, Uint8Array.of(1, 2, 3)), 1);
    assert.equal(uploader.upload(5, Uint8Array.of(4, 5)), 1);
    assert.deepEqual(payloads, [[1, 2, 3], [4, 5]]);
    assert.deepEqual(calls.map(x => x.method), [
      `${backend}.uploadMemory`, `${backend}.sharedUpload`, `${backend}.sharedUpload`,
    ]);
  });

  test(`${backend} does not retain oversized staging data after acknowledgement`, () => {
    const calls = [];
    let notify;
    const uploader = sharedUploader(backend, message => {
      calls.push(message.method);
      if (message.method === `${backend}.uploadMemory`) notify = new Int32Array(message.notify);
      if (message.method === `${backend}.sharedUpload`) {
        Atomics.store(notify, 0, 1);
        Atomics.notify(notify, 0);
      }
    }, 16);
    assert.equal(uploader.upload(1, new Uint8Array(17)), 1);
    assert.deepEqual(calls, [
      `${backend}.uploadMemory`, `${backend}.sharedUpload`, `${backend}.releaseUploadMemory`,
    ]);
  });

  test(`${backend} uploads two prefilled regions without recopying or payload RPC`, () => {
    const calls = [], payloads = [];
    let memory, notify;
    const uploader = sharedUploader(backend, message => {
      calls.push(message);
      if (message.method === `${backend}.uploadMemory`) {
        memory = message.memory;
        notify = new Int32Array(message.notify);
      } else if (message.method === `${backend}.sharedUpload`) {
        assert.equal('data' in message, false);
        payloads.push([...new Uint8Array(memory, message.byteOffset, message.byteLength)]);
        Atomics.store(notify, 0, 1);
        Atomics.notify(notify, 0);
      }
    });
    const staging = uploader.prepare(8);
    staging.set([1, 2, 3, 4, 5, 6, 7, 8]);
    assert.equal(uploader.uploadPrepared(1, 0, 4), 1);
    assert.equal(uploader.uploadPrepared(2, 4, 4), 1);
    uploader.releasePrepared();
    assert.deepEqual(payloads, [[1, 2, 3, 4], [5, 6, 7, 8]]);
    assert.equal(calls.filter(x => x.method === `${backend}.uploadMemory`).length, 1);
  });
}

test('WebGPU metadata uses the same shared staging, not a transferred payload', () => {
  let memory, notify, observed;
  const uploader = sharedUploader('gpu', message => {
    if (message.method === 'gpu.uploadMemory') {
      memory = message.memory;
      notify = new Int32Array(message.notify);
    } else if (message.method === 'gpu.sharedMetaBuffer') {
      assert.equal('data' in message, false);
      observed = [...new Uint8Array(memory, message.byteOffset, message.byteLength)];
      Atomics.store(notify, 0, 1);
      Atomics.notify(notify, 0);
    }
  });
  assert.equal(uploader.upload(7, Uint8Array.of(4, 2, 9), undefined,
    'sharedMetaBuffer'), 1);
  assert.deepEqual(observed, [4, 2, 9]);
});
