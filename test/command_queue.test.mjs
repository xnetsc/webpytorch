import assert from 'node:assert/strict';
import test from 'node:test';
import { commandQueue } from '../src/commandQueue.ts';

for (const backend of ['gl', 'gpu']) {
  test(`${backend} queues ordered JS commands and flushes at the bound`, async () => {
    const sent = [], decoded = [];
    let shared;
    const queue = commandQueue(backend, message => {
      sent.push(message);
      if (message.method === `${backend}.sharedQueue`) shared = message;
      if (message.method === `${backend}.signal`) {
        const head = new Int32Array(shared.memory, 0, shared.slotCount * 2);
        const offset = shared.slotCount * 8 + message.slot * shared.slotBytes;
        const bytes = new Uint8Array(shared.memory, offset,
          Atomics.load(head, message.slot * 2 + 1));
        decoded.push(...JSON.parse(new TextDecoder().decode(bytes)));
        Atomics.store(head, message.slot * 2, 0);
        Atomics.notify(head, message.slot * 2);
      }
    }, 3);
    queue.enqueue({ method: `${backend}.createBuffer`, id: 1 });
    queue.enqueue({ method: `${backend}.runKernel`, id: 2 });
    assert.equal(sent.length, 0);
    queue.enqueue({ method: `${backend}.disposeBuffer`, id: 1 });
    assert.deepEqual(sent.map(x => x.method),
      [`${backend}.sharedQueue`, `${backend}.signal`]);
    assert.deepEqual(decoded, [
      { method: `${backend}.createBuffer`, id: 1 },
      { method: `${backend}.runKernel`, id: 2 },
      { method: `${backend}.disposeBuffer`, id: 1 },
    ]);
    await Promise.resolve();
    assert.equal(sent.length, 2, 'the scheduled microtask does not duplicate a flushed batch');
  });

  test(`${backend} explicit barrier and microtask both preserve command order`, async () => {
    const decoded = [];
    let shared;
    const queue = commandQueue(backend, message => {
      if (message.method === `${backend}.sharedQueue`) shared = message;
      if (message.method === `${backend}.signal`) {
        const head = new Int32Array(shared.memory, 0, shared.slotCount * 2);
        const offset = shared.slotCount * 8 + message.slot * shared.slotBytes;
        const bytes = new Uint8Array(shared.memory, offset,
          Atomics.load(head, message.slot * 2 + 1));
        decoded.push(...JSON.parse(new TextDecoder().decode(bytes)));
        Atomics.store(head, message.slot * 2, 0);
        Atomics.notify(head, message.slot * 2);
      }
    });
    queue.enqueue({ method: `${backend}.createBuffer`, id: 1 });
    queue.flush(); // upload/readback barrier
    queue.enqueue({ method: `${backend}.runKernel`, id: 2 });
    await Promise.resolve();
    assert.deepEqual(decoded, [
      { method: `${backend}.createBuffer`, id: 1 },
      { method: `${backend}.runKernel`, id: 2 },
    ]);
  });
}
