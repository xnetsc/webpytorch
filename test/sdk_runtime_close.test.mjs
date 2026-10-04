import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../webtorch/js/webtorch-main.js', import.meta.url), 'utf8');

test('closing an SDK runtime terminates Python and disposes its GPU backend once', async () => {
  let worker, disposals = 0;
  class FakeWorker {
    listeners = new Map();
    terminated = 0;
    constructor() { worker = this; }
    addEventListener(name, fn) {
      const list = this.listeners.get(name) || [];
      list.push(fn); this.listeners.set(name, list);
    }
    postMessage(message) {
      if (message.method !== 'start') return;
      queueMicrotask(() => {
        for (const fn of this.listeners.get('message') || []) {
          fn({ data: { __wt: 'reply', id: message.id, ok: true,
                       value: { backend: 'webgpu' } } });
        }
      });
    }
    terminate() { this.terminated++; }
  }
  const root = {};
  vm.runInNewContext(source, {
    self: root, location: { href: 'http://localhost/chat/' }, Worker: FakeWorker,
    URL, URLSearchParams, console, navigator: {}, document: {}, queueMicrotask,
  });
  root.webtorch.initMain = async () => ({ backend: 'webgpu',
    tasks: { cancel() {}, resources() {} }, dispose: () => { disposals++; } });
  const api = await root.webtorch.start({ baseURL: '../' });
  const pending = api.stats(); // deliberately unanswered: close must reject it
  api.close(); api.close();
  await assert.rejects(pending, /runtime closed/);
  await assert.rejects(api.stats(), /runtime closed/);
  assert.equal(worker.terminated, 1);
  assert.equal(disposals, 1);
});

test('GPU backend is still disposed when worker termination fails', async () => {
  let disposals = 0;
  class FakeWorker {
    listeners = new Map();
    addEventListener(name, fn) {
      const list = this.listeners.get(name) || [];
      list.push(fn); this.listeners.set(name, list);
    }
    postMessage(message) {
      if (message.method !== 'start') return;
      queueMicrotask(() => {
        for (const fn of this.listeners.get('message') || []) {
          fn({ data: { __wt: 'reply', id: message.id, ok: true,
                       value: { backend: 'webgpu' } } });
        }
      });
    }
    terminate() { throw new Error('worker termination failed'); }
  }
  const root = {};
  vm.runInNewContext(source, {
    self: root, location: { href: 'http://localhost/chat/' }, Worker: FakeWorker,
    URL, URLSearchParams, console, navigator: {}, document: {}, queueMicrotask,
  });
  root.webtorch.initMain = async () => ({ backend: 'webgpu',
    tasks: { cancel() {}, resources() {} }, dispose: () => { disposals++; } });
  const api = await root.webtorch.start({ baseURL: '../' });
  assert.throws(() => api.close(), /worker termination failed/);
  assert.equal(disposals, 1);
  await assert.rejects(api.stats(), /runtime closed/);
});

test('a shared resource snapshot reports zero GPU bytes after release', async () => {
  let worker;
  class FakeWorker {
    listeners = new Map();
    constructor() { worker = this; }
    addEventListener(name, fn) {
      const list = this.listeners.get(name) || [];
      list.push(fn); this.listeners.set(name, list);
    }
    postMessage(message) {
      if (message.method !== 'start') return;
      queueMicrotask(() => this.emit({ __wt: 'reply', id: message.id, ok: true,
                                      value: { backend: 'cpu' } }));
    }
    emit(data) { for (const fn of this.listeners.get('message') || []) fn({ data }); }
    terminate() {}
  }
  const root = {};
  vm.runInNewContext(source, {
    self: root, location: { href: 'http://localhost/chat/' }, Worker: FakeWorker,
    URL, URLSearchParams, console: { warn() {} }, navigator: {}, document: {}, queueMicrotask,
  });
  const api = await root.webtorch.start({ baseURL: '../', backendOrder: [] });
  const memory = new SharedArrayBuffer(40);
  const stats = new Float64Array(memory);
  worker.emit({ __webtorch: 'channels', channels: { stat: memory } });
  assert.equal(api.resources(), null, 'timestamp distinguishes unwritten from zero');
  stats[0] = 4096; stats[1] = 8192; stats[2] = 1; stats[4] = Date.now();
  assert.equal(api.resources().gpuBytes, 4096);
  stats[0] = 0; stats[2] = 0; stats[4]++;
  assert.equal(api.resources().gpuBytes, 0);
  assert.equal(api.resources().gpuPeak, 8192);
  api.close();
});

test('an unavailable or failing requested GPU backend never becomes CPU', async () => {
  const messages = [];
  const root = { navigator: { gpu: {} } };
  vm.runInNewContext(source, {
    self: root, navigator: root.navigator, document: {}, console, SharedArrayBuffer,
  });
  const worker = { postMessage: value => messages.push(value), addEventListener() {} };
  await assert.rejects(root.webtorch.initMain(worker, { backendOrder: ['webgpu'] }),
                       /wgpy-main.js/);
  assert.equal(messages.length, 0);
  root.wgpy = { initMain: async () => { throw new Error('device lost'); } };
  await assert.rejects(root.webtorch.initMain(worker, { backendOrder: ['webgpu'] }),
                       /device lost/);
  assert.equal(messages.length, 0);
  const cpu = await root.webtorch.initMain(worker, { backendOrder: [] });
  assert.equal(cpu.backend, 'cpu');
  assert.equal(messages.length, 1);
  assert.equal(messages[0].backend, 'cpu');
});

test('SDK rejects a worker backend mismatch and releases the runtime', async () => {
  let worker, disposals = 0;
  class FakeWorker {
    listeners = new Map();
    terminated = 0;
    constructor() { worker = this; }
    addEventListener(name, fn) {
      const list = this.listeners.get(name) || [];
      list.push(fn); this.listeners.set(name, list);
    }
    postMessage(message) {
      if (message.method !== 'start') return;
      queueMicrotask(() => {
        for (const fn of this.listeners.get('message') || []) {
          fn({ data: { __wt: 'reply', id: message.id, ok: true,
                       value: { backend: 'cpu' } } });
        }
      });
    }
    terminate() { this.terminated++; }
  }
  const root = {};
  vm.runInNewContext(source, {
    self: root, location: { href: 'http://localhost/chat/' }, Worker: FakeWorker,
    URL, URLSearchParams, console, navigator: {}, document: {}, queueMicrotask,
  });
  root.webtorch.initMain = async () => ({ backend: 'webgpu',
    tasks: { cancel() {}, resources() {} }, dispose: () => { disposals++; } });
  await assert.rejects(root.webtorch.start({ baseURL: '../' }), /backend mismatch/);
  assert.equal(worker.terminated, 1);
  assert.equal(disposals, 1);
});
