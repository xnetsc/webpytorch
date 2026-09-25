import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../chat/cache-sw.js', import.meta.url), 'utf8');

function worker(response) {
  let handler = null;
  let fetches = 0;
  let cacheOpens = 0;
  const notes = [];
  const context = vm.createContext({
    URL, Request, Response,
    console: { log() {} },
    self: { location: new URL('https://xnetsc.github.io/webpytorch/chat/'),
      addEventListener() {} },
    fetch: async () => { fetches += 1; return response; },
    caches: {
      match: async () => null,
      keys: async () => [],
      open: async () => { cacheOpens += 1; return {
        put: async () => {}, keys: async () => [], delete: async () => true,
      }; },
    },
    webtorch: {
      handleFetch(fn) { handler = fn; },
      sendToPage(note) { notes.push(note); },
      onPageMessage() {},
    },
  });
  vm.runInContext(source, context);
  return { handler, notes, fetches: () => fetches, cacheOpens: () => cacheOpens };
}

test('range model reads bypass the service-worker HTTP cache', async () => {
  const runtime = worker(new Response('part', {
    status: 206, headers: { 'Content-Range': 'bytes 0-3/100' },
  }));
  const waits = [];
  const response = await runtime.handler(new Request(
    'https://huggingface.co/convaiinnovations/laya-multilingual/resolve/main/model.safetensors',
    { headers: { Range: 'bytes=0-3' } },
  ), { keepUntil(promise) { waits.push(promise); } });
  await Promise.all(waits);
  assert.equal(response.status, 206);
  assert.equal(runtime.fetches(), 1);
  assert.equal(runtime.cacheOpens(), 0);
  assert.deepEqual(runtime.notes, []);
});

test('an unexpected partial response is not handed to CacheStorage', async () => {
  const runtime = worker(new Response('part', {
    status: 206, headers: { 'Content-Range': 'bytes 0-3/100' },
  }));
  const waits = [];
  const response = await runtime.handler(new Request(
    'https://huggingface.co/model.safetensors',
  ), { keepUntil(promise) { waits.push(promise); } });
  await Promise.all(waits);
  assert.equal(response.status, 206);
  assert.equal(runtime.fetches(), 1);
  assert.equal(runtime.cacheOpens(), 0);
  assert.deepEqual(runtime.notes, []);
});
