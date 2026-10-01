import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const list = JSON.parse(await readFile(new URL('../chat/models.json', import.meta.url), 'utf8'));
const appSource = await readFile(new URL('../chat/app.js', import.meta.url), 'utf8');

test('chat model list has valid optional metadata and a usable source', () => {
  assert.equal(list.format_version, 1);
  assert.ok(list.models.length > 0);
  const names = new Set();
  for (const [index, model] of list.models.entries()) {
    assert.equal(typeof model.name, 'string', `item ${index} name`);
    assert.ok(model.name.trim(), `item ${index} name`);
    assert.ok(!names.has(model.name), `duplicate name: ${model.name}`);
    names.add(model.name);
    assert.ok(model.repo || model.url, `${model.name} needs repo or url`);
    if (model.size != null) assert.ok(Number.isSafeInteger(model.size) && model.size > 0);
    if (model.hash != null) assert.match(model.hash, /^[a-f0-9]{64}$/i);
    if (model.url != null) assert.doesNotThrow(() => new URL(model.url));
  }
});

test('Vision has a complete browser-readable fallback publication', () => {
  const model = list.models.find(item => item.kind === 'vision-decision');
  assert.ok(model);
  assert.equal(model.probe, 'laya_web.json');
  assert.match(model.url, /^https:\/\/xnetsc\.github\.io\/webpytorch\/chat\/models\//);
  assert.equal(model.size, undefined, 'omitted size exercises selected-entry probing');
});

test('xDecision list exposes only the published Q8 GGUF', () => {
  const models = list.models.filter(item => item.repo === 'mccoysc/xDecision');
  assert.equal(models.length, 1);
  const [model] = models;
  assert.equal(model.file, 'models/gguf/xDecision-Q8_0.gguf');
  assert.equal(model.size, 402546752);
});

function modelSourceHarness(fetch) {
  const start = appSource.indexOf('const chosenModelSources = new Map();');
  const end = appSource.indexOf('\nfunction remoteModelSpec(', start);
  assert.ok(start >= 0 && end > start, 'source-selection implementation is present');
  let now = 0;
  const context = vm.createContext({
    AbortSignal, URL, fetch,
    performance: { now: () => ++now },
  });
  vm.runInContext(`${appSource.slice(start, end)}\nthis.selectSource = applicationModelSource;`, context);
  return context.selectSource;
}

function rangeResponse(length, total = 402546752) {
  let sent = false;
  return {
    ok: true,
    status: 206,
    headers: { get: name => name.toLowerCase() === 'content-range'
      ? `bytes 0-${length - 1}/${total}` : null },
    body: { getReader: () => ({
      read: async () => sent ? { done: true } : (sent = true,
        { done: false, value: new Uint8Array(length) }),
      cancel: async () => {},
    }) },
  };
}

test('a single reachable model source is probed but never throughput-raced', async () => {
  const calls = [];
  const selectSource = modelSourceHarness(async (url, options) => {
    calls.push({ url, range: options.headers.Range });
    if (url.startsWith('https://modelscope.cn/')) throw new Error('not published here');
    return rangeResponse(1);
  });
  const source = await selectSource({
    repo: 'mccoysc/xDecision', probe: 'models/gguf/xDecision-Q8_0.gguf', url: '',
  });
  assert.equal(source.id, 'huggingface');
  assert.equal(calls.length, 2, 'one existence probe per configured hub');
  assert.deepEqual(calls.map(call => call.range), ['bytes=0-0', 'bytes=0-0']);
});

test('multiple reachable model sources are throughput-raced after probing', async () => {
  const calls = [];
  const selectSource = modelSourceHarness(async (url, options) => {
    calls.push({ url, range: options.headers.Range });
    return rangeResponse(options.headers.Range === 'bytes=0-0' ? 1 : 300000);
  });
  await selectSource({ repo: 'org/model', probe: 'model.gguf', url: '' });
  assert.equal(calls.length, 4, 'two probes followed by two throughput samples');
  assert.deepEqual(calls.map(call => call.range), [
    'bytes=0-0', 'bytes=0-0', 'bytes=0-1048575', 'bytes=0-1048575',
  ]);
});

test('browser source selection excludes endpoints without CORS support', () => {
  assert.doesNotMatch(appSource, /endpoint:\s*['"]https:\/\/hf-mirror\.com/);
});
