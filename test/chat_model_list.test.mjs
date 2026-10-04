import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const list = JSON.parse(await readFile(new URL('../chat/models.json', import.meta.url), 'utf8'));
const appSource = await readFile(new URL('../chat/app.js', import.meta.url), 'utf8');
const htmlSource = await readFile(new URL('../chat/index.html', import.meta.url), 'utf8');

test('an explicitly selected WebGL backend is not described as a WebGPU failure', () => {
  const start = appSource.indexOf('function webglBackendCause(');
  const end = appSource.indexOf('\n}', start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({});
  vm.runInContext(`${appSource.slice(start, end + 2)}\nthis.explain = webglBackendCause;`, context);
  assert.match(context.explain('webgl', true), /explicitly selected/);
  assert.doesNotMatch(context.explain('webgl', true), /did not start|no WebGPU/);
  assert.match(context.explain(null, true), /present but did not start/);
  assert.match(context.explain(null, false), /no WebGPU/);
});

test('WebGL warning does not present settled decode ratios as cold latency guarantees', () => {
  assert.doesNotMatch(appSource, /WEBGL_SLOWDOWN/);
  assert.match(appSource, /not cold-load or first-token guarantees/);
  assert.match(appSource, /There is no universal speed ratio/);
});

test('the WebGPU speed floor does not falsely flag a normal WebGL reply', () => {
  const start = appSource.indexOf('const SLOW_LIMITS = ');
  const end = appSource.indexOf('\nfunction hideSlowNote', start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({ ENV: { backend: 'webgl' }, paintSlowNote() {} });
  vm.runInContext(`${appSource.slice(start, end)}\nloadedGB = 0.35;`
    + '\nthis.check = checkSlow; this.state = () => slowState;', context);
  context.check({ tok_s: 18.7 });
  assert.equal(context.state(), null);
  context.ENV.backend = 'webgpu';
  context.check({ tok_s: 18.7 });
  assert.equal(context.state().lim.floor, 40);
});

test('decode footer does not mislabel GPU sampling and readback as host time', () => {
  assert.match(appSource, /step ' \+ Number\(g\)/);
  assert.match(appSource, /pick\/readback /);
  assert.doesNotMatch(appSource, /\+ host ' \+ Number\(k\)/);
});

test('chat presentation defaults do not override an explicit plain-text request', () => {
  const start = appSource.indexOf('const UI_SYSTEM = [');
  const end = appSource.indexOf('].join(', start);
  assert.ok(start >= 0 && end > start);
  const prompt = appSource.slice(start, end);
  assert.match(prompt, /Follow the user’s requested output format exactly/);
  assert.match(prompt, /plain text is requested, do not add Markdown, code fences/);
  assert.doesNotMatch(prompt, /Format every reply as Markdown/);
});

test('live Markdown updates adapt to measured token cadence without a model-name branch', () => {
  const start = appSource.indexOf('const STREAM_RENDER_MS = ');
  const end = appSource.indexOf('\nconst liveRender = ', start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({ res: { steps: [] } });
  vm.runInContext(`${appSource.slice(start, end)}\nthis.interval = streamRenderInterval;`, context);
  assert.equal(context.interval(), 60);
  context.res.steps = Array(8).fill(7);
  assert.equal(context.interval(), 200);
  context.res.steps = Array(8).fill(28);
  assert.equal(context.interval(), 60);
});

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
    if (url.startsWith('https://modelscope.')) throw new Error('not published here');
    return rangeResponse(1);
  });
  const source = await selectSource({
    repo: 'mccoysc/xDecision', probe: 'models/gguf/xDecision-Q8_0.gguf', url: '',
  });
  assert.equal(source.id, 'huggingface');
  assert.equal(calls.length, 3, 'one existence probe per configured source');
  assert.deepEqual(calls.map(call => call.range), ['bytes=0-0', 'bytes=0-0', 'bytes=0-0']);
});

test('multiple reachable model sources are throughput-raced after probing', async () => {
  const calls = [];
  const selectSource = modelSourceHarness(async (url, options) => {
    calls.push({ url, range: options.headers.Range });
    return rangeResponse(options.headers.Range === 'bytes=0-0' ? 1 : 300000);
  });
  await selectSource({ repo: 'org/model', probe: 'model.gguf', url: '' });
  assert.equal(calls.length, 6, 'three probes followed by three throughput samples');
  assert.deepEqual(calls.map(call => call.range), [
    'bytes=0-0', 'bytes=0-0', 'bytes=0-0',
    'bytes=0-1048575', 'bytes=0-1048575', 'bytes=0-1048575',
  ]);
});

test('browser source selection excludes endpoints without CORS support', () => {
  assert.doesNotMatch(appSource, /endpoint:\s*['"]https:\/\/hf-mirror\.com/);
});

test('local multi-file model UI states that ZIP archives must be extracted first', () => {
  assert.match(appSource, /Load a model folder from this device \(extract a ZIP first\)/);
});

test('local GGUF selection passes a direct disk-backed File to the SDK', () => {
  assert.match(htmlSource, /id="localModelInput"[^>]*type="file"[^>]*accept="[^"]*\.gguf/);
  assert.match(appSource, /input\.files && input\.files\[0\]/);
  assert.match(appSource, /cache\.import\(handle, name\)/);
  assert.match(appSource, /localModelIds\.add\(id\)/);
  const localBranch = appSource.slice(appSource.indexOf('if (localModelIds.has(id))'),
                                      appSource.indexOf('} else {', appSource.indexOf('if (localModelIds.has(id))')));
  assert.match(localBranch, /webtorch\.use_default_io\(\)/);
  assert.match(localBranch, /runner\.load\(id,/);
  assert.doesNotMatch(localBranch, /installApplicationReader|applicationModelSource|remoteModelSpec/);
});

test('local GGUF picker is a visible native file input', () => {
  assert.match(htmlSource, /id="localModelInput" type="file"[^>]+\/>/);
  assert.match(appSource, /localPick\('local-file', \(\) => \{\}, event\.target\.files\[0\]\)/);
});

test('local model status never claims a browser cache copy', () => {
  assert.match(appSource, /loadedFromDisk = localModelIds\.has\(m\.id\)/);
  assert.match(appSource, /no cache copy was made/);
  assert.match(appSource, /The original file remains on this device/);
});

test('a failed local load closes the partial runtime and clears its invalid File handle', async () => {
  const start = appSource.indexOf('async function restartRuntimeAfterFailedLoad(');
  const end = appSource.indexOf("\n$('#loadBtn').onclick", start);
  assert.ok(start >= 0 && end > start);
  const events = [];
  const modelId = { value: 'local-model' };
  const oldRuntime = { close: () => events.push('close') };
  const newRuntime = {};
  const localIds = new Set(['local-model']);
  const context = vm.createContext({ events, modelId, oldRuntime, newRuntime, localIds });
  vm.runInContext(`
    let wt = oldRuntime, sdk = Promise.resolve(oldRuntime), envReady = true;
    let visionDecision = null;
    const localModelIds = localIds;
    const $ = selector => selector === '#modelId' ? modelId : null;
    function startSdk() { events.push('start'); wt = newRuntime; envReady = true;
      return Promise.resolve(newRuntime); }
    function afterRelease() { events.push('afterRelease'); }
    ${appSource.slice(start, end)}
    this.restart = restartRuntimeAfterFailedLoad;
    this.state = () => ({ wt, sdk, envReady, ids: [...localModelIds] });
  `, context);
  const result = await context.restart(true);
  assert.equal(result.restartError, null);
  assert.deepEqual(events, ['close', 'start', 'afterRelease']);
  assert.equal(modelId.value, '');
  assert.equal(context.state().ids.length, 0);
  assert.equal(context.state().wt, newRuntime);
  assert.equal(await context.state().sdk, newRuntime);
  const loadCatch = appSource.slice(appSource.indexOf('  catch (e) {', end));
  assert.match(loadCatch, /await restartRuntimeAfterFailedLoad\(wasLocal\)/);
});

test('ordered decisions headline the actual maximum-probability level', () => {
  const scoreBranch = appSource.indexOf('if (a.score !== undefined)');
  const choiceBranch = appSource.indexOf('else if (a.choice !== undefined)', scoreBranch);
  assert.ok(scoreBranch >= 0 && choiceBranch > scoreBranch);
  const body = appSource.slice(scoreBranch, choiceBranch);
  assert.match(body, /verdict\.textContent = peak \|\| balance/);
  assert.doesNotMatch(body, /verdict\.textContent = legend\[String\(near\)\]/);
});

test('load-stage UI never renders an impossible completed/total fraction', () => {
  assert.match(appSource, /m\.total && m\.done <= m\.total/);
  assert.ok(appSource.includes("m.done + ' completed)'"));
});

test('each load attempt clears stage timings even without a reading stage', () => {
  const handler = appSource.slice(appSource.indexOf("$('#loadBtn').onclick = async () => {"));
  const reset = handler.indexOf('stageLog = [];');
  const invoke = handler.indexOf('await runner.load(id,');
  assert.ok(reset >= 0 && reset < invoke, 'reset precedes local model load');
  const stageHandler = appSource.slice(appSource.indexOf('function onLoadStage(m) {'),
                                       appSource.indexOf('\nfunction afterLoad(m) {'));
  assert.doesNotMatch(stageHandler, /stageLog\s*=\s*\[\]/,
    'individual stage events must not erase earlier stages in one attempt');
});

test('finished chat turns clear only a stale stopping status', () => {
  const start = appSource.indexOf('function clearStopNote() {');
  const end = appSource.indexOf('\nfunction promptFor(', start);
  assert.ok(start >= 0 && end > start);
  const clearStopNote = appSource.slice(start, end);
  for (const [before, after] of [
    ['Stopping…', ''],
    ['A different status', 'A different status'],
  ]) {
    const hint = { textContent: before };
    vm.runInNewContext(`${clearStopNote}\nclearStopNote();`, {
      $: selector => {
        assert.equal(selector, '#hintbar');
        return hint;
      },
    });
    assert.equal(hint.textContent, after);
  }
  const runTurn = appSource.slice(appSource.indexOf('async function runTurn('),
                                  appSource.indexOf("$('#send').addEventListener('click'"));
  assert.match(runTurn, /streaming = null;\s*syncButtons\(\);[^\n]*\n\s*clearStopNote\(\);/);
});
