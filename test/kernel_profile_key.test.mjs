import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../webtorch/js/webtorch-host.js', import.meta.url), 'utf8');
const begin = source.indexOf('async function kpKey()');
const end = source.indexOf('async function kpGet(', begin);
assert.ok(begin >= 0 && end > begin);

function harness() {
  let gpuRequests = 0;
  const gl = {
    VENDOR: 1, RENDERER: 2, VERSION: 3,
    getExtension: () => ({ UNMASKED_VENDOR_WEBGL: 4, UNMASKED_RENDERER_WEBGL: 5 }),
    getParameter: value => ({ 3: 'WebGL 2.0', 4: 'Vendor GL', 5: 'Renderer GL' })[value],
  };
  const context = vm.createContext({
    navigator: { gpu: { requestAdapter: async () => {
      gpuRequests += 1;
      return { info: { vendor: 'Vendor GPU', architecture: 'arch', device: 'dev' } };
    } } },
    OffscreenCanvas: class { getContext(kind) { return kind === 'webgl2' ? gl : null; } },
  });
  vm.runInContext(`let backendName = 'webgpu'; let profileKey;\n${source.slice(begin, end)}\n` +
    'this.profileKey = kpKey; this.setBackend = value => { backendName = value; profileKey = undefined; };', context);
  return { context, get gpuRequests() { return gpuRequests; } };
}

test('tuning profile keys use the active backend and its own device identity', async () => {
  const h = harness();
  const gpuKey = await h.context.profileKey();
  assert.match(gpuKey, /^webgpu\/Vendor GPU\/arch\/dev\//);
  assert.equal(h.gpuRequests, 1);
  assert.equal(await h.context.profileKey(), gpuKey);
  assert.equal(h.gpuRequests, 1, 'one adapter query per worker lifetime');
  h.context.setBackend('webgl');
  const glKey = await h.context.profileKey();
  assert.equal(glKey, 'webgl/Vendor GL/Renderer GL/WebGL 2.0');
  assert.notEqual(glKey, gpuKey);
  assert.equal(h.gpuRequests, 1, 'WebGL does not read WebGPU adapter identity');
  h.context.setBackend('cpu');
  assert.equal(await h.context.profileKey(), null);
});
