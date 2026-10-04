import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const webgl = await readFile(new URL('../src/webgl/webglContext.ts', import.meta.url), 'utf8');
const webgpu = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
const gpuBuffer = await readFile(new URL('../src/webgpu/webgpuTensorBuffer.ts', import.meta.url), 'utf8');
const glCompute = await readFile(new URL('../src/webgl/webglComputeContext.ts', import.meta.url), 'utf8');
const glWorker = await readFile(new URL('../src/webgl/webglWorker.ts', import.meta.url), 'utf8');
const main = await readFile(new URL('../src/main.ts', import.meta.url), 'utf8');
const glPlatform = await readFile(new URL('../webgl/wgpy_backends/webgl/platform.py', import.meta.url), 'utf8');
const chat = await readFile(new URL('../chat/app.js', import.meta.url), 'utf8');

test('a lost WebGL context is rejected at upload, readback and dispatch boundaries', () => {
  assert.match(webgl, /addEventListener\('webglcontextlost'/);
  assert.match(webgl, /this\.contextLost \|\| this\.gl\.isContextLost\(\)/);
  assert.match(webgl, /WebGL context lost; release and reload the model/);
  for (const name of ['setDataRaw', 'readPixels2D', 'readPixels2DArray', 'createTexture']) {
    const start = webgl.indexOf((name.startsWith('readPixels') ? 'private ' : '  ')
                                + `${name}(`);
    assert.ok(start >= 0, `${name} exists`);
    assert.match(webgl.slice(start, start + 750), /assertAlive\(\)/, `${name} checks loss`);
  }
});

test('WebGPU rejects lost devices at the equivalent model-facing boundaries', () => {
  assert.match(webgpu, /this\.device\.lost\?\.then/);
  assert.match(webgpu, /WebGPU device lost:/);
  assert.match(webgpu, /runKernel\([\s\S]*?this\.assertAlive\(\)/);
  assert.match(gpuBuffer, /async getDataRaw\(\)[\s\S]*?ctx\.assertAlive\?\.\(\)/);
});

test('a failed WebGL readback wakes the worker and Python refuses stale data', () => {
  assert.match(glCompute, /\.catch\(\(reason\) => \{[\s\S]*?this\.mnotify!\[0\] = -1;[\s\S]*?Atomics\.notify\(this\.mnotify!, 0\)/);
  assert.match(glPlatform, /if status == -1:[\s\S]*?raise RuntimeError\("WebGL readback failed/);
});

test('WebGL accounts live texture bytes in JS shared memory without a Python GPU query', () => {
  assert.match(main, /glWorker\?\.postMessage\(\{ __webtorch: 'channels', stat: e\.data\.channels\?\.stat \|\| null \}\)/);
  assert.match(glWorker, /context\.setResourceStats\(message\.stat \|\| null\)/);
  assert.match(glCompute, /setResourceStats\(memory: SharedArrayBuffer \| null\)/);
  assert.match(glCompute, /this\.heldTextureBytes \+= bytes/);
  assert.match(glCompute, /this\.heldTextureBytes -= this\.textureBytes\.get\(id\) \|\| 0/);
  assert.match(glCompute, /this\.resourceStats\[0\] = this\.heldTextureBytes/);
});

test('chat discards untrusted partial output and releases a lost GPU runtime', () => {
  const start = chat.indexOf('function lostGpuRuntime(');
  const end = chat.indexOf('\n}', start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({});
  vm.runInContext(`${chat.slice(start, end + 2)}\nthis.lostGpuRuntime = lostGpuRuntime;`, context);
  assert.equal(context.lostGpuRuntime(new Error('WebGL context lost')), true);
  assert.equal(context.lostGpuRuntime(new Error('WebGPU device lost: reset')), true);
  assert.equal(context.lostGpuRuntime(new Error('ordinary failure')), false);
  assert.match(chat, /if \(!withTools \|\| lostGpuRuntime\(err\)\) throw err/);
  assert.match(chat, /reply\.content = 'Error: the GPU context was lost/);
  assert.match(chat, /await \$\('#releaseBtn'\)\.onclick\(\)/);
});
