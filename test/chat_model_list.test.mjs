import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const list = JSON.parse(await readFile(new URL('../chat/models.json', import.meta.url), 'utf8'));

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
