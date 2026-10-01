import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const stamp = await readFile(new URL('../scripts/stamp.sh', import.meta.url), 'utf8');
const host = await readFile(new URL('../webtorch/js/webtorch-host.js', import.meta.url), 'utf8');
const hook = await readFile(new URL('../.githooks/check-stamp.py', import.meta.url), 'utf8');

test('SDK browser version covers Python modules and worker bootstraps', () => {
  assert.match(stamp, /find webtorch .* -name '\*\.py'/);
  assert.match(stamp, /webtorch\/js\/webtorch-host\.js/);
  assert.match(stamp, /webtorch\/js\/webtorch-worker\.js/);
  assert.match(stamp, /webtorch\/modules\.json/);
  assert.match(stamp, /stamp_html_version chat\/index\.html "\.\.\/webtorch\/js\/webtorch-main\.js"/);
  assert.match(hook, /def sdk_hash\(length\):/);
  assert.match(hook, /url == '\.\.\/webtorch\/js\/webtorch-main\.js'/);
});

test('versioned host also versions its imported worker bootstrap', () => {
  assert.match(host, /const VQ = VERSION \?/);
  assert.match(host, /webtorch\/js\/webtorch-worker\.js' \+ VQ/);
});
