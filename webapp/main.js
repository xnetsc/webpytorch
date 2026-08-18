function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

let audioCtx = null;
function playAudio(samples, sr) {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const buf = audioCtx.createBuffer(1, samples.length, sr);
    buf.copyToChannel(samples, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf; src.connect(audioCtx.destination); src.start();
    log(`\n▶ playing ${(samples.length / sr).toFixed(2)}s of audio @ ${sr} Hz`);
  } catch (e) { log(`audio playback failed: ${e.message}`); }
}

async function run() {
  // create AudioContext under the Run click (user gesture) so later playback is allowed
  try { if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) {}
  const worker = new Worker('worker.js');
  const backendOrder = [];
  for (const b of ['webgpu', 'webgl']) {
    if (document.getElementById(b).checked) backendOrder.push(b);
  }

  log('init wgpy main interface');
  let backend = 'cpu';
  try {
    const r = await wgpy.initMain(worker, { backendOrder });
    backend = r.backend;
  } catch (e) { log(`initMain failed: ${e.message}`); }
  log(`main backend: ${backend}`);

  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'app') return;
    if (e.data.method === 'log') log(e.data.message);
    if (e.data.method === 'result') {
      window.__RESULT = JSON.parse(e.data.json);
      log('\n=== RESULT ===\n' + JSON.stringify(window.__RESULT, null, 2));
    }
    if (e.data.method === 'audio') {
      window.__AUDIO = { samples: e.data.samples, sr: e.data.sr };
      playAudio(e.data.samples, e.data.sr);
    }
  });

  const sel = document.getElementById('script');
  const script = sel ? sel.value : 'run.py';
  worker.postMessage({ namespace: 'app', method: 'start', config: { script } });
}

window.addEventListener('load', () => {
  document.getElementById('run').onclick = () => {
    document.getElementById('run').disabled = true;
    document.getElementById('log').innerHTML = '';
    window.__RESULT = null;
    run().catch((e) => log(`Main error: ${e.message}`));
  };
});
