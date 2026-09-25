/* Browser decision worker derived from laya-vision's Apache-2.0 web demo.
 * Upstream: https://github.com/r33drichards/laya-vision
 */
// Inference worker: loads the exported ONNX graphs with onnxruntime-web (WebGPU, else WASM) and answers
// {type: "run"} messages with the same answer schema as VLMAgent.predict. All model work happens here so the
// page stays responsive. Model bytes are requested from the page and therefore go through the same installed
// SDK io_read callback as every other model. This worker owns no source selection, transport or persistent cache.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
import * as laya from "./decision-vision-runtime.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let state = null; // {cfg, tok, sessions, backend, variant, featureCache, maxFeatureCache}
let nextReadId = 1;
const pendingReads = new Map();

function cachedFeature(key) {
  if (!key || !state.featureCache.has(key)) return null;
  const value = state.featureCache.get(key);
  state.featureCache.delete(key);
  state.featureCache.set(key, value);
  return value;
}

function rememberFeature(key, value) {
  if (!key) return;
  state.featureCache.delete(key);
  state.featureCache.set(key, value);
  while (state.featureCache.size > state.maxFeatureCache) {
    state.featureCache.delete(state.featureCache.keys().next().value);
  }
}

const send = (type, data = {}) => self.postMessage({ type, ...data });

function requestBytes(name, offset = 0, length = null) {
  const id = nextReadId++;
  return new Promise((resolve, reject) => {
    pendingReads.set(id, { resolve, reject });
    send("read", { id, name, offset, length });
  });
}

/** Read a known-size model file in the cache's native 16 MiB units. Each request goes to
 * webtorch.io_read on the Python worker; interrupted downloads therefore retain every chunk
 * that hf_read/modelscope_read/hub_read already persisted. */
async function readFile(model, path, label, size, sha256) {
  const name = `${model}/${path}${sha256 ? `?sha256=${encodeURIComponent(sha256)}` : ""}`;
  if (!size) return new Uint8Array(await requestBytes(name, 0, null));
  const chunk = 16 * 1024 * 1024;
  const parallel = 4;
  const out = new Uint8Array(size);
  let loaded = 0;
  for (let base = 0; base < size; base += chunk * parallel) {
    const requests = [];
    for (let offset = base; offset < Math.min(size, base + chunk * parallel); offset += chunk) {
      const length = Math.min(chunk, size - offset);
      requests.push(requestBytes(name, offset, length).then((value) => {
        const bytes = new Uint8Array(value);
        if (bytes.length !== length) {
          throw new Error(`${path}: short read at ${offset} (${bytes.length}/${length})`);
        }
        out.set(bytes, offset);
        loaded += bytes.length;
        send("progress", { phase: "load", label, loaded, total: size });
      }));
    }
    await Promise.all(requests);
  }
  return out;
}

async function readJSON(model, path) {
  const bytes = await readFile(model, path, path, 0, null);
  return JSON.parse(new TextDecoder().decode(bytes));
}

/** ``{ep, f16, adapter}``: WebGPU when asked for (or "auto") and an adapter exists, else WASM. ``f16`` says whether
 * the adapter can run the fp16 graphs (the ``shader-f16`` feature); software adapters usually cannot. */
async function pickBackend(requested) {
  if (requested === "wasm") return { ep: "wasm", f16: true, adapter: "" };
  let adapter = null;
  try { adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch { adapter = null; }
  if (!adapter) {
    if (requested === "webgpu") throw new Error("WebGPU was requested but no adapter is available in this browser.");
    return { ep: "wasm", f16: true, adapter: "" };
  }
  const info = adapter.info || {};
  const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown adapter";
  return { ep: "webgpu", f16: adapter.features.has("shader-f16"), adapter: name + (adapter.isFallbackAdapter ? " (fallback)" : "") };
}

async function load({ model, variant, backend, imageCacheEntries = 32 }) {
  const t0 = performance.now();
  const cfg = await readJSON(model, "laya_web.json");
  if (cfg.format_version !== 1 || cfg.readout !== "terminator") throw new Error("unsupported laya_web.json (format or readout)");
  const [tj, tc] = await Promise.all(cfg.tokenizer.map((p) => readJSON(model, p)));
  if (!tj?.model) throw new Error(`invalid tokenizer.json (keys: ${Object.keys(tj || {}).join(", ")})`);
  const tok = new Tokenizer(tj, tc);
  const { ep, f16, adapter } = await pickBackend(backend);
  if (variant === "auto") variant = ep === "webgpu" && f16 ? "fp16" : "q8";
  if (variant === "fp16" && !f16) throw new Error("this WebGPU adapter has no shader-f16 support; pick fp32, q8 or q4, or WASM");
  const suffix = variant === "fp32" ? "" : "_" + variant;
  const sessions = {};
  const timings = {};
  for (const name of ["vision", "text", "head"]) {
    const file = `${name}${suffix}.onnx`;
    if (!cfg.files[file]) throw new Error(`${file} is not in laya_web.json; export it with --quantize ${variant}`);
    const t = performance.now();
    const bytes = await readFile(model, file, file, Number(cfg.files[file].bytes || 0), cfg.files[file].sha256);
    timings[`download ${file}`] = performance.now() - t;
    const t2 = performance.now();
    // the tiny head graph runs on WASM: a WebGPU dispatch per op costs more than the arithmetic
    const eps = name === "head" ? ["wasm"] : [ep];
    sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
    timings[`session ${file}`] = performance.now() - t2;
  }
  state = { cfg, tok, sessions, backend: ep, variant, featureCache: new Map(), maxFeatureCache: Math.max(1, Math.min(256, Number(imageCacheEntries) || 32)) };
  send("loaded", { backend: ep, adapter, variant, files: cfg.files, suffix, seconds: (performance.now() - t0) / 1000, timings,
                   source: cfg.source, temperature: cfg.temperature,
                   temperatureByOptions: cfg.temperature_by_options || {},
                   maxLen: cfg.max_len, headMaxLen: cfg.head_max_len,
                   qtypes: Object.keys(cfg.qtypes || {}), isolated: self.crossOriginIsolated });
}

const i64 = (arr) => new ort.Tensor("int64", BigInt64Array.from(arr.map((v) => BigInt(v))), [arr.length]);

async function run({ images, stateObj, questions, nPermutations }) {
  if (!state) throw new Error("load a model first");
  const { cfg, tok, sessions } = state;
  const encode = (s) => tok.encode(s, { add_special_tokens: false }).ids;
  const t0 = performance.now();
  const d = cfg.hidden_size, L = cfg.image.seq_len, S = cfg.image.size;

  // images -> features [n, L, d]
  let feats = new ort.Tensor("float32", new Float32Array(0), [0, L, d]);
  const tImg = performance.now();
  let visionCacheHits = 0, visionCacheMisses = 0;
  if (images.length) {
    const chunks = [];
    for (const im of images) {
      let feature = cachedFeature(im.cacheKey);
      if (feature) {
        visionCacheHits++;
      } else {
        const px = laya.pixelValues(im.rgba, im.height, im.width, cfg);
        const out = await sessions.vision.run({ pixel_values: new ort.Tensor("float32", px.data, [1, 3, S, S]) });
        feature = out.image_features.data.slice();
        rememberFeature(im.cacheKey, feature);
        visionCacheMisses++;
      }
      chunks.push(feature);
    }
    const all = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
    let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    feats = new ort.Tensor("float32", all, [images.length, L, d]);
  }
  const visionMs = performance.now() - tImg;

  const prefix = laya.prefixIds(encode, cfg, images.length);
  const text = laya.stateText(stateObj);
  const answers = {};
  const details = {};
  let nTokens = 0;
  const qids = Object.keys(questions);
  for (const [qi, qid] of qids.entries()) {
    const q = laya.toInternal(questions[qid]);
    const k = laya.renderOptions(q).length;
    if (k < 2) throw new Error(`question ${qid} needs at least two options`);
    const rows = [];
    for (const order of laya.permutations(k, nPermutations)) {
      const it = laya.buildInputs(encode, cfg, prefix, text, q, order);
      if (it.markers.length !== k) throw new Error(`question ${qid}: options exceed head_max_len`);
      nTokens += it.ids.length;
      const ids = new ort.Tensor("int64", BigInt64Array.from(it.ids.map(BigInt)), [1, it.ids.length]);
      const h = await sessions.text.run({ input_ids: ids, image_features: feats, option_span: i64(it.option_span) });
      const out = await sessions.head.run({ hidden: h.last_hidden_state, marker_pos: i64(it.markers),
                                            qtype: i64([laya.QTYPES[q.t]]) });
      const act = laya.softmax(Array.from(out.act_logits.data));
      rows.push({ order, logits: Array.from(out.logits.data), actProb: act[0] });
      details[qid] = { tokens: it.ids.length, state_tokens: it.state_tokens, state_truncated: it.state_truncated };
      send("progress", { phase: "run", label: `question ${qi + 1}/${qids.length}`, loaded: qi + 1, total: qids.length });
    }
    answers[qid] = laya.answer(cfg, q, rows);
  }
  send("result", {
    result: { model: cfg.source || "laya-vlm", answers, usage: { input_tokens: nTokens, output_tokens: 0, images: images.length, vision_cache_hits: visionCacheHits, vision_cache_misses: visionCacheMisses } },
    details,
    timing: { vision_ms: visionMs, total_ms: performance.now() - t0, vision_cache_hits: visionCacheHits, vision_cache_misses: visionCacheMisses },
  });
}

self.onmessage = async ({ data }) => {
  if (data.type === "read-result" || data.type === "read-error") {
    const pending = pendingReads.get(data.id);
    if (!pending) return;
    pendingReads.delete(data.id);
    if (data.type === "read-result") pending.resolve(data.bytes);
    else pending.reject(new Error(data.message || "model read failed"));
    return;
  }
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "run") await run(data);
  } catch (err) {
    send("error", { message: String(err?.message || err) });
  }
};
