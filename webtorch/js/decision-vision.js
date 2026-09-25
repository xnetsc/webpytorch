/* Browser-side typed decisions over image + text.
 *
 * This is a small adapter around the exported three-graph decision format used by
 * laya-vision. It deliberately does not choose or fetch from a model host: the application
 * passes the same read callback it installed for the rest of the SDK.
 */
(function (root) {
  const wt = root.webtorch || (root.webtorch = {});
  const script = typeof document !== 'undefined' && document.currentScript;
  const defaultWorker = script && new URL('decision-vision-worker.js', script.src).href;

  function imageSpec(value) {
    if (typeof value === 'string') return { type: 'image', data: value };
    if (!value || typeof value !== 'object' || value.type !== 'image') {
      throw new TypeError('decision image must be a data URL or {type:"image", data, media_type}');
    }
    return value;
  }

  /** Keep text/json calls unchanged while giving image states one explicit wire shape. */
  wt.normalizeDecisionState = function (state) {
    if (state && typeof state === 'object' && !Array.isArray(state)) {
      if (state.type === 'text') return { state: String(state.text ?? state.data ?? ''), images: [] };
      if (state.type === 'json') return { state: state.value ?? state.data ?? {}, images: [] };
      if (state.type === 'image') return { state: state.text ?? '', images: [imageSpec(state)] };
      if (state.type === 'multimodal') {
        const media = state.images ?? (state.image == null ? [] : [state.image]);
        return { state: state.text ?? state.value ?? {}, images: [].concat(media).map(imageSpec) };
      }
      if (Object.prototype.hasOwnProperty.call(state, 'image') ||
          Object.prototype.hasOwnProperty.call(state, 'images')) {
        const media = state.images ?? (state.image == null ? [] : [state.image]);
        const text = Object.fromEntries(Object.entries(state)
          .filter(([key]) => key !== 'image' && key !== 'images'));
        return { state: text, images: [].concat(media).map(imageSpec) };
      }
    }
    return { state, images: [] };
  };

  function bytesFromSpec(spec) {
    let data = String(spec.data || '');
    let mediaType = String(spec.media_type || spec.mime_type || 'image/png');
    const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(data);
    if (match) { mediaType = match[1]; data = match[2]; }
    if (!/^[A-Za-z0-9+/=\s]+$/.test(data)) {
      throw new TypeError('decision image data must be base64 or a base64 data URL');
    }
    const binary = atob(data.replace(/\s+/g, ''));
    if (!binary.length) throw new TypeError('decision image is empty');
    if (binary.length > 16 * 1024 * 1024) throw new RangeError('decision image exceeds 16 MB');
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return { bytes, mediaType };
  }

  async function decodeImage(spec) {
    const { bytes, mediaType } = bytesFromSpec(spec);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const cacheKey = [...new Uint8Array(digest)].map((v) => v.toString(16).padStart(2, '0')).join('');
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mediaType }));
    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(bitmap.width, bitmap.height)
      : Object.assign(document.createElement('canvas'), { width: bitmap.width, height: bitmap.height });
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    if (bitmap.close) bitmap.close();
    return { rgba, width: canvas.width, height: canvas.height, cacheKey };
  }

  function questionTypes(names) {
    const all = {
      choice: { label: 'Pick one', min: 2, needs: 'options', options: 'named' },
      score: { label: 'Rate on a scale', min: 2, needs: 'levels', options: 'ordered' },
      noul: { label: 'How likely is this true?', min: 2, needs: null, options: 'fixed' },
    };
    return Object.fromEntries((names || ['choice', 'score', 'noul'])
      .filter((name) => all[name]).map((name) => [name, all[name]]));
  }

  wt.loadVisionDecision = async function (options) {
    options = options || {};
    if (typeof options.read !== 'function') throw new TypeError('loadVisionDecision needs read(name, offset, length)');
    if (!options.model) throw new TypeError('loadVisionDecision needs model');
    const workerURL = options.workerURL || defaultWorker;
    if (!workerURL) throw new TypeError('loadVisionDecision needs workerURL outside a document');
    const worker = new Worker(workerURL, { type: 'module' });
    let loadResolve, loadReject, runResolve, runReject;
    let released = false;
    let loaded = null;
    let runTail = Promise.resolve();
    const loadPromise = new Promise((resolve, reject) => { loadResolve = resolve; loadReject = reject; });

    worker.addEventListener('message', async ({ data }) => {
      if (data.type === 'read') {
        try {
          const value = await options.read(data.name, data.offset, data.length);
          const view = value instanceof Uint8Array ? value : new Uint8Array(value);
          // Never transfer a buffer owned by the callback: transfer detaches it, and a host
          // callback is allowed to retain or reuse what it returned.
          const bytes = view.slice();
          worker.postMessage({ type: 'read-result', id: data.id, bytes }, [bytes.buffer]);
        } catch (error) {
          worker.postMessage({ type: 'read-error', id: data.id,
            message: String(error?.message || error) });
        }
      } else if (data.type === 'progress') {
        if (options.onProgress) options.onProgress(data);
      } else if (data.type === 'loaded') {
        loaded = data; loadResolve(data);
      } else if (data.type === 'result') {
        const resolve = runResolve; runResolve = runReject = null;
        if (resolve) resolve({ ...data.result, timing: data.timing, details: data.details });
      } else if (data.type === 'error') {
        const error = new Error(data.message || 'vision decision worker failed');
        if (!loaded) loadReject(error);
        else if (runReject) { const reject = runReject; runResolve = runReject = null; reject(error); }
        if (options.onError) options.onError(error);
      }
    });
    worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'vision decision worker stopped');
      if (!loaded) loadReject(error); else if (runReject) runReject(error);
      if (options.onError) options.onError(error);
    });
    worker.postMessage({ type: 'load', model: String(options.model).replace(/\/$/, ''),
      variant: options.variant || 'auto', backend: options.backend || 'auto',
      imageCacheEntries: options.imageCacheEntries || 32 });
    const abortLoad = () => {
      released = true;
      worker.terminate();
      loadReject(new Error('load cancelled'));
    };
    if (options.signal) {
      if (options.signal.aborted) abortLoad();
      else options.signal.addEventListener('abort', abortLoad, { once: true });
    }
    try {
      await loadPromise;
    } catch (error) {
      worker.terminate();
      throw error;
    } finally {
      if (options.signal) options.signal.removeEventListener('abort', abortLoad);
    }

    const surface = {
      kind: 'decision',
      takes: {
        state: { kinds: ['text', 'json', 'image', 'multimodal'], image: {
          wire: '{type:"image", media_type:"image/png", data:"<base64>"}', max_bytes: 16 * 1024 * 1024,
        } },
        questions: { types: questionTypes(loaded.qtypes), max: null },
      },
      returns: { per_question: ['probabilities', 'confidence', 'act_probability'] },
      calibration: { method: 'temperature-scaling', status: 'checkpoint',
        domain_calibrated: false, groups: [], adjustments: [],
        note: 'Checkpoint calibration is not evidence of accuracy on this application data.' },
      limits: { sequence_tokens: loaded.maxLen, question_tokens: loaded.headMaxLen,
        image_bytes: 16 * 1024 * 1024 },
    };

    async function decide(state, questions) {
      if (released) throw new Error('vision decision model was released');
      const normalized = wt.normalizeDecisionState(state);
      const task = async () => {
        const images = await Promise.all(normalized.images.map(decodeImage));
        return await new Promise((resolve, reject) => {
          runResolve = resolve; runReject = reject;
          worker.postMessage({ type: 'run', images, stateObj: normalized.state, questions,
            nPermutations: options.nPermutations || 1 });
        });
      };
      const result = runTail.then(task, task);
      runTail = result.then(() => {}, () => {});
      return result;
    }

    return {
      kind: 'decision', model: loaded.source, backend: loaded.backend, variant: loaded.variant,
      decide, surface: () => surface, status: () => ({ ready: !released, ...loaded }),
      release: () => { if (!released) worker.terminate(); released = true; },
    };
  };
})(typeof self !== 'undefined' ? self : globalThis);
