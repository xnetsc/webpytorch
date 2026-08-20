// Cross-origin-isolated static server (COOP+COEP) for the WgPy spike.
// SharedArrayBuffer (WgPy's sync bridge) requires cross-origin isolation.
// Supports HTTP Range + HEAD so multi-GB model files can be streamed per-tensor
// from disk without reading the whole file into memory.
import { createServer } from 'node:http';
import { stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { extname, normalize, join } from 'node:path';

const ROOT = process.argv[2] || process.cwd();
const PORT = Number(process.argv[3] || 8119);
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.css': 'text/css', '.wasm': 'application/wasm',
  '.whl': 'application/octet-stream', '.py': 'text/plain', '.data': 'application/octet-stream',
  '.map': 'application/json', '.ts': 'text/plain', '.bz2': 'application/octet-stream',
  '.safetensors': 'application/octet-stream', '.gguf': 'application/octet-stream',
  '.txt': 'text/plain',
};

function setCommon(res, ctype) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Accept-Ranges', 'bytes');
  if (ctype) res.setHeader('Content-Type', ctype);
}

// Same-origin proxy for model downloads.
// A cross-origin-isolated page (which SharedArrayBuffer, and therefore the GPU backend,
// requires) can only read a remote file if that host sends CORS headers. Model hosts
// generally do not, so the page fetches `/ms/<path>` from us and we relay it — passing the
// Range header through in both directions so tensors can still be streamed.
const PROXY = { '/ms/': 'https://modelscope.cn/', '/hf/': 'https://huggingface.co/' };

async function proxy(req, res, prefix, origin) {
  const rest = req.url.slice(prefix.length);
  const target = origin + rest.replace(/^\/+/, '');
  const headers = { 'User-Agent': 'webtorch-chat' };
  if (req.headers.range) headers.Range = req.headers.range;
  const r = await fetch(target, { headers, redirect: 'follow' });
  res.statusCode = r.status;
  for (const h of ['content-type', 'content-length', 'content-range', 'accept-ranges']) {
    const v = r.headers.get(h);
    if (v) res.setHeader(h, v);
  }
  // same-origin response: safe under COEP, and readable by the page
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cache-Control', 'no-store');
  if (req.method === 'HEAD' || !r.body) { res.end(); return; }
  const buf = Buffer.from(await r.arrayBuffer());
  res.end(buf);
}

createServer(async (req, res) => {
  try {
    for (const [prefix, origin] of Object.entries(PROXY)) {
      if (req.url.startsWith(prefix)) {
        setCommon(res);
        return await proxy(req, res, prefix, origin);
      }
    }
    let path = decodeURIComponent(req.url.split('?')[0]);
    if (path === '/') path = '/index.html';
    if (path.endsWith('/')) path += 'index.html';
    const filePath = join(ROOT, normalize(path).replace(/^(\.\.[/\\])+/, ''));
    const st = await stat(filePath);
    const size = st.size;
    const ctype = MIME[extname(filePath)] || 'application/octet-stream';

    if (req.method === 'HEAD') {
      setCommon(res, ctype);
      res.setHeader('Content-Length', size);
      res.statusCode = 200;
      res.end();
      return;
    }

    const range = req.headers.range;
    const m = range && /^bytes=(\d*)-(\d*)$/.exec(range);
    if (m) {
      const start = m[1] === '' ? 0 : parseInt(m[1], 10);
      const end = m[2] === '' ? size - 1 : parseInt(m[2], 10);
      if (isNaN(start) || isNaN(end) || start > end || end >= size) {
        setCommon(res, ctype);
        res.statusCode = 416;
        res.setHeader('Content-Range', `bytes */${size}`);
        res.end();
        return;
      }
      setCommon(res, ctype);
      res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
      res.setHeader('Content-Length', end - start + 1);
      createReadStream(filePath, { start, end }).pipe(res);
      return;
    }

    setCommon(res, ctype);
    res.statusCode = 200;
    res.setHeader('Content-Length', size);
    createReadStream(filePath).pipe(res);
  } catch {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.statusCode = 404;
    res.end('not found');
  }
}).listen(PORT, () => console.log(`WgPy spike on http://localhost:${PORT} (cross-origin isolated, Range enabled)`));
