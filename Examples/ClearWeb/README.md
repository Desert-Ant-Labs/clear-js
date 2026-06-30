# ClearWeb — browser demo

A no-bundler browser demo for [`@desert-ant-labs/clear`](../../): pick an audio
file, watch it decode → enhance → master, and A/B the before/after with a WAV
download.

## Run

Build the package once at the repo root, then serve this folder:

```bash
# from the repo root
npm install
npm run build

# then (serve.py serves the repo root so ../../dist resolves)
python3 Examples/ClearWeb/serve.py
# open http://localhost:8765/Examples/ClearWeb/
```

`serve.py` sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless` so the page is
`crossOriginIsolated` — threaded WASM and WebGPU both need that. Opening the
files over `file://` won't work (browsers block WASM and `fetch()` there).

## How it imports the library

There's no bundler. The import map in [`index.html`](./index.html) points the
package's bare specifiers at the local build and a CDN:

```html
<script type="importmap">
{
  "imports": {
    "@desert-ant-labs/clear": "../../dist/index.browser.js",
    "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.min.mjs"
  }
}
</script>
```

In a real app you'd `npm install @desert-ant-labs/clear onnxruntime-web` and let
your bundler resolve them.

The model files load at runtime from
[`huggingface.co/desert-ant-labs/clear`](https://huggingface.co/desert-ant-labs/clear)
and cache in the browser's Cache Storage.
