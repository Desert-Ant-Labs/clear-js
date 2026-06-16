# Browser demo + reusable library

Speech enhancement (denoise + dereverb) in the browser via ONNX Runtime
Web. The demo runs the full pipeline (file pick → decode → STFT →
features → inference → ISTFT → before/after playback + WAV download);
the `lib/` directory is the **copy-pasteable library** — drop those files
into your own app and `import { Clear } from './lib/clear.js'`. No
bundler required.

## Run the demo locally

```bash
python3 examples/web/serve.py        # serves on :8765
open http://localhost:8765/examples/web/
```

`serve.py` sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless` so the page becomes
`crossOriginIsolated` — threaded WASM and WebGPU both need that.
Direct `file://` opens won't work — browsers block WASM and `fetch()`
in that context.

## Use the library in your own app

`lib/` is self-contained. Copy these into your project (keep the file
names — they import each other by path):

```
lib/
  clear.js       ← the public API
  stft.js        ← STFT / ISTFT (Vorbis window, n_fft=960, hop=480)
  fft.js         ← Bluestein FFT — handles the non-power-of-2 length
  erb.js         ← 32-band ERB filterbank
  features.js   ← erb dB-norm + complex spec unit-norm
  r128.js        ← EBU R128 / BS.1770 measure + look-ahead limiter
  decode.js      ← any-format audio → mono Float32 @ 48 kHz
  wav.js         ← Float32 → 16-bit WAV blob
```

Then in your code:

```js
import { Clear, encodeWav } from './lib/clear.js';

const clear = await Clear.create({
  variant: 'studio',                          // 'studio' | 'natural'
  onDownloadProgress: (loaded, total) => {
    console.log(`download ${(loaded/total*100).toFixed(0)}%`);
  },
});

const { audio, durationSec, sampleRate } = await clear.enhance(file, {
  onProgress: (stage, value, ctx) => {
    console.log(stage, value, ctx);           // 'decode' | 'inference'
  },
});

// audio: Float32Array @ 48 kHz mono — enhanced speech
const wavBlob = encodeWav(audio, sampleRate);
const url = URL.createObjectURL(wavBlob);

await clear.dispose();
```

### Server-side requirements

Two HTTP headers are required for the fast paths to engage:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

Together these unlock `crossOriginIsolated` (and therefore threaded WASM
+ `SharedArrayBuffer`). `credentialless` (vs the older `require-corp`)
lets the page fetch cross-origin model files from HuggingFace without
HF needing to set `Cross-Origin-Resource-Policy: cross-origin` on each
file. Supported on Chrome 96+, Firefox 119+, Safari 17.4+.

Without these the demo still works — it just falls back to single-thread
WASM (3-4× slower) and can't pick up WebGPU.

### Models on HuggingFace

The Clear Swift Package pins to HF revisions; the JS lib defaults to
following `main` for the same reason — old apps pick up new weights
without redeploying. For determinism, fork-or-mirror the model files
and pass your own `modelURLs` to `Clear.create()`:

```js
const clear = await Clear.create({
  variant: 'studio',
  modelURLs: {
    studio:  'https://your-cdn.com/clear-studio.onnx',
    natural: 'https://your-cdn.com/clear-natural.onnx',
  },
});
```

### Picking a variant

| Variant | Character | Size |
|---|---|---|
| `clear-studio` | Quieter, more processed — studio-like default | 8.9 MB |
| `clear-natural` | Preserves room tone, breath, lip texture | 8.9 MB |

Both share the same architecture, trained on different target recipes.
Numerically they have the same compute cost.

## What the demo adds on top

`main.js` is intentionally separate from the lib — it's pure UI:

- File pick UI, before/after audio playback.
- "Download enhanced WAV" link generated from the returned `Float32Array`.
- HEAD-probes model file sizes at page load so the variant chips display
  the actual byte counts.

If you only want enhancement, copy `lib/` and ignore `main.js`.

## Other considerations

- **Input format** — any container the browser can decode (wav, m4a,
  mp3, flac, ogg, mov, mp4). The lib decodes to 48 kHz mono via an
  `OfflineAudioContext`, which is the browser's native high-quality
  resampler.
- **Mastering** — EBU R128 + true-peak limiting are built in (presets:
  `applePodcasts`, `spotify`, `youtube`, `broadcast`, or `bypass` for raw
  model output). Same chain as the Swift / Android packages.
- **Long files** — the model processes 200-frame chunks (~2 s each)
  internally. The full STFT lives in memory, so a 30-min file uses
  ~70 MB for the spec buffer; consider streaming or chunked processing
  for hour-long inputs.
- **WebGPU vs WASM** — WebGPU is tried first on browsers that have it
  (Chrome / Firefox 113+ / Safari 26+ on Tahoe) and falls back to WASM
  automatically. WebGPU is several× faster but has occasional driver
  quirks; `?wasm` in the URL forces WASM for diagnosis.
- **Memory** — `Clear.dispose()` releases the ORT session. Call it when
  you're done if you're not tearing down the page.

## Parity with the Swift / Android packages

The JS pipeline is a byte-for-byte port of the Swift reference:

- STFT — Vorbis window, n_fft=960, hop=480, `wnorm = 2·hop/n_fft²`
- ERB filterbank — the same hard-coded `widths[]` array the model was
  trained against
- Feature extraction — `erb_norm` and `unit_norm` with the same EMA
  initialisation ramps and `α = 0.99`
- Chunk loop — T=200, `conv_lookahead=2`, same scatter-shift semantics

A node smoke test against the Swift CLI on the bundled sample reports
correlation = 0.9999996 over the full waveform (the only delta is
fp16 Core ML vs fp32 ONNX).
