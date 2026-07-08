# @desert-ant-labs/clear

On-device speech enhancement for **Node and the browser**. Takes noisy mono
audio (laptop mic, untreated room, traffic) and returns a podcast-ready 48 kHz
signal: denoised and dereverbed by a fine-tuned DeepFilterNet 3 model, then
loudness-normalized to platform spec (EBU R128 / BS.1770).

```ts
import { load, decodeToMono, encodeWav } from "@desert-ant-labs/clear";

const clear = await load({ variant: "studio" });           // fetches + caches the model
const pcm = await decodeToMono(file);                       // browser: File/Blob → mono 48 kHz
const { audio } = await clear.enhance(pcm, { mastering: "applePodcasts" });
const wav = new Blob([encodeWav(audio, 48_000)], { type: "audio/wav" });
```

## Features

- Runs everywhere the same import resolves (**Node, browsers, bundlers, and
  edge/worker runtimes**); the right build is selected automatically
- Denoise + dereverb + **loudness mastering** (Apple Podcasts, Spotify, YouTube,
  Broadcast presets, or a custom LUFS/dBTP target) in one call
- Two variants: `studio` (default, cleaner) and `natural` (preserves room tone)
- WebGPU when available, threaded WASM otherwise (browser); CPU (Node)
- The model is fetched from the Hugging Face Hub at a **pinned revision**, then
  cached, to the **filesystem** on Node and to **Cache Storage** in the browser

## Install

```bash
npm install @desert-ant-labs/clear
```

The ONNX Runtime is a **peer dependency**. Install the one for your platform:

```bash
npm install onnxruntime-web     # browser / workers
npm install onnxruntime-node    # Node / server
```

## Importing

Pure ESM. The **same import works everywhere**; the right build is selected by
the package `exports` map:

```ts
import { load } from "@desert-ant-labs/clear";
```

In a no-bundler browser page, map the bare specifiers with an import map (see
[`Examples/ClearWeb`](./Examples/ClearWeb)).

## Usage

### Browser

```ts
import { load, decodeToMono, encodeWav } from "@desert-ant-labs/clear";

const clear = await load({
  variant: "studio",
  onDownloadProgress: (loaded, total) => console.log(`${loaded}/${total}`),
});

const pcm = await decodeToMono(file);                 // any format the browser can decode
const result = await clear.enhance(pcm, {
  mastering: "applePodcasts",                          // or "spotify" | "youtube" | "broadcast" | "bypass"
  onProgress: (stage, frac) => console.log(stage, frac),
});

const url = URL.createObjectURL(
  new Blob([encodeWav(result.audio, result.sampleRate)], { type: "audio/wav" }),
);
```

> Threaded WASM and WebGPU require a **cross-origin-isolated** page
> (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
> require-corp`/`credentialless`). The example `serve.py` sets these.

### Node

Node has no Web Audio, so input is a `Float32Array` of mono PCM at 48 kHz. Use
`decodeWav` for WAV buffers, or decode compressed formats upstream (e.g. ffmpeg).

```ts
import { readFile, writeFile } from "node:fs/promises";
import { load, decodeWav, encodeWav } from "@desert-ant-labs/clear";

const clear = await load({ variant: "studio" });
const { samples } = decodeWav(await readFile("in.wav"));
const { audio, measuredLUFS } = await clear.enhance(samples, { mastering: "applePodcasts" });
await writeFile("out.wav", encodeWav(audio, 48_000));
console.log("integrated loudness:", measuredLUFS, "LUFS");
```

## API

### `load(options?) => Promise<ClearModel>`

Resolves the model (local dir → cache → Hugging Face Hub) and builds an
inference session. Options:

| Option | Default | Notes |
|---|---|---|
| `variant` | `"studio"` | `"studio"` or `"natural"` |
| `useCache` | `true` | Cache to disk (Node) / Cache Storage (browser) |
| `allowRemote` | `true` | Set `false` to require a local/cached copy |
| `localModelPath` | n/a | Node: dir of pre-downloaded `clear-*.onnx` |
| `cacheDir` | `~/.cache/clear` | Node only |
| `token` | `$HF_TOKEN` | Node only, for gated repos |
| `forceWasm` / `numThreads` | n/a | Browser only |
| `onDownloadProgress` | n/a | `(loaded, total) => void` |

### `ClearModel.enhance(pcm, options?) => Promise<EnhanceResult>`

`pcm` is mono `Float32Array` at 48 kHz. Returns `{ audio, durationSec,
sampleRate, measuredLUFS, measuredTruePeakDBFS }`. Options: `mastering`
(preset name, `{ integratedLUFS, truePeakDBTP?, maxLoudnessGainDB? }`, or
`"bypass"`) and `onProgress`.

### Also exported

`decodeToMono` (browser), `decodeWav`/`encodeWav`, `measureLUFS`,
`applyLimiter`, `MASTERING_PRESETS`, `setOrt` (inject a preloaded runtime),
and the `@desert-ant-labs/clear/core` subpath for the platform-agnostic
`ClearModel`.

## Example

[`Examples/ClearWeb`](./Examples/ClearWeb) is a no-bundler browser demo (file
pick → decode → enhance → before/after playback + WAV download):

```bash
npm run build                       # build dist/ at the repo root
python3 Examples/ClearWeb/serve.py  # serves the repo root with COOP/COEP
# open http://localhost:8765/Examples/ClearWeb/
```

## Other platforms

Same model, native on each platform:

- [`clear-swift`](https://github.com/Desert-Ant-Labs/clear-swift): Swift for iOS and macOS (Core ML)
- [`clear-kotlin`](https://github.com/Desert-Ant-Labs/clear-kotlin): Kotlin for Android and the JVM
- Model weights and card: [`desert-ant-labs/clear`](https://huggingface.co/desert-ant-labs/clear)

## License

[Desert Ant Labs Source-Available License](https://license.desertant.ai/1.0). Free for
most apps; a commercial license is required at scale. Full terms are at the link.
Licensing: <licensing@desertant.ai>.
