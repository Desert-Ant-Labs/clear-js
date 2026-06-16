# AGENTS.md, clear-web

Read these first, in order:

1. [`../lab/AGENTS.md`](../lab/AGENTS.md), org-wide rules.
2. [`../lab/catalog/clear.md`](../lab/catalog/clear.md), catalog entry.
3. [`../clear-swift/AGENTS.md`](../clear-swift/AGENTS.md), Apple-side SDK shape (mirror its public surface in JS where idiomatic).

> **Write for the next reader, human or agent.**

## What this repo is

Browser demo + reusable JS library for the `clear` speech enhancement model. Runs the full pipeline (file pick → decode → STFT → ONNX inference → ISTFT → before/after playback + WAV download) entirely client-side via ONNX Runtime Web.

Two parts in one repo:

- `index.html`, `main.js`, `clear.css`: the runnable demo (open with `python3 serve.py`).
- `lib/`: the copy-pasteable library. Drop into your own page and `import { Clear } from './lib/clear.js'`. No bundler required.

## Status

Ported from the legacy detail-co browser demo. Model still pulled from `huggingface.co/detail-co/clear` in the existing JS, a pass to redirect the fetch URLs to `huggingface.co/desert-ant-labs/clear` is pending alongside the HF model migration.

## Rules

- **No bundler.** Native ES modules. The lib should drop into an app with a `<script type="module">` tag and nothing else.
- **`crossOriginIsolated` required.** The threaded WASM and WebGPU paths need `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`. `serve.py` already does this; document for anyone hosting elsewhere.
- **Match `clear-swift` defaults.** Mastering target Apple Podcasts (-19 LUFS), full strength, podcast preset. The web demo should produce the same output as the Swift call given the same input.
- **Comments.** Org rule applies.
