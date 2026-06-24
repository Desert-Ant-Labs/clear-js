---
title: Clear: on-device speech enhancement
emoji: 🌊
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Denoise + dereverb at 48 kHz, on device, in the browser
custom_headers:
  cross-origin-opener-policy: same-origin
  cross-origin-embedder-policy: credentialless
  cross-origin-resource-policy: cross-origin
---

# Clear: browser demo

On-device speech enhancement (denoise + dereverb) running entirely in the
browser via ONNX Runtime Web. Audio never leaves your machine.

- **clear-studio**: fuller cleanup; the default for noisy or reverberant rooms.
- **clear-natural**: gentler; preserves more of the original room tone.
- 48 kHz sample rate, mastered to broadcast loudness presets, raw / enhanced A/B.
- WebGPU on Chrome / Safari Tahoe / Firefox; WASM threaded fallback elsewhere.

Model card: [detail-co/clear](https://huggingface.co/detail-co/clear).
