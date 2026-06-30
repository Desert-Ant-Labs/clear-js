# Third-party notices: clear-js

The `clear` models this package runs (`clear-studio`, `clear-natural`) are downloaded at runtime from `huggingface.co/desert-ant-labs/clear` and are a fine-tune of an upstream architecture. The upstream license applies to that base; nothing in the Desert Ant Labs Source-Available License overrides it.

For training-time dependencies and data sources see [`THIRD_PARTY_NOTICES.md` in clear-training](https://github.com/Desert-Ant-Labs/clear-training/blob/main/hf/THIRD_PARTY_NOTICES.md). This file covers only what's used at inference time.

## Runtime model components

### DeepFilterNet 3 (Hendrik Schröter)
- **Source:** [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (DFN3-half configuration).
- **License:** MIT License.
- **Use:** Base architecture for both `clear-studio` and `clear-natural`. Fine-tuned on the Desert Ant Labs speech corpus and exported to ONNX. The shipped `.onnx` is bit-identical to the trained graph within fp16 precision.

## Runtime library dependencies

These are **peer dependencies** — installed by the host application, not bundled in this package — but required for it to run.

### ONNX Runtime (Microsoft)
- **Packages:** `onnxruntime-web` (browser/workers), `onnxruntime-node` (Node).
- **Source:** [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime).
- **License:** MIT License.
- **Use:** Executes the ONNX model graph.

The STFT/ISTFT, ERB filterbank, feature extraction, WAV codec, and R128 loudness mastering are original Desert Ant Labs implementations with no third-party code.

## License-notice retention

The MIT license requires preservation of the upstream copyright and permission notice in substantial portions of the software.

### DeepFilterNet 3

```
MIT License

Copyright (c) 2022 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The DeepFilterNet 3 notice also ships inside `huggingface.co/desert-ant-labs/clear` alongside the model weights. ONNX Runtime ships its own MIT license text within its npm package.
