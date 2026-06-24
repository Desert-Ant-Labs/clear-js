// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// In-browser speech enhancement via ONNX Runtime Web.
//
// Pipeline: decode → STFT → erb + complex-spec features → chunked ONNX
// (T=200 frames) → ISTFT → optional mastering. WebGPU when available,
// threaded WASM otherwise.
//
// Peer dep: onnxruntime-web 1.20+. Loaded lazily; `setOrt()` to inject.
//
// Usage:
//   import { Clear } from './lib/clear.js';
//   const clear = await Clear.create({ variant: 'studio' });
//   const { audio } = await clear.enhance(file);  // Float32Array @ 48 kHz mono
//   await clear.dispose();

import { decodeToMono, SR } from './decode.js';
import { STFT, FFT_SIZE, HOP_SIZE, N_FREQ } from './stft.js';
import { computeFeatures, N_DF } from './features.js';
import { N_ERB } from './erb.js';
import { measureLUFS, applyLimiter, resolveMastering, MODEL_ATTENUATION_DB } from './r128.js';

const HF = 'https://huggingface.co/detail-co/clear/resolve/main';

const DEFAULT_MODEL_URLS = {
  studio:  `${HF}/clear-studio.onnx`,
  natural: `${HF}/clear-natural.onnx`,
};

const T_FIXED = 200;
const CONV_LOOKAHEAD = 2;

let ort = null;
let ortLoading = null;

/** Inject a pre-loaded ORT module. */
export function setOrt(module) { ort = module; ortLoading = null; }

async function loadOrt() {
  if (ort) return ort;
  if (!ortLoading) {
    ortLoading = import(
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.min.mjs'
    ).then((m) => { ort = m; return m; });
  }
  return ortLoading;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

export class Clear {
  #session;
  #buf;
  #opts;
  #backend;
  #variant;
  #stft;

  /**
   * Load a variant and return a ready-to-use Clear.
   *
   * @param {object} options
   * @param {'studio'|'natural'} [options.variant='studio']
   * @param {Record<'studio'|'natural', string>} [options.modelURLs]
   * @param {boolean} [options.forceWasm]  skip WebGPU even when available
   * @param {number}  [options.numThreads] override WASM thread count. Lower
   *   values reduce SharedArrayBuffer pressure: useful when the browser is
   *   hitting wasm RangeError after repeated reloads in the same tab.
   * @param {(loaded:number, total:number)=>void} [options.onDownloadProgress]
   * @param {(phase:'compiling-webgpu'|'compiling-wasm')=>void} [options.onPhase]
   */
  static async create(options = {}) {
    const variant = options.variant ?? 'studio';
    const modelURLs = options.modelURLs ?? DEFAULT_MODEL_URLS;
    const url = modelURLs[variant];
    if (!url) throw new Error(`unknown variant: ${variant}`);

    const ortMod = await loadOrt();
    const buf = await fetchWithProgress(url, options.onDownloadProgress);

    ortMod.env.logLevel = 'error';
    ortMod.env.wasm.simd = true;
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    const autoThreads = (self.crossOriginIsolated && !isSafari)
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;
    ortMod.env.wasm.numThreads = options.numThreads ?? autoThreads;

    const sessionOptions = {
      enableMemPattern: false,
      enableCpuMemArena: false,
      executionMode: 'sequential',
      graphOptimizationLevel: 'all',
    };

    let session = null;
    let backend = 'wasm';
    if ('gpu' in navigator && !options.forceWasm) {
      options.onPhase?.('compiling-webgpu');
      try {
        session = await ortMod.InferenceSession.create(buf, {
          executionProviders: ['webgpu'], ...sessionOptions,
        });
        backend = 'webgpu';
      } catch (e) {
        console.log('[clear] WebGPU init failed, falling back to WASM:', e?.message || e);
      }
    }
    if (!session) {
      options.onPhase?.('compiling-wasm');
      session = await ortMod.InferenceSession.create(buf, {
        executionProviders: ['wasm'], ...sessionOptions,
      });
    }

    return new Clear({
      session, buf,
      opts: { ...sessionOptions, executionProviders: [backend] },
      backend, variant,
    });
  }

  /** @internal */
  constructor({ session, buf, opts, backend, variant }) {
    this.#session = session;
    this.#buf = buf;
    this.#opts = opts;
    this.#backend = backend;
    this.#variant = variant;
    this.#stft = new STFT(FFT_SIZE, HOP_SIZE);
  }

  get backend() { return this.#backend; }
  get variant() { return this.#variant; }
  get sampleRate() { return SR; }

  /**
   * Enhance an audio file or raw PCM buffer.
   *
   * @param {File|Blob|Float32Array} input
   *   File / Blob → decoded to mono 48 kHz first.
   *   Float32Array → assumed mono PCM at 48 kHz.
   * @param {object} [options]
   * @param {'applePodcasts'|'spotify'|'youtube'|'broadcast'|'bypass'|null|object} [options.mastering='applePodcasts']
   *   Loudness target. Defaults to Apple Podcasts (−19 LUFS, −1.5 dBTP).
   *   Pass `'bypass'` / `null` / `false` for raw model output.
   * @param {(stage:'decode'|'inference'|'mastering', value:number, ctx?:object)=>void} [options.onProgress]
   * @returns {Promise<{
   *   audio: Float32Array,
   *   durationSec: number,
   *   sampleRate: number,
   *   measuredLUFS: number|null,
   *   measuredTruePeakDBFS: number|null
   * }>}
   */
  async enhance(input, options = {}) {
    const onProgress = options.onProgress;
    const mastering = resolveMastering(
      options.mastering === undefined ? 'applePodcasts' : options.mastering);

    onProgress?.('decode', 0);
    const audio = (input instanceof Float32Array)
      ? input
      : await decodeToMono(input);
    onProgress?.('decode', 1);

    const sanitized = sanitize(audio);

    // Bypass path skips K-weighting; only measure when mastering is on.
    let inputLUFS = Infinity;
    if (mastering.enabled) {
      inputLUFS = measureLUFS(sanitized, SR);
    }

    const padded = padToWindowMultiple(sanitized);
    const { real, imag, nFrames } = this.#stft.forward(padded);
    if (nFrames === 0) {
      return {
        audio: sanitized,
        durationSec: sanitized.length / SR,
        sampleRate: SR,
        measuredLUFS: null,
        measuredTruePeakDBFS: null,
      };
    }

    const { featErb, featSpecReal, featSpecImag } =
      computeFeatures(real, imag, nFrames, N_FREQ);

    const enhancedReal = new Float32Array(real.length);
    const enhancedImag = new Float32Array(imag.length);

    const T = T_FIXED;
    const nChunks = Math.ceil(nFrames / T);
    const specBuf = new Float32Array(T * N_FREQ * 2);
    const erbBuf  = new Float32Array(T * N_ERB);
    const dfBuf   = new Float32Array(T * N_DF  * 2);

    for (let c = 0; c < nChunks; c++) {
      const start = c * T;
      const end   = Math.min(start + T, nFrames);
      await this.#runChunk(start, end, nFrames,
        real, imag, featErb, featSpecReal, featSpecImag,
        enhancedReal, enhancedImag, specBuf, erbBuf, dfBuf);
      onProgress?.('inference', (c + 1) / nChunks, { chunk: c + 1, totalChunks: nChunks });
      await yieldToBrowser();
    }

    const enhancedTime = this.#stft.inverse(enhancedReal, enhancedImag, nFrames);
    let out = enhancedTime.slice(0, sanitized.length);

    // Mastering: gain to target LUFS (capped at maxLoudnessGainDB), then
    // look-ahead true-peak limiter. Matches the Swift mastering chain.
    let measuredLUFS = null, measuredTP = null;
    if (mastering.enabled && Number.isFinite(inputLUFS)) {
      onProgress?.('mastering', 0);
      const predictedEnhancedLUFS = inputLUFS - MODEL_ATTENUATION_DB;
      const requestedGainDB = mastering.integratedLUFS - predictedEnhancedLUFS;
      const effectiveGainDB = Math.min(requestedGainDB, mastering.maxLoudnessGainDB);
      const masterGain = Math.pow(10, effectiveGainDB / 20);
      for (let i = 0; i < out.length; i++) out[i] *= masterGain;
      out = applyLimiter(out, SR, mastering.truePeakDBTP);
      measuredLUFS = predictedEnhancedLUFS + effectiveGainDB;
      measuredTP   = mastering.truePeakDBTP;
      onProgress?.('mastering', 1);
    }

    return {
      audio: out,
      durationSec: out.length / SR,
      sampleRate: SR,
      measuredLUFS,
      measuredTruePeakDBFS: measuredTP,
    };
  }

  async #runChunk(start, end, nFrames,
                  specReal, specImag, featErb, featSpecReal, featSpecImag,
                  outReal, outImag, specBuf, erbBuf, dfBuf) {
    const T = T_FIXED;
    const nErb = N_ERB;
    const nDf  = N_DF;
    const nFreq = N_FREQ;

    specBuf.fill(0); erbBuf.fill(0); dfBuf.fill(0);

    // Lookahead shift: chunk frame t reads featGlobal[start + t + CONV_LOOKAHEAD].
    const tStart = Math.max(0, -start - CONV_LOOKAHEAD);
    const tEnd   = Math.min(T, nFrames - start - CONV_LOOKAHEAD);
    if (tEnd > tStart) {
      const srcFrameStart = start + tStart + CONV_LOOKAHEAD;
      erbBuf.set(
        featErb.subarray(srcFrameStart * nErb, (srcFrameStart + (tEnd - tStart)) * nErb),
        tStart * nErb,
      );
      // Interleave (re, im) into the dfBuf slot.
      for (let t = 0; t < (tEnd - tStart); t++) {
        const srcOff = (srcFrameStart + t) * nDf;
        const dstOff = (tStart + t) * nDf * 2;
        for (let f = 0; f < nDf; f++) {
          dfBuf[dstOff + f * 2]     = featSpecReal[srcOff + f];
          dfBuf[dstOff + f * 2 + 1] = featSpecImag[srcOff + f];
        }
      }
    }

    // Pack raw spec for this chunk into [B=1, C=1, T, F, 2].
    const sStart = Math.max(0, -start);
    const sEnd   = Math.min(T, nFrames - start);
    if (sEnd > sStart) {
      const srcFrameStart = start + sStart;
      for (let t = 0; t < (sEnd - sStart); t++) {
        const srcOff = (srcFrameStart + t) * nFreq;
        const dstOff = (sStart + t) * nFreq * 2;
        for (let f = 0; f < nFreq; f++) {
          specBuf[dstOff + f * 2]     = specReal[srcOff + f];
          specBuf[dstOff + f * 2 + 1] = specImag[srcOff + f];
        }
      }
    }

    const tensors = {
      spec:      new ort.Tensor('float32', specBuf, [1, 1, T, nFreq, 2]),
      feat_erb:  new ort.Tensor('float32', erbBuf,  [1, 1, T, nErb]),
      feat_spec: new ort.Tensor('float32', dfBuf,   [1, 1, T, nDf, 2]),
    };
    const { spec_enhanced } = await this.#session.run(tensors);
    const enhanced = spec_enhanced.data;

    const validFrames = end - start;
    for (let t = 0; t < validFrames; t++) {
      const srcOff = t * nFreq * 2;
      const dstOff = (start + t) * nFreq;
      for (let f = 0; f < nFreq; f++) {
        outReal[dstOff + f] = enhanced[srcOff + f * 2];
        outImag[dstOff + f] = enhanced[srcOff + f * 2 + 1];
      }
    }
    spec_enhanced.dispose?.();
  }

  /** Free the model session. Safe to call multiple times. */
  async dispose() {
    try { await this.#session?.release(); } catch {}
    this.#session = null;
    this.#buf = null;
  }
}

function yieldToBrowser() {
  if (typeof requestAnimationFrame === 'function') {
    return new Promise((r) => requestAnimationFrame(r));
  }
  return new Promise((r) => setTimeout(r, 0));
}

function sanitize(samples) {
  let nonFinite = false;
  for (let i = 0; i < samples.length; i++) {
    if (!Number.isFinite(samples[i])) { nonFinite = true; break; }
  }
  if (!nonFinite) return samples;
  const out = new Float32Array(samples);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) out[i] = 0;
  }
  return out;
}

function padToWindowMultiple(samples) {
  const needed = Math.max(
    FFT_SIZE,
    Math.ceil(samples.length / HOP_SIZE) * HOP_SIZE + FFT_SIZE,
  );
  if (samples.length >= needed) return samples;
  const out = new Float32Array(needed);
  out.set(samples);
  return out;
}

export { SR, FFT_SIZE, HOP_SIZE, N_FREQ, N_ERB, N_DF };
export { encodeWav } from './wav.js';
export { decodeToMono } from './decode.js';
export { measureLUFS, applyLimiter, PRESETS as MASTERING_PRESETS } from './r128.js';
