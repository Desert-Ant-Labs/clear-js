//
// Core speech-enhancement model. Platform-agnostic: it takes a loaded ONNX
// Runtime session and operates on mono Float32 PCM at 48 kHz.
//
// Pipeline: STFT → erb + complex-spec features → chunked ONNX (T=200
// frames) → ISTFT → optional R128 mastering.

import { N_ERB } from "./erb.js";
import { computeFeatures, N_DF } from "./features.js";
import type { Ort, OrtSession } from "./ort.js";
import {
  applyLimiter,
  measureLUFS,
  MODEL_ATTENUATION_DB,
  resolveMastering,
  type MasteringSpec,
} from "./r128.js";
import { FFT_SIZE, HOP_SIZE, N_FREQ, STFT } from "./stft.js";

/** Model sample rate. Inputs must be mono PCM at this rate. */
export const SR = 48_000;

export type Variant = "studio" | "natural";

const T_FIXED = 200;
const CONV_LOOKAHEAD = 2;

export type EnhanceStage = "inference" | "mastering";
export type ProgressFn = (stage: EnhanceStage, value: number, ctx?: { chunk: number; totalChunks: number }) => void;

export interface EnhanceOptions {
  /**
   * Loudness target. Defaults to Apple Podcasts (−19 LUFS, −1.5 dBTP).
   * Pass `"bypass"` / `null` for raw model output.
   */
  mastering?: MasteringSpec;
  /** Progress callback across inference chunks and the mastering pass. */
  onProgress?: ProgressFn;
}

export interface EnhanceResult {
  /** Enhanced mono PCM at 48 kHz. */
  audio: Float32Array;
  durationSec: number;
  sampleRate: number;
  /** Integrated loudness of the output, or `null` when mastering is bypassed. */
  measuredLUFS: number | null;
  /** True-peak ceiling enforced, or `null` when mastering is bypassed. */
  measuredTruePeakDBFS: number | null;
}

export interface ClearModelInit {
  ort: Ort;
  session: OrtSession;
  variant: Variant;
  /** Execution provider in use, e.g. `"webgpu"`, `"wasm"`, `"cpu"`. */
  backend: string;
  /** Cooperative yield between chunks (the browser uses requestAnimationFrame). */
  yieldFn?: () => Promise<void>;
}

/** A loaded model. Reuse across many `enhance` calls; `dispose` when done. */
export class ClearModel {
  readonly #ort: Ort;
  #session: OrtSession | null;
  readonly #variant: Variant;
  readonly #backend: string;
  readonly #stft: STFT;
  readonly #yield: () => Promise<void>;

  constructor(init: ClearModelInit) {
    this.#ort = init.ort;
    this.#session = init.session;
    this.#variant = init.variant;
    this.#backend = init.backend;
    this.#stft = new STFT(FFT_SIZE, HOP_SIZE);
    this.#yield = init.yieldFn ?? (() => Promise.resolve());
  }

  get backend(): string {
    return this.#backend;
  }
  get variant(): Variant {
    return this.#variant;
  }
  get sampleRate(): number {
    return SR;
  }

  /** Enhance mono PCM at 48 kHz. */
  async enhance(pcm: Float32Array, options: EnhanceOptions = {}): Promise<EnhanceResult> {
    const session = this.#session;
    if (!session) throw new Error("clear: model has been disposed");
    const onProgress = options.onProgress;
    const mastering = resolveMastering(options.mastering === undefined ? "applePodcasts" : options.mastering);

    const sanitized = sanitize(pcm);

    let inputLUFS = Infinity;
    if (mastering.enabled) inputLUFS = measureLUFS(sanitized, SR);

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

    const { featErb, featSpecReal, featSpecImag } = computeFeatures(real, imag, nFrames, N_FREQ);

    const enhancedReal = new Float32Array(real.length);
    const enhancedImag = new Float32Array(imag.length);

    const T = T_FIXED;
    const nChunks = Math.ceil(nFrames / T);
    const specBuf = new Float32Array(T * N_FREQ * 2);
    const erbBuf = new Float32Array(T * N_ERB);
    const dfBuf = new Float32Array(T * N_DF * 2);

    for (let c = 0; c < nChunks; c++) {
      const start = c * T;
      const end = Math.min(start + T, nFrames);
      await this.#runChunk(
        session, start, end, nFrames,
        real, imag, featErb, featSpecReal, featSpecImag,
        enhancedReal, enhancedImag, specBuf, erbBuf, dfBuf,
      );
      onProgress?.("inference", (c + 1) / nChunks, { chunk: c + 1, totalChunks: nChunks });
      await this.#yield();
    }

    const enhancedTime = this.#stft.inverse(enhancedReal, enhancedImag, nFrames);
    let out = enhancedTime.slice(0, sanitized.length);

    let measuredLUFS: number | null = null;
    let measuredTP: number | null = null;
    if (mastering.enabled && Number.isFinite(inputLUFS)) {
      onProgress?.("mastering", 0);
      const predictedEnhancedLUFS = inputLUFS - MODEL_ATTENUATION_DB;
      const requestedGainDB = mastering.integratedLUFS - predictedEnhancedLUFS;
      const effectiveGainDB = Math.min(requestedGainDB, mastering.maxLoudnessGainDB);
      const masterGain = Math.pow(10, effectiveGainDB / 20);
      for (let i = 0; i < out.length; i++) out[i] *= masterGain;
      out = applyLimiter(out, SR, mastering.truePeakDBTP);
      measuredLUFS = predictedEnhancedLUFS + effectiveGainDB;
      measuredTP = mastering.truePeakDBTP;
      onProgress?.("mastering", 1);
    }

    return {
      audio: out,
      durationSec: out.length / SR,
      sampleRate: SR,
      measuredLUFS,
      measuredTruePeakDBFS: measuredTP,
    };
  }

  async #runChunk(
    session: OrtSession,
    start: number, end: number, nFrames: number,
    specReal: Float32Array, specImag: Float32Array,
    featErb: Float32Array, featSpecReal: Float32Array, featSpecImag: Float32Array,
    outReal: Float32Array, outImag: Float32Array,
    specBuf: Float32Array, erbBuf: Float32Array, dfBuf: Float32Array,
  ): Promise<void> {
    const T = T_FIXED;
    const nErb = N_ERB;
    const nDf = N_DF;
    const nFreq = N_FREQ;

    specBuf.fill(0);
    erbBuf.fill(0);
    dfBuf.fill(0);

    // Lookahead shift: chunk frame t reads featGlobal[start + t + CONV_LOOKAHEAD].
    const tStart = Math.max(0, -start - CONV_LOOKAHEAD);
    const tEnd = Math.min(T, nFrames - start - CONV_LOOKAHEAD);
    if (tEnd > tStart) {
      const srcFrameStart = start + tStart + CONV_LOOKAHEAD;
      erbBuf.set(
        featErb.subarray(srcFrameStart * nErb, (srcFrameStart + (tEnd - tStart)) * nErb),
        tStart * nErb,
      );
      for (let t = 0; t < tEnd - tStart; t++) {
        const srcOff = (srcFrameStart + t) * nDf;
        const dstOff = (tStart + t) * nDf * 2;
        for (let f = 0; f < nDf; f++) {
          dfBuf[dstOff + f * 2] = featSpecReal[srcOff + f];
          dfBuf[dstOff + f * 2 + 1] = featSpecImag[srcOff + f];
        }
      }
    }

    // Pack raw spec for this chunk into [B=1, C=1, T, F, 2].
    const sStart = Math.max(0, -start);
    const sEnd = Math.min(T, nFrames - start);
    if (sEnd > sStart) {
      const srcFrameStart = start + sStart;
      for (let t = 0; t < sEnd - sStart; t++) {
        const srcOff = (srcFrameStart + t) * nFreq;
        const dstOff = (sStart + t) * nFreq * 2;
        for (let f = 0; f < nFreq; f++) {
          specBuf[dstOff + f * 2] = specReal[srcOff + f];
          specBuf[dstOff + f * 2 + 1] = specImag[srcOff + f];
        }
      }
    }

    const Tensor = this.#ort.Tensor;
    const feeds = {
      spec: new Tensor("float32", specBuf, [1, 1, T, nFreq, 2]),
      feat_erb: new Tensor("float32", erbBuf, [1, 1, T, nErb]),
      feat_spec: new Tensor("float32", dfBuf, [1, 1, T, nDf, 2]),
    };
    const results = await session.run(feeds);
    const spec_enhanced = results.spec_enhanced;
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
  async dispose(): Promise<void> {
    try {
      await this.#session?.release?.();
    } catch {
      /* best-effort */
    }
    this.#session = null;
  }
}

function sanitize(samples: Float32Array): Float32Array {
  let nonFinite = false;
  for (let i = 0; i < samples.length; i++) {
    if (!Number.isFinite(samples[i])) {
      nonFinite = true;
      break;
    }
  }
  if (!nonFinite) return samples;
  const out = new Float32Array(samples);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) out[i] = 0;
  }
  return out;
}

function padToWindowMultiple(samples: Float32Array): Float32Array {
  const needed = Math.max(FFT_SIZE, Math.ceil(samples.length / HOP_SIZE) * HOP_SIZE + FFT_SIZE);
  if (samples.length >= needed) return samples;
  const out = new Float32Array(needed);
  out.set(samples);
  return out;
}
