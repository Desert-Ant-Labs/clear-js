// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// STFT / ISTFT: Vorbis window, n_fft=960, hop=480, forward gain
// wnorm = 2·hop/n_fft² = 1/960. Pre-pads (n_fft − hop) zeros; trims them
// off the inverse output. Unnormalized DFT both directions — wnorm plus
// the squared-Vorbis COLA closes the round-trip at unity.

import { BluesteinFFT } from './fft.js';

export const FFT_SIZE = 960;
export const HOP_SIZE = 480;
export const N_FREQ   = FFT_SIZE / 2 + 1;

function vorbisWindow(N) {
  const w = new Float32Array(N);
  const halfN = N / 2;
  for (let n = 0; n < N; n++) {
    const s = Math.sin(0.5 * Math.PI * (n + 0.5) / halfN);
    w[n] = Math.sin(0.5 * Math.PI * s * s);
  }
  return w;
}

export class STFT {
  constructor(fftSize = FFT_SIZE, hopSize = HOP_SIZE) {
    this.fftSize = fftSize;
    this.hopSize = hopSize;
    this.nFreq   = fftSize / 2 + 1;
    this.window  = vorbisWindow(fftSize);
    this.wnorm   = (2 * hopSize) / (fftSize * fftSize);
    this.fft     = new BluesteinFFT(fftSize);

    this._rIn  = new Float64Array(fftSize);
    this._iIn  = new Float64Array(fftSize);
    this._rOut = new Float64Array(fftSize);
    this._iOut = new Float64Array(fftSize);
  }

  /**
   * @param {Float32Array} audio
   * @returns {{ real: Float32Array, imag: Float32Array, nFrames: number }}
   *   `real` and `imag` are [nFrames, nFreq] row-major.
   */
  forward(audio) {
    const { fftSize, hopSize, nFreq, window, wnorm, fft } = this;
    const prePad = fftSize - hopSize;
    const paddedLen = prePad + audio.length;
    if (paddedLen < fftSize) {
      return { real: new Float32Array(0), imag: new Float32Array(0), nFrames: 0 };
    }
    const padded = new Float32Array(paddedLen);
    padded.set(audio, prePad);

    const nFrames = ((paddedLen - fftSize) / hopSize | 0) + 1;
    const real = new Float32Array(nFrames * nFreq);
    const imag = new Float32Array(nFrames * nFreq);

    const rIn = this._rIn, iIn = this._iIn, rOut = this._rOut, iOut = this._iOut;
    for (let t = 0; t < nFrames; t++) {
      const off = t * hopSize;
      for (let i = 0; i < fftSize; i++) {
        rIn[i] = padded[off + i] * window[i];
        iIn[i] = 0;
      }
      fft.transform(rIn, iIn, rOut, iOut, false);
      const base = t * nFreq;
      for (let k = 0; k < nFreq; k++) {
        real[base + k] = rOut[k] * wnorm;
        imag[base + k] = iOut[k] * wnorm;
      }
    }
    return { real, imag, nFrames };
  }

  /**
   * @param {Float32Array} real shape [nFrames, nFreq]
   * @param {Float32Array} imag shape [nFrames, nFreq]
   * @param {number} nFrames
   * @returns {Float32Array} time-domain output, prePad trimmed off the front
   */
  inverse(real, imag, nFrames) {
    const { fftSize, hopSize, nFreq, window, fft } = this;
    const prePad = fftSize - hopSize;
    const rawLen = (nFrames - 1) * hopSize + fftSize;
    const out = new Float32Array(rawLen);

    const rIn = this._rIn, iIn = this._iIn, rOut = this._rOut, iOut = this._iOut;
    const mirrorLen = fftSize - nFreq;
    for (let t = 0; t < nFrames; t++) {
      const base = t * nFreq;
      for (let k = 0; k < nFreq; k++) {
        rIn[k] = real[base + k];
        iIn[k] = imag[base + k];
      }
      for (let m = 0; m < mirrorLen; m++) {
        const srcK = mirrorLen - m;
        rIn[nFreq + m] =  real[base + srcK];
        iIn[nFreq + m] = -imag[base + srcK];
      }
      fft.transform(rIn, iIn, rOut, iOut, true);
      const off = t * hopSize;
      for (let i = 0; i < fftSize; i++) {
        out[off + i] += rOut[i] * window[i];
      }
    }
    if (rawLen > prePad) return out.subarray(prePad, rawLen);
    return out;
  }
}
