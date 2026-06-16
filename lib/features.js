// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// Per-frame feature extraction (erb dB-norm + complex spec unit-norm).
// The model is highly scale-sensitive — keep this byte-exact with the
// Swift / Android references.

import { projectPower, N_ERB } from './erb.js';

export const N_DF = 96;
export const NORM_ALPHA = 0.99;

/**
 * @param {Float32Array} real shape [nFrames, nFreq] row-major
 * @param {Float32Array} imag shape [nFrames, nFreq] row-major
 * @param {number} nFrames
 * @param {number} nFreq
 * @returns {{ featErb: Float32Array, featSpecReal: Float32Array, featSpecImag: Float32Array }}
 *   featErb [nFrames, N_ERB], featSpec* [nFrames, N_DF]
 */
export function computeFeatures(real, imag, nFrames, nFreq) {
  const alpha = NORM_ALPHA;
  const oneMinus = 1.0 - alpha;
  const nDf = N_DF;
  const nErb = N_ERB;

  const power = new Float32Array(nFrames * nFreq);
  for (let i = 0; i < power.length; i++) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }

  const erbPower = projectPower(power, nFrames);
  const erbDB = new Float32Array(erbPower.length);
  const EPS = 1e-10;
  for (let i = 0; i < erbDB.length; i++) {
    erbDB[i] = 10 * Math.log10(erbPower[i] + EPS);
  }

  const featErb = new Float32Array(nFrames * nErb);
  const erbState = new Float32Array(nErb);
  const mnStep = (-90.0 - -60.0) / (nErb - 1);
  for (let f = 0; f < nErb; f++) erbState[f] = -60.0 + f * mnStep;
  for (let t = 0; t < nFrames; t++) {
    const off = t * nErb;
    for (let f = 0; f < nErb; f++) {
      erbState[f] = erbDB[off + f] * oneMinus + erbState[f] * alpha;
      featErb[off + f] = (erbDB[off + f] - erbState[f]) / 40.0;
    }
  }

  const featSpecReal = new Float32Array(nFrames * nDf);
  const featSpecImag = new Float32Array(nFrames * nDf);
  const s = new Float32Array(nDf);
  const unStep = (0.0001 - 0.001) / (nDf - 1);
  for (let f = 0; f < nDf; f++) s[f] = 0.001 + f * unStep;

  for (let t = 0; t < nFrames; t++) {
    const inOff  = t * nFreq;
    const outOff = t * nDf;
    for (let f = 0; f < nDf; f++) {
      const rr = real[inOff + f];
      const ii = imag[inOff + f];
      const mag = Math.sqrt(rr * rr + ii * ii);
      s[f] = mag * oneMinus + s[f] * alpha;
      const inv = 1.0 / Math.sqrt(s[f]);
      featSpecReal[outOff + f] = rr * inv;
      featSpecImag[outOff + f] = ii * inv;
    }
  }

  return { featErb, featSpecReal, featSpecImag };
}
