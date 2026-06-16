// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// Bluestein FFT wrapping a radix-2 Cooley-Tukey kernel — needed because
// the clear STFT length N=960 isn't a power of two. Unnormalized DFT both
// directions (no 1/N); the STFT layer's `wnorm` compensates.

/**
 * Bluestein planner for a fixed length N. Reuse across many frames.
 */
export class BluesteinFFT {
  constructor(N) {
    this.N = N;
    let M = 1;
    while (M < 2 * N - 1) M <<= 1;
    this.M = M;

    const aRe = new Float64Array(N);
    const aIm = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const ang = -Math.PI * ((n * n) % (2 * N)) / N;
      aRe[n] = Math.cos(ang);
      aIm[n] = Math.sin(ang);
    }
    this.aRe = aRe; this.aIm = aIm;

    const bfRe = new Float64Array(M);
    const bfIm = new Float64Array(M);
    bfRe[0] = aRe[0]; bfIm[0] = -aIm[0];
    for (let m = 1; m < N; m++) {
      bfRe[m]     = aRe[m]; bfIm[m]     = -aIm[m];
      bfRe[M - m] = aRe[m]; bfIm[M - m] = -aIm[m];
    }
    this.Bf_Re = bfRe;
    this.Bf_Im = bfIm;
    fftRadix2Inplace(this.Bf_Re, this.Bf_Im, M, false);

    const biRe = new Float64Array(M);
    const biIm = new Float64Array(M);
    biRe[0] = aRe[0]; biIm[0] = aIm[0];
    for (let m = 1; m < N; m++) {
      biRe[m]     = aRe[m]; biIm[m]     = aIm[m];
      biRe[M - m] = aRe[m]; biIm[M - m] = aIm[m];
    }
    this.Bi_Re = biRe;
    this.Bi_Im = biIm;
    fftRadix2Inplace(this.Bi_Re, this.Bi_Im, M, false);

    this._sRe = new Float64Array(M);
    this._sIm = new Float64Array(M);
  }

  /**
   * @param {Float32Array|Float64Array} xRe  length ≥ N
   * @param {Float32Array|Float64Array} xIm  length ≥ N (zero for real input)
   * @param {Float32Array|Float64Array} outRe length ≥ N — written
   * @param {Float32Array|Float64Array} outIm length ≥ N — written
   * @param {boolean} inverse  false=forward, true=inverse (no 1/N either way)
   */
  transform(xRe, xIm, outRe, outIm, inverse = false) {
    const { N, M, aRe, aIm, _sRe: sRe, _sIm: sIm } = this;
    const BRe = inverse ? this.Bi_Re : this.Bf_Re;
    const BIm = inverse ? this.Bi_Im : this.Bf_Im;
    const cSign = inverse ? -1 : +1;

    sRe.fill(0); sIm.fill(0);
    for (let n = 0; n < N; n++) {
      const cr = aRe[n], ci = cSign * aIm[n];
      const xr = xRe[n], xi = xIm[n];
      sRe[n] = xr * cr - xi * ci;
      sIm[n] = xr * ci + xi * cr;
    }

    fftRadix2Inplace(sRe, sIm, M, false);
    for (let i = 0; i < M; i++) {
      const sr = sRe[i], si = sIm[i];
      const br = BRe[i], bi = BIm[i];
      sRe[i] = sr * br - si * bi;
      sIm[i] = sr * bi + si * br;
    }
    fftRadix2Inplace(sRe, sIm, M, true);

    for (let k = 0; k < N; k++) {
      const cr = aRe[k], ci = cSign * aIm[k];
      const yr = sRe[k], yi = sIm[k];
      outRe[k] = yr * cr - yi * ci;
      outIm[k] = yr * ci + yi * cr;
    }
  }
}

function fftRadix2Inplace(re, im, M, inverse) {
  for (let i = 1, j = 0; i < M; i++) {
    let bit = M >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= M; len <<= 1) {
    const half = len >> 1;
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wReStep = Math.cos(ang);
    const wImStep = Math.sin(ang);
    for (let i = 0; i < M; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < half; j++) {
        const ur = re[i + j], ui = im[i + j];
        const vr = re[i + j + half] * wr - im[i + j + half] * wi;
        const vi = re[i + j + half] * wi + im[i + j + half] * wr;
        re[i + j]        = ur + vr;
        im[i + j]        = ui + vi;
        re[i + j + half] = ur - vr;
        im[i + j + half] = ui - vi;
        const nwr = wr * wReStep - wi * wImStep;
        wi        = wr * wImStep + wi * wReStep;
        wr        = nwr;
      }
    }
  }
  if (inverse) {
    const s = 1 / M;
    for (let i = 0; i < M; i++) { re[i] *= s; im[i] *= s; }
  }
}
