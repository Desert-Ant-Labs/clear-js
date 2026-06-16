// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// ERB filterbank — band widths are fixed at training time. The trained
// model only accepts these widths.

export const ERB_WIDTHS = Int32Array.of(
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20,
  24, 28, 31, 37, 42, 50, 56, 67,
);
export const N_ERB = ERB_WIDTHS.length;
export const N_FREQ_FROM_ERB = (() => { let s = 0; for (const w of ERB_WIDTHS) s += w; return s; })();

const WIDTH_RECIP = new Float32Array(N_ERB);
for (let i = 0; i < N_ERB; i++) WIDTH_RECIP[i] = 1.0 / ERB_WIDTHS[i];

/**
 * Mean per-bin power per band.
 * @param {Float32Array} power [nFrames, N_FREQ_FROM_ERB] row-major
 * @param {number} nFrames
 * @returns {Float32Array} [nFrames, N_ERB] row-major
 */
export function projectPower(power, nFrames) {
  if (nFrames <= 0 || power.length !== nFrames * N_FREQ_FROM_ERB) {
    return new Float32Array(0);
  }
  const out = new Float32Array(nFrames * N_ERB);
  for (let t = 0; t < nFrames; t++) {
    const rowIn  = t * N_FREQ_FROM_ERB;
    const rowOut = t * N_ERB;
    let off = 0;
    for (let band = 0; band < N_ERB; band++) {
      const w = ERB_WIDTHS[band];
      let sum = 0;
      const p0 = rowIn + off;
      for (let k = 0; k < w; k++) sum += power[p0 + k];
      out[rowOut + band] = sum * WIDTH_RECIP[band];
      off += w;
    }
  }
  return out;
}
