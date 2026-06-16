// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// EBU R128 / ITU-R BS.1770-4 integrated LUFS + look-ahead true-peak
// limiter. Non-streaming (buffers in RAM); K-weighting coefficients are
// the BS.1770-4 reference filters at 48 kHz.

const KWEIGHT_PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285];
const KWEIGHT_PRE_A = [1.0, -1.69065929318241, 0.73248077421585];
const KWEIGHT_RLB_B = [1.0, -2.0, 1.0];
const KWEIGHT_RLB_A = [1.0, -1.99004745483398, 0.99007225036621];

/**
 * Direct-form-I biquad. Returns a new Float32Array.
 */
function biquad(x, b, a) {
  const a0 = a[0];
  const b0 = b[0] / a0, b1 = b[1] / a0, b2 = b[2] / a0;
  const a1 = a[1] / a0, a2 = a[2] / a0;
  const n = x.length;
  const y = new Float32Array(n);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    y[i] = yi;
    x2 = x1; x1 = xi;
    y2 = y1; y1 = yi;
  }
  return y;
}

/**
 * BS.1770-gated integrated LUFS for a single (mono) Float32Array.
 *
 * @param {Float32Array} audio  mono PCM
 * @param {number} sampleRate
 * @returns {number}  integrated LUFS, -Infinity for silence
 */
export function measureLUFS(audio, sampleRate) {
  const blockLen = Math.round(0.400 * sampleRate);
  const blockHop = Math.round(0.100 * sampleRate);
  if (audio.length < blockLen) return -Infinity;

  const pre = biquad(audio, KWEIGHT_PRE_B, KWEIGHT_PRE_A);
  const kw  = biquad(pre,   KWEIGHT_RLB_B, KWEIGHT_RLB_A);

  const blocks = [];
  for (let i = 0; i + blockLen <= kw.length; i += blockHop) {
    let sq = 0;
    for (let j = 0; j < blockLen; j++) {
      const v = kw[i + j];
      sq += v * v;
    }
    const ms = sq / blockLen;
    blocks.push(ms > 1e-12 ? -0.691 + 10 * Math.log10(ms) : -Infinity);
  }
  return gatedIntegrated(blocks);
}

function gatedIntegrated(blocks) {
  const absGated = blocks.filter((b) => b > -70.0);
  if (!absGated.length) return -Infinity;
  let sumE = 0;
  for (const b of absGated) sumE += Math.pow(10, b / 10);
  const meanE = sumE / absGated.length;
  const relThresh = 10 * Math.log10(meanE) - 10;
  const relGated = absGated.filter((b) => b > relThresh);
  if (!relGated.length) return -Infinity;
  let sumE2 = 0;
  for (const b of relGated) sumE2 += Math.pow(10, b / 10);
  return 10 * Math.log10(sumE2 / relGated.length);
}

/**
 * Look-ahead true-peak limiter (5 ms look-ahead, 50 ms release).
 *
 * @param {Float32Array} audio  mono PCM
 * @param {number} sampleRate
 * @param {number} ceilDBTP     ceiling in dBFS (negative — e.g. -1.5)
 * @returns {Float32Array}      limited PCM, same length
 */
export function applyLimiter(audio, sampleRate, ceilDBTP) {
  const n = audio.length;
  if (n === 0) return audio;
  const ceil = Math.pow(10, ceilDBTP / 20);
  const lookahead = Math.round(0.005 * sampleRate);
  const releaseCoef = Math.exp(-1 / (sampleRate * 0.050));
  const out = new Float32Array(n);

  // |x| precomputed once — saves Math.abs lookups in the inner loop.
  const ax = new Float32Array(n);
  for (let i = 0; i < n; i++) ax[i] = audio[i] >= 0 ? audio[i] : -audio[i];

  const dq = new Int32Array(n + 1);
  let head = 0, tail = 0;

  const seedN = Math.min(lookahead, n);
  for (let j = 0; j < seedN; j++) {
    const vj = ax[j];
    while (tail > head && ax[dq[tail - 1]] <= vj) tail--;
    dq[tail++] = j;
  }

  let env = 1.0;
  for (let i = 0; i < n; i++) {
    const end = i + lookahead;
    if (end < n) {
      const ve = ax[end];
      while (tail > head && ax[dq[tail - 1]] <= ve) tail--;
      dq[tail++] = end;
    }
    while (head < tail && dq[head] < i) head++;
    const maxAhead = head < tail ? ax[dq[head]] : 0;
    if (maxAhead > ceil) {
      const required = ceil / maxAhead;
      if (required < env) env = required;
    }
    out[i] = audio[i] * env;
    env = 1.0 - (1.0 - env) * releaseCoef;
  }
  return out;
}

/** Model-output attenuation in dB — feeds the mastering gain calc. */
export const MODEL_ATTENUATION_DB = 0.7;

/** Loudness presets matching `Clear.LoudnessPreset` in the Swift package. */
export const PRESETS = {
  applePodcasts: { integratedLUFS: -19.0, truePeakDBTP: -1.5, maxLoudnessGainDB: 9.0 },
  spotify:       { integratedLUFS: -14.0, truePeakDBTP: -1.5, maxLoudnessGainDB: 9.0 },
  youtube:       { integratedLUFS: -14.0, truePeakDBTP: -1.5, maxLoudnessGainDB: 9.0 },
  broadcast:     { integratedLUFS: -23.0, truePeakDBTP: -1.5, maxLoudnessGainDB: 9.0 },
};

/**
 * Resolve a mastering spec (string preset, object, or null) to a normalized
 * shape with `{integratedLUFS, truePeakDBTP, maxLoudnessGainDB, enabled}`.
 *
 * Accepts:
 *   - 'applePodcasts' | 'spotify' | 'youtube' | 'broadcast'
 *   - 'bypass' | 'off' | false | null
 *   - { integratedLUFS, truePeakDBTP?, maxLoudnessGainDB? }
 */
export function resolveMastering(spec) {
  if (spec === null || spec === undefined || spec === false
      || spec === 'bypass' || spec === 'off') {
    return { enabled: false };
  }
  if (typeof spec === 'string') {
    const p = PRESETS[spec];
    if (!p) throw new Error(`unknown mastering preset: ${spec}`);
    return { ...p, enabled: true };
  }
  const p = { ...PRESETS.applePodcasts, ...spec };
  return { ...p, enabled: true };
}
