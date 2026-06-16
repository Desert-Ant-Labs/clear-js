// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// Decode any browser-supported audio file to a mono Float32Array at the
// target sample rate (48 kHz for clear). Uses an OfflineAudioContext for
// resampling so the math is the browser's native, not a linear interp.

export const SR = 48_000;

/**
 * @param {File|Blob|ArrayBuffer} input
 * @param {number} sampleRate target SR (default 48000)
 * @returns {Promise<Float32Array>} mono PCM at the requested SR
 */
export async function decodeToMono(input, sampleRate = SR) {
  const buf = (input instanceof ArrayBuffer) ? input : await input.arrayBuffer();
  const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await tmpCtx.decodeAudioData(buf.slice(0));
  } finally {
    tmpCtx.close();
  }
  const mono = mixToMono(decoded);
  if (decoded.sampleRate === sampleRate) return mono;
  return await resampleOffline(mono, decoded.sampleRate, sampleRate);
}

function mixToMono(audioBuffer) {
  const n = audioBuffer.length;
  const c = audioBuffer.numberOfChannels;
  if (c === 1) return audioBuffer.getChannelData(0).slice();
  const out = new Float32Array(n);
  for (let ch = 0; ch < c; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < n; i++) out[i] += data[i] / c;
  }
  return out;
}

async function resampleOffline(mono, srIn, srOut) {
  const outLen = Math.ceil(mono.length * srOut / srIn);
  const ctx = new OfflineAudioContext(1, outLen, srOut);
  const src = ctx.createBuffer(1, mono.length, srIn);
  src.copyToChannel(mono, 0);
  const node = ctx.createBufferSource();
  node.buffer = src;
  node.connect(ctx.destination);
  node.start(0);
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}
