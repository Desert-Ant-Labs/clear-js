import assert from "node:assert/strict";
import { test } from "node:test";

import { ClearModel, SR, type Variant } from "../src/model.js";
import type { Ort, OrtSession, OrtTensor } from "../src/ort.js";
import { applyLimiter, measureLUFS, MASTERING_PRESETS, resolveMastering } from "../src/r128.js";
import { FFT_SIZE, HOP_SIZE, STFT } from "../src/stft.js";
import { decodeWav, encodeWav } from "../src/wav.js";

function sine(freq: number, seconds: number, amp = 0.5): Float32Array {
  const n = Math.round(seconds * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return out;
}

function rms(a: Float32Array, start = 0, end = a.length): number {
  let s = 0;
  for (let i = start; i < end; i++) s += a[i] * a[i];
  return Math.sqrt(s / (end - start));
}

test("STFT round-trips a sine at unity gain", () => {
  const x = sine(440, 0.5);
  const stft = new STFT(FFT_SIZE, HOP_SIZE);
  const { real, imag, nFrames } = stft.forward(x);
  assert.ok(nFrames > 0);
  const y = stft.inverse(real, imag, nFrames).slice(0, x.length);
  // Compare the interior (skip window ramp-up/down at the very edges).
  const a = FFT_SIZE;
  const b = x.length - FFT_SIZE;
  let err = 0;
  for (let i = a; i < b; i++) err += (y[i] - x[i]) ** 2;
  const nrmse = Math.sqrt(err / (b - a)) / rms(x, a, b);
  assert.ok(nrmse < 1e-3, `STFT round-trip NRMSE ${nrmse} too high`);
});

test("WAV encode → decode round-trips mono PCM", () => {
  const x = sine(220, 0.1, 0.7);
  const { samples, sampleRate } = decodeWav(encodeWav(x, SR));
  assert.equal(sampleRate, SR);
  assert.equal(samples.length, x.length);
  // 16-bit quantization error is bounded by 1 LSB.
  let maxErr = 0;
  for (let i = 0; i < x.length; i++) maxErr = Math.max(maxErr, Math.abs(samples[i] - x[i]));
  assert.ok(maxErr < 2 / 32768, `max WAV round-trip error ${maxErr}`);
});

test("decodeWav downmixes stereo to mono", () => {
  // Build a 2-channel 16-bit WAV by hand: L = +0.5, R = -0.5 → mono 0.
  const frames = 100;
  const buf = new ArrayBuffer(44 + frames * 4);
  const v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + frames * 4, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 2, true); // stereo
  v.setUint32(24, SR, true);
  v.setUint32(28, SR * 4, true);
  v.setUint16(32, 4, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, frames * 4, true);
  for (let i = 0; i < frames; i++) {
    v.setInt16(44 + i * 4, 0.5 * 0x7fff, true);
    v.setInt16(44 + i * 4 + 2, -0.5 * 0x7fff, true);
  }
  const { samples } = decodeWav(buf);
  assert.equal(samples.length, frames);
  assert.ok(Math.max(...samples.map(Math.abs)) < 1e-3);
});

test("measureLUFS tracks amplitude (≈ +6 LU per doubling)", () => {
  const loud = measureLUFS(sine(1000, 1.0, 0.5), SR);
  const quiet = measureLUFS(sine(1000, 1.0, 0.25), SR);
  assert.ok(Number.isFinite(loud) && Number.isFinite(quiet));
  assert.ok(Math.abs(loud - quiet - 6.0) < 0.5, `expected ~6 LU, got ${loud - quiet}`);
});

test("applyLimiter holds the true-peak ceiling", () => {
  const x = sine(500, 0.2, 0.95);
  const limited = applyLimiter(x, SR, -6); // ceiling 0.5012
  const ceil = Math.pow(10, -6 / 20);
  assert.ok(Math.max(...limited.map(Math.abs)) <= ceil + 1e-3);
});

test("resolveMastering: presets, bypass, custom", () => {
  assert.deepEqual(resolveMastering("spotify"), { ...MASTERING_PRESETS.spotify, enabled: true });
  assert.deepEqual(resolveMastering("bypass"), { enabled: false });
  assert.deepEqual(resolveMastering(null), { enabled: false });
  assert.equal(resolveMastering({ integratedLUFS: -16 }).enabled, true);
  assert.throws(() => resolveMastering("nope" as never));
});

// ── Full pipeline with an injected identity model ───────────────────────────
// session.run returns spec_enhanced == the spec feed, so enhance() exercises
// STFT → features → chunked run → ISTFT end-to-end without the real weights.
function identityOrt(): Ort {
  class Tensor implements OrtTensor {
    data: Float32Array;
    dims: number[];
    constructor(_type: "float32", data: Float32Array, dims: number[]) {
      this.data = data;
      this.dims = dims;
    }
  }
  const session: OrtSession = {
    async run(feeds) {
      const spec = feeds.spec;
      return { spec_enhanced: new Tensor("float32", spec.data.slice(), [] ) };
    },
  };
  return { Tensor: Tensor as unknown as Ort["Tensor"], InferenceSession: { create: async () => session } };
}

test("enhance() runs the full pipeline (identity model, bypass mastering)", async () => {
  const model = new ClearModel({ ort: identityOrt(), session: await identityOrt().InferenceSession.create(new Uint8Array()), variant: "studio" as Variant, backend: "test" });
  const x = sine(440, 0.6, 0.4);
  const { audio, sampleRate, measuredLUFS } = await model.enhance(x, { mastering: "bypass" });
  assert.equal(sampleRate, SR);
  assert.equal(audio.length, x.length);
  assert.equal(measuredLUFS, null);
  const a = FFT_SIZE;
  const b = x.length - FFT_SIZE;
  let err = 0;
  for (let i = a; i < b; i++) err += (audio[i] - x[i]) ** 2;
  const nrmse = Math.sqrt(err / (b - a)) / rms(x, a, b);
  assert.ok(nrmse < 1e-2, `identity enhance NRMSE ${nrmse} too high`);
  await model.dispose();
});

test("enhance() applies mastering and reports loudness", async () => {
  const model = new ClearModel({ ort: identityOrt(), session: await identityOrt().InferenceSession.create(new Uint8Array()), variant: "studio" as Variant, backend: "test" });
  const x = sine(300, 0.8, 0.05); // quiet → mastering should raise it
  const { measuredLUFS, measuredTruePeakDBFS } = await model.enhance(x, { mastering: "applePodcasts" });
  assert.ok(measuredLUFS !== null && Number.isFinite(measuredLUFS));
  assert.equal(measuredTruePeakDBFS, MASTERING_PRESETS.applePodcasts.truePeakDBTP);
  await model.dispose();
});
