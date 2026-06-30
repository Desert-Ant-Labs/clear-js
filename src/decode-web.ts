//
// Decode any browser-supported audio file to mono Float32 at 48 kHz, using
// an OfflineAudioContext for resampling so the math is the browser's native
// path, not a linear interpolation.

import { SR } from "./model.js";

// Web Audio globals — present in browsers and audio Workers, not in the type
// surface of a NodeNext build. Declared loosely here.
declare const window: { AudioContext?: AudioContextCtor; webkitAudioContext?: AudioContextCtor } | undefined;
declare const AudioContext: AudioContextCtor | undefined;
declare const OfflineAudioContext: OfflineAudioContextCtor;

interface AudioBufferLike {
  length: number;
  numberOfChannels: number;
  sampleRate: number;
  getChannelData(channel: number): Float32Array;
  copyToChannel(source: Float32Array, channel: number): void;
}
interface AudioContextLike {
  decodeAudioData(data: ArrayBuffer): Promise<AudioBufferLike>;
  close(): Promise<void>;
}
type AudioContextCtor = new () => AudioContextLike;
interface OfflineAudioContextLike {
  createBuffer(channels: number, length: number, sampleRate: number): AudioBufferLike;
  createBufferSource(): { buffer: AudioBufferLike | null; connect(d: unknown): void; start(t: number): void };
  destination: unknown;
  startRendering(): Promise<AudioBufferLike>;
}
type OfflineAudioContextCtor = new (channels: number, length: number, sampleRate: number) => OfflineAudioContextLike;

/** Decode a `File`/`Blob`/`ArrayBuffer` to mono PCM at `sampleRate` (default 48 kHz). */
export async function decodeToMono(
  input: Blob | ArrayBuffer,
  sampleRate: number = SR,
): Promise<Float32Array> {
  const buf = input instanceof ArrayBuffer ? input : await input.arrayBuffer();
  const Ctor = (typeof AudioContext !== "undefined" && AudioContext) || window?.AudioContext || window?.webkitAudioContext;
  if (!Ctor) throw new Error("clear: Web Audio API unavailable (use the Node entry, or decode to a Float32Array first)");

  const tmpCtx = new Ctor();
  let decoded: AudioBufferLike;
  try {
    decoded = await tmpCtx.decodeAudioData(buf.slice(0));
  } finally {
    await tmpCtx.close();
  }
  const mono = mixToMono(decoded);
  if (decoded.sampleRate === sampleRate) return mono;
  return resampleOffline(mono, decoded.sampleRate, sampleRate);
}

function mixToMono(audioBuffer: AudioBufferLike): Float32Array {
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

async function resampleOffline(mono: Float32Array, srIn: number, srOut: number): Promise<Float32Array> {
  const outLen = Math.ceil((mono.length * srOut) / srIn);
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
