//
// Minimal mono WAV codec. `encodeWav` writes 16-bit PCM bytes (wrap in a
// Blob for browser download); `decodeWav` reads PCM int16/24/32 and
// float32 mono/multi-channel WAVs (downmixing to mono) for the Node path.

/**
 * Encode mono PCM (`[-1, 1]`) to 16-bit WAV bytes.
 *
 * In the browser, wrap for download: `new Blob([encodeWav(pcm, 48000)], { type: "audio/wav" })`.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const numSamples = samples.length;
  const byteRate = sampleRate * 2;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);
  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    off += 2;
  }
  return new Uint8Array(buffer);
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

export interface DecodedWav {
  /** Mono PCM in `[-1, 1]`. */
  samples: Float32Array;
  sampleRate: number;
}

/**
 * Decode a PCM/float WAV (the Node input path — no Web Audio). Supports
 * 16/24/32-bit integer and 32-bit float, any channel count (downmixed to mono).
 * For compressed formats (mp3, m4a, …) decode upstream and pass a Float32Array.
 */
export function decodeWav(input: ArrayBuffer | ArrayBufferView): DecodedWav {
  const bytes =
    input instanceof ArrayBuffer
      ? new Uint8Array(input)
      : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (str(view, 0, 4) !== "RIFF" || str(view, 8, 4) !== "WAVE") {
    throw new Error("clear: not a RIFF/WAVE file");
  }

  let fmtFound = false;
  let audioFormat = 1;
  let channels = 1;
  let sampleRate = 48_000;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataSize = 0;

  let pos = 12;
  while (pos + 8 <= view.byteLength) {
    const id = str(view, pos, 4);
    const size = view.getUint32(pos + 4, true);
    const body = pos + 8;
    if (id === "fmt ") {
      audioFormat = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bitsPerSample = view.getUint16(body + 14, true);
      fmtFound = true;
    } else if (id === "data") {
      dataOffset = body;
      dataSize = Math.min(size, view.byteLength - body);
    }
    pos = body + size + (size & 1); // chunks are word-aligned
  }
  if (!fmtFound || dataOffset < 0) throw new Error("clear: WAV missing fmt/data chunk");

  const isFloat = audioFormat === 3;
  const bytesPerSample = bitsPerSample >> 3;
  const frameSize = bytesPerSample * channels;
  const numFrames = Math.floor(dataSize / frameSize);
  const mono = new Float32Array(numFrames);
  const invCh = 1 / channels;

  for (let i = 0; i < numFrames; i++) {
    let acc = 0;
    const frame = dataOffset + i * frameSize;
    for (let ch = 0; ch < channels; ch++) {
      acc += readSample(view, frame + ch * bytesPerSample, bitsPerSample, isFloat);
    }
    mono[i] = acc * invCh;
  }
  return { samples: mono, sampleRate };
}

function readSample(view: DataView, off: number, bits: number, isFloat: boolean): number {
  if (isFloat) {
    return bits === 64 ? view.getFloat64(off, true) : view.getFloat32(off, true);
  }
  switch (bits) {
    case 8:
      return (view.getUint8(off) - 128) / 128;
    case 16:
      return view.getInt16(off, true) / 0x8000;
    case 24: {
      const b0 = view.getUint8(off);
      const b1 = view.getUint8(off + 1);
      const b2 = view.getUint8(off + 2);
      let v = b0 | (b1 << 8) | (b2 << 16);
      if (v & 0x800000) v |= ~0xffffff; // sign-extend
      return v / 0x800000;
    }
    case 32:
      return view.getInt32(off, true) / 0x80000000;
    default:
      throw new Error(`clear: unsupported WAV bit depth ${bits}`);
  }
}

function str(view: DataView, off: number, len: number): string {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
  return s;
}
