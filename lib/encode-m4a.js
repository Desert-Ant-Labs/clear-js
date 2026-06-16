// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// Mono Float32 PCM → AAC-LC in MP4 (.m4a). Browser-side via mediabunny +
// WebCodecs (no MediaRecorder / no AudioContext playback).

import {
  Output,
  Mp4OutputFormat,
  BufferTarget,
  AudioBufferSource,
  QUALITY_HIGH,
} from './vendor/mediabunny.min.mjs';

/**
 * @param {Float32Array} samples mono PCM
 * @param {number} sampleRate
 * @returns {Promise<Blob>} audio/mp4 blob ready for download
 */
export async function encodeM4A(samples, sampleRate) {
  const output = new Output({
    format: new Mp4OutputFormat(),
    target: new BufferTarget(),
  });
  const source = new AudioBufferSource({
    codec: 'aac',
    bitrate: QUALITY_HIGH,
  });
  output.addAudioTrack(source);
  await output.start();

  const audioBuf = new AudioBuffer({
    length: samples.length,
    sampleRate,
    numberOfChannels: 1,
  });
  audioBuf.copyToChannel(samples, 0);
  await source.add(audioBuf);
  source.close();
  await output.finalize();

  return new Blob([output.target.buffer], { type: 'audio/mp4' });
}
