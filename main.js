// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Detail Technologies B.V.
//
// Demo glue between lib/ and the UI. To integrate Clear into your own
// app, copy lib/ and ignore this file — see README for the recipe.

import { Clear, SR, encodeWav, decodeToMono } from './lib/clear.js';

const HF = 'https://huggingface.co/detail-co/clear/resolve/main';
const MODEL_URLS = {
  studio:  `${HF}/clear-studio.onnx`,
  natural: `${HF}/clear-natural.onnx`,
};
const forceWasm = new URLSearchParams(location.search).has('wasm');

const ICON_PLAY  = `<svg viewBox="0 0 16 16" fill="currentColor" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true"><path d="M5.5 3.5 L13 8 L5.5 12.5 Z"/></svg>`;
const ICON_PAUSE = `<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="3" width="2.8" height="10" rx="1.2" fill="currentColor"/><rect x="9.2" y="3" width="2.8" height="10" rx="1.2" fill="currentColor"/></svg>`;
const START_LEAD = 0.05;

const $ = (id) => document.getElementById(id);
const setStatus = (msg) => { $('status').textContent = msg; };

(async function fillSizes() {
  for (const [variant, url] of Object.entries(MODEL_URLS)) {
    try {
      const r = await fetch(url, { method: 'HEAD' });
      const len = parseInt(r.headers.get('content-length') || '0', 10);
      const span = document.querySelector(`span[data-size="${variant}"]`);
      if (span && len) span.textContent = `${(len / 1_048_576).toFixed(1)} MB`;
    } catch (e) {
      console.log(`[demo] size probe failed for ${variant}:`, e);
    }
  }
})();

let clear = null;
let clearVariant = null;
let enhancedObjectURL = null;
let player = null;

function fmtTime(sec) {
  if (!Number.isFinite(sec)) return '0:00';
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function setProgress(el, fraction, visible = true) {
  el.style.display = visible ? 'block' : 'none';
  el.value = Math.round(fraction * 100);
}

function enableFilePicker() {
  $('fileInput').disabled = false;
  $('fileLabel').setAttribute('aria-disabled', 'false');
}
function disableFilePicker() {
  $('fileInput').disabled = true;
  $('fileLabel').setAttribute('aria-disabled', 'true');
}

function teardownPlayer() {
  if (player) { player.dispose(); player = null; }
  $('player').style.display = 'none';
  if (enhancedObjectURL) { URL.revokeObjectURL(enhancedObjectURL); enhancedObjectURL = null; }
}

function resetForNewModel() {
  teardownPlayer();
  $('summary').style.display = 'none';
  $('fileInput').value = '';
  $('fileName').textContent = 'No file selected';
  setProgress($('decodeProgress'), 0, false);
  setProgress($('inferProgress'), 0, false);
}

$('loadBtn').addEventListener('click', async () => {
  const variant = document.querySelector('input[name=variant]:checked').value;
  const label = variant === 'natural' ? 'clear-natural' : 'clear-studio';

  if (clearVariant === variant && clear) {
    setStatus(`${label} ready · ${clear.backend}. Pick a file to enhance.`);
    enableFilePicker();
    return;
  }

  if (clear) { try { await clear.dispose(); } catch {} clear = null; clearVariant = null; }
  resetForNewModel();

  setStatus(`Downloading ${label} weights…`);
  setProgress($('loadProgress'), 0, true);
  $('loadBtn').disabled = true;
  document.querySelectorAll('input[name=variant]').forEach((r) => r.disabled = true);
  disableFilePicker();

  try {
    clear = await Clear.create({
      variant,
      forceWasm,
      onDownloadProgress: (loaded, total) => {
        if (!total) return;
        setProgress($('loadProgress'), loaded / total, true);
        const mb = (n) => (n / 1_048_576).toFixed(1);
        setStatus(`Downloading ${label} weights · ${mb(loaded)} / ${mb(total)} MB`);
      },
      onPhase: (phase) => {
        if (phase === 'compiling-webgpu') setStatus('Compiling for WebGPU…');
        else if (phase === 'compiling-wasm') setStatus('Compiling for WASM…');
      },
    });
    clearVariant = variant;
    setProgress($('loadProgress'), 1, false);
    setStatus(`${label} ready · ${clear.backend}. Pick a file to enhance.`);
    enableFilePicker();
  } catch (e) {
    setStatus(`Failed to load ${label}: ${e.message || e}`);
    setProgress($('loadProgress'), 0, false);
  } finally {
    $('loadBtn').disabled = false;
    document.querySelectorAll('input[name=variant]').forEach((r) => r.disabled = false);
  }
});

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file || !clear) return;

  $('fileName').textContent = file.name;
  teardownPlayer();
  $('summary').style.display = 'none';

  setStatus(`Decoding ${file.name}…`);
  setProgress($('decodeProgress'), 0, true);
  setProgress($('inferProgress'), 0, false);

  // Decode once so raw and enhanced share the exact 48 kHz mono Float32
  // — A/B alignment is sample-exact.
  let rawAudio;
  try {
    rawAudio = await decodeToMono(file);
  } catch (err) {
    setStatus(`Couldn't decode: ${err.message || err}`);
    setProgress($('decodeProgress'), 0, false);
    return;
  }
  setProgress($('decodeProgress'), 1, false);

  const mastering = document.querySelector('input[name=mastering]:checked').value;

  const t0 = performance.now();
  let result;
  try {
    result = await clear.enhance(rawAudio, {
      mastering,
      onProgress: (stage, frac) => {
        if (stage === 'inference') setProgress($('inferProgress'), frac, true);
      },
    });
  } catch (err) {
    setStatus(`Enhance failed: ${err.message || err}`);
    setProgress($('inferProgress'), 0, false);
    return;
  }
  const processingSec = (performance.now() - t0) / 1000;
  setProgress($('inferProgress'), 1, false);

  // Build the player.
  player = createABPlayer(rawAudio, result.audio, SR);
  $('player').style.display = 'flex';

  // Download link from the enhanced WAV.
  const wavBlob = encodeWav(result.audio, result.sampleRate);
  enhancedObjectURL = URL.createObjectURL(wavBlob);
  const dl = $('downloadLink');
  dl.href = enhancedObjectURL;
  const stem = file.name.replace(/\.[^.]+$/, '');
  dl.download = `${stem}_clear.wav`;

  const rt = processingSec > 0 ? result.durationSec / processingSec : 0;
  $('statDuration').textContent = fmtTime(result.durationSec);
  $('statSpeed').textContent = rt > 0 ? `${rt.toFixed(1)}×` : '—';
  $('statLUFS').textContent = result.measuredLUFS != null
    ? `${result.measuredLUFS.toFixed(1)}`
    : '—';
  $('statBackend').textContent = clear.backend;
  $('summary').style.display = 'grid';

  const lufsStr = result.measuredLUFS != null
    ? ` · target ${result.measuredLUFS.toFixed(1)} LUFS`
    : ' · no mastering';
  setStatus(
    `Enhanced ${fmtTime(result.durationSec)} in ${processingSec.toFixed(2)} s · ` +
    `${rt.toFixed(1)}× realtime · ${clear.backend}${lufsStr}`,
  );
});

// Sample-aligned A/B player: two AudioBufferSourceNodes scheduled at the
// same ctx.currentTime + lead, each through its own GainNode; the mix
// slider crossfades them in real time.
function createABPlayer(rawSamples, enhSamples, sampleRate) {
  let ctx = null;
  let bufRaw = null, bufEnh = null;
  let srcRaw = null, srcEnh = null;
  let gainRaw = null, gainEnh = null;
  let isPlaying = false;
  let playStartCtxTime = 0;
  let pausePos = 0;
  let mix = 1.0;
  let rafId = null;
  const duration = Math.max(rawSamples.length, enhSamples.length) / sampleRate;

  const playBtn = $('playBtn');
  const scrubber = $('scrubber');
  const progress = scrubber.querySelector('.progress');
  const cur = scrubber.parentElement.querySelector('.cur');
  const tot = scrubber.parentElement.querySelector('.tot');
  const slider = $('mixSlider');

  function ensureCtx() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
    bufRaw = ctx.createBuffer(1, rawSamples.length, sampleRate);
    bufRaw.copyToChannel(rawSamples, 0);
    bufEnh = ctx.createBuffer(1, enhSamples.length, sampleRate);
    bufEnh.copyToChannel(enhSamples, 0);
    gainRaw = ctx.createGain();
    gainEnh = ctx.createGain();
    gainRaw.connect(ctx.destination);
    gainEnh.connect(ctx.destination);
    applyMix();
  }

  function applyMix() {
    if (!ctx) return;
    gainRaw.gain.value = 1 - mix;
    gainEnh.gain.value = mix;
  }

  function startSources(offset) {
    const startAt = ctx.currentTime + START_LEAD;
    srcRaw = ctx.createBufferSource(); srcRaw.buffer = bufRaw;
    srcEnh = ctx.createBufferSource(); srcEnh.buffer = bufEnh;
    srcRaw.connect(gainRaw); srcEnh.connect(gainEnh);
    srcRaw.start(startAt, offset);
    srcEnh.start(startAt, offset);
    srcEnh.onended = () => {
      if (isPlaying) {
        isPlaying = false;
        pausePos = 0;
        playBtn.dataset.state = 'paused';
        playBtn.innerHTML = ICON_PLAY;
        stopRaf();
      }
    };
    playStartCtxTime = startAt;
    isPlaying = true;
    pausePos = offset;
    startRaf();
  }

  function stopSources() {
    for (const s of [srcRaw, srcEnh]) {
      if (!s) continue;
      try { s.onended = null; s.stop(); } catch {}
      try { s.disconnect(); } catch {}
    }
    srcRaw = null; srcEnh = null;
    isPlaying = false;
    stopRaf();
  }

  function currentPos() {
    if (isPlaying) {
      const elapsed = pausePos + (ctx.currentTime - playStartCtxTime);
      return Math.min(duration, Math.max(0, elapsed));
    }
    return pausePos;
  }

  function syncTimes() {
    const p = currentPos();
    cur.textContent = fmtTime(p);
    tot.textContent = fmtTime(duration);
    progress.style.width = `${Math.min(100, (p / duration) * 100)}%`;
  }

  function startRaf() {
    if (rafId) return;
    const tick = () => {
      syncTimes();
      if (isPlaying) rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }
  function stopRaf() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    syncTimes();
  }

  async function play() {
    ensureCtx();
    if (ctx.state === 'suspended') await ctx.resume();
    startSources(pausePos);
  }
  function pause() {
    const pos = currentPos();
    stopSources();
    pausePos = Math.min(duration, pos);
  }
  function seekTo(t) {
    const wasPlaying = isPlaying;
    if (wasPlaying) stopSources();
    pausePos = Math.max(0, Math.min(duration - 0.01, t));
    if (wasPlaying) startSources(pausePos);
    else syncTimes();
  }

  function paintSlider() {
    slider.style.setProperty('--p', `${slider.value}%`);
    progress.classList.toggle('accent', mix > 0.5);
  }

  // ── wire events ──
  const onPlayClick = async () => {
    if (!isPlaying) {
      await play();
      playBtn.dataset.state = 'playing';
      playBtn.innerHTML = ICON_PAUSE;
    } else {
      pause();
      playBtn.dataset.state = 'paused';
      playBtn.innerHTML = ICON_PLAY;
    }
  };
  const onScrubClick = (e) => {
    const rect = scrubber.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    seekTo(pct * duration);
  };
  const onSlider = () => {
    mix = slider.value / 100;
    applyMix();
    paintSlider();
  };
  const mixEnds = document.querySelectorAll('.mix-row .mix-end');
  const mixEndHandlers = [];
  mixEnds.forEach((btn) => {
    const h = () => {
      slider.value = Number(btn.dataset.mix);
      mix = slider.value / 100;
      applyMix();
      paintSlider();
    };
    btn.addEventListener('click', h);
    mixEndHandlers.push([btn, h]);
  });

  playBtn.dataset.state = 'paused';
  playBtn.innerHTML = ICON_PLAY;
  slider.value = 100;
  mix = 1.0;
  paintSlider();
  syncTimes();

  playBtn.addEventListener('click', onPlayClick);
  scrubber.addEventListener('click', onScrubClick);
  slider.addEventListener('input', onSlider);

  return {
    dispose() {
      stopSources();
      playBtn.removeEventListener('click', onPlayClick);
      scrubber.removeEventListener('click', onScrubClick);
      slider.removeEventListener('input', onSlider);
      for (const [btn, h] of mixEndHandlers) btn.removeEventListener('click', h);
      if (ctx) ctx.close();
    },
  };
}
