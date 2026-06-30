// Copyright (c) 2026 Desert Ant Labs
//
// Clear demo glue. Pick an audio file, decode it, run Clear, and expose an
// A/B player + WAV download.

import { load, SR, encodeWav, decodeToMono } from '@desert-ant-labs/clear';
import { paintSwarm, setSwarm } from './ds/swarm.js';

// ─────────────────────────────── helpers ─────────────────────────────────

const $ = (id) => document.getElementById(id);
function fmtTime(sec) {
  if (!Number.isFinite(sec)) return '0:00';
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}
const htmlEscape = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));

const ICON_PLAY  = `<svg viewBox="0 0 16 16" fill="currentColor" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true"><path d="M5.5 3.5 L13 8 L5.5 12.5 Z"/></svg>`;
const ICON_PAUSE = `<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="3" width="2.8" height="10" rx="1.2" fill="currentColor"/><rect x="9.2" y="3" width="2.8" height="10" rx="1.2" fill="currentColor"/></svg>`;
const START_LEAD = 0.05;

// ─────────────────────────── upload pipeline ─────────────────────────────

const HF = 'https://huggingface.co/desert-ant-labs/clear/resolve/main';
const MODEL_URLS = { studio: `${HF}/clear-studio.onnx` };
const forceWasm = new URLSearchParams(location.search).has('wasm');

const VARIANT = 'studio';
const VARIANT_LABEL = 'clear-studio';

let clear = null;
let clearVariant = null;
let enhancedObjectURL = null;
let abPlayer = null;

const setStatus = (msg, kind) => {
  $('stageLead').textContent = msg;
  $('stage').classList.toggle('err', kind === 'error');
};
const setBusy = (on) => {
  setSwarm($('loadSwarm'), on);
  $('stage').classList.toggle('busy', !!on);
};

paintSwarm($('loadSwarm').querySelector('svg'));

function enableFilePicker() { $('fileInput').disabled = false; }
function disableFilePicker() { $('fileInput').disabled = true; }

function showResults(on) {
  $('results').hidden = !on;
  $('stage').hidden = !!on;
}

function teardownPlayer() {
  if (abPlayer) { abPlayer.dispose(); abPlayer = null; }
  if (enhancedObjectURL) { URL.revokeObjectURL(enhancedObjectURL); enhancedObjectURL = null; }
}

function resetForNewModel() {
  teardownPlayer();
  showResults(false);
  $('fileInput').value = '';
}

async function loadModel() {
  if (clearVariant === VARIANT && clear) {
    setStatus(`Drop audio here, or click to pick a file`);
    enableFilePicker();
    return;
  }

  if (clear) { try { await clear.dispose(); } catch {} clear = null; clearVariant = null; }
  resetForNewModel();

  setStatus('Downloading model…');
  setBusy(true);
  $('loadBtn').disabled = true;
  disableFilePicker();

  // Single-thread WASM. Multi-threaded WASM each reserves its own
  // SharedArrayBuffer; repeated reloads exhaust the browser's per-origin
  // cap and the next page bombs with `RangeError: Out of memory`.
  const baseOpts = {
    variant: VARIANT,
    numThreads: 1,
    onDownloadProgress: (loaded, total) => {
      if (!total) return;
      const mb = (n) => (n / 1_048_576).toFixed(1);
      setStatus(`Downloading model · ${mb(loaded)} / ${mb(total)} MB`);
    },
    onPhase: (phase) => {
      if (phase === 'compiling-webgpu') setStatus('Compiling for WebGPU…');
      else if (phase === 'compiling-wasm') setStatus('Compiling for WASM…');
    },
  };

  try {
    try {
      clear = await load({ ...baseOpts, forceWasm });
    } catch (e1) {
      const msg = String(e1?.message || e1);
      if (/out of memory|no available backend/i.test(msg)) {
        setStatus('Recovering, retrying on WASM…');
        await new Promise((r) => setTimeout(r, 250));
        clear = await load({ ...baseOpts, forceWasm: true });
      } else {
        throw e1;
      }
    }
    clearVariant = VARIANT;
    setStatus('Drop audio here, or click to pick a file');
    enableFilePicker();
  } catch (e) {
    setStatus(`Failed to load model: ${e.message || e}`, 'error');
  } finally {
    setBusy(false);
    $('loadBtn').disabled = false;
  }
}

$('loadBtn').addEventListener('click', loadModel);
loadModel();

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file || !clear) return;

  teardownPlayer();

  setStatus(`Decoding ${file.name}…`);
  setBusy(true);

  let rawAudio;
  try {
    rawAudio = await decodeToMono(file);
  } catch (err) {
    setStatus(`Couldn't decode: ${err.message || err}`, 'error');
    setBusy(false);
    return;
  }

  setStatus(`Enhancing ${fmtTime(rawAudio.length / SR)} of audio…`);
  const t0 = performance.now();
  let result;
  try {
    result = await clear.enhance(rawAudio, {
      onProgress: (stage, frac) => {
        if (stage === 'inference') {
          setStatus(`Enhancing ${fmtTime(rawAudio.length / SR)} · ${Math.round(frac * 100)}%`);
        }
      },
    });
  } catch (err) {
    setStatus(`Enhance failed: ${err.message || err}`, 'error');
    setBusy(false);
    return;
  }
  const processingSec = (performance.now() - t0) / 1000;
  setBusy(false);

  abPlayer = createUploadPlayer(rawAudio, result.audio, SR);

  const wavBlob = new Blob([encodeWav(result.audio, result.sampleRate)], { type: 'audio/wav' });
  enhancedObjectURL = URL.createObjectURL(wavBlob);
  const dl = $('downloadLink');
  dl.href = enhancedObjectURL;
  const stem = file.name.replace(/\.[^.]+$/, '');
  dl.download = `${stem}_clear.wav`;

  const rt = processingSec > 0 ? result.durationSec / processingSec : 0;
  $('statDuration').textContent = fmtTime(result.durationSec);
  $('statSpeed').textContent = rt > 0 ? `${rt.toFixed(1)}×` : '…';
  $('statLUFS').textContent = result.measuredLUFS != null
    ? `${result.measuredLUFS.toFixed(1)}`
    : '…';
  $('statBackend').textContent = clear.backend;
  showResults(true);
});

$('resetBtn').addEventListener('click', () => {
  resetForNewModel();
  setStatus('Drop audio here, or click to pick a file');
  setBusy(false);
});

// Sample-aligned A/B player for the uploaded result. Two AudioBufferSources
// scheduled at the same ctx.currentTime + lead, each through its own gain;
// the slider crossfades them. Querying #player scopes the .mix-end buttons.

function createUploadPlayer(rawSamples, enhSamples, sampleRate) {
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

  const root = $('player');
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
  const mixEnds = root.querySelectorAll('.mix-row .mix-end');
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

// ─────────────── Release the WASM heap before page unload ────────────────
// Without this, repeated reloads in the same tab eventually hit a wasm
// RangeError because the previous session's memory pool isn't reclaimed
// before the new page initializes its own.
window.addEventListener('pagehide', () => {
  if (clear) { try { clear.dispose(); } catch {} clear = null; }
  if (abPlayer) { try { abPlayer.dispose(); } catch {} abPlayer = null; }
});
