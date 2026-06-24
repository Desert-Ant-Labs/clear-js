// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Desert Ant Labs
//
// Clear demo glue. Two parallel interactive surfaces:
//
//   1. Upload pipeline (in the hero card on the right):
//      file pick, decode, run Clear, expose an A/B player + WAV download.
//   2. Live A/B clip list (the 12 curated recordings below):
//      lazy-loads pre-rendered raw / studio / natural WAVs, sample-aligned.
//
// Audio for the clip list lives in audio/<track>/<id>.wav relative to this
// page. In the deployed HF Space, that directory is committed alongside the
// bundle; locally, run scripts/fetch-audio.sh to mirror it.

import { Clear, SR, encodeWav, decodeToMono } from './lib/clear.js';
import { paintSwarm, setSwarm } from './lib/ds/swarm.js';

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

const HF = 'https://huggingface.co/detail-co/clear/resolve/main';
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
      clear = await Clear.create({ ...baseOpts, forceWasm });
    } catch (e1) {
      const msg = String(e1?.message || e1);
      if (/out of memory|no available backend/i.test(msg)) {
        setStatus('Recovering, retrying on WASM…');
        await new Promise((r) => setTimeout(r, 250));
        clear = await Clear.create({ ...baseOpts, forceWasm: true });
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

  const wavBlob = encodeWav(result.audio, result.sampleRate);
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
// the slider crossfades them. Querying #player scopes the .mix-end buttons
// so the clip-list cards below aren't accidentally wired up too.
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

// ───────────────────────── live A/B clip list ────────────────────────────

const CLIPS_BASE = 'audio';
const TRACKS = ['raw', 'studio', 'natural'];

const CLIPS = [
  { id: '02-demo_01_-_launch_day__bca8894f-7937-49de-8240-f746',
    quote: 'You import a video, or record a video, or import a video from web even.',
    challenge: 'Demo-room reverb and HVAC hum' },
  { id: '04-portable_video_podcast_setup_001__646677e5-66ec-46',
    quote: 'You record one audio track for you, one for your guest and they are just incredibly tiny.',
    challenge: 'Portable rig, room reflections' },
  { id: '06-the_pitch_03__88508b92-f423-4e65-a973-818ff4002e41',
    quote: "Subwave isn't just built for publishers, it has to be a place that people love to watch.",
    challenge: 'Pitch energy, breath, plosives' },
  { id: '08-wirelessmicreview__be99d613-e272-499d-ad28-38bea7b',
    quote: 'Cumbersome when recording a selfie video. I want to be able to replay it right away.',
    challenge: 'Wireless lav, dropouts' },
  { id: '09-the_explosion_of_software__c759ed82-38a5-471d-b493',
    quote: 'The API that was designed for a developer at another company is now consumed by an agent working for you.',
    challenge: 'Wireless mic in hard room' },
  { id: '12-foureyes-matt-haig',
    quote: "Today we're going to be talking about The Life Impossible by Matt Haig.",
    challenge: 'Clean audio, single speaker' },
  { id: '09-flower-field',
    quote: "I think that's a little bit sad. Yeah, it is. But then you can get new flowers.",
    challenge: 'Two-speaker outdoor, soft delivery' },
  { id: '07-frank-noise-test',
    quote: 'I always want to learn about these things by reading several fora or watching videos on YouTube.',
    challenge: 'Outdoor, construction noise' },
  { id: '08-randomchats-aries',
    quote: "I like that about Leo, but they're not as chill as Aries or Sagittarius.",
    challenge: 'Outdoor podcast, street noise' },
  { id: '11-livestream-outdoor',
    quote: "It's really easy to add effect captions, overlays, cut out sentences. Turn that stream into great content.",
    challenge: 'Outdoor livestream, wind, traffic' },
  { id: '11-randomchats-coachella',
    quote: 'Would you go to Coachella? I would go if I had the treatment, like the celebrities.',
    challenge: 'Two speakers, city noise' },
  { id: '12-paul-designing-on-device',
    quote: 'Closer to a podcast studio than an actual phone call.',
    challenge: 'Indoor room, wireless mic' },
];

let clipsAudioCtx = null;
let activeClip = null;

class ClipPlayer {
  constructor(clip, el) {
    this.clip = clip;
    this.el = el;
    this.buffers = {};
    this.sources = {};
    this.gains = {};
    this.activeEnhanced = 'studio';
    this.mix = 1.0;
    this.isLoading = false;
    this.isLoaded = false;
    this.isPlaying = false;
    this.startCtxTime = 0;
    this.offset = 0;
    this.rafId = null;
    this.bind();
  }

  bind() {
    this.playBtn = this.el.querySelector('.play');
    this.playBtn.innerHTML = ICON_PLAY;
    this.playBtn.addEventListener('click', () => this.toggle());

    this.slider = this.el.querySelector('.mix-slider');
    this.slider.addEventListener('input', (e) => this.setMix(Number(e.target.value) / 100));

    this.el.querySelectorAll('.pill').forEach((pill) => {
      pill.addEventListener('click', () => this.setEnhanced(pill.dataset.track));
    });
    this.el.querySelector('.mix-end[data-mix="0"]').addEventListener('click', () => {
      this.slider.value = 0; this.setMix(0);
    });
    this.el.querySelector('.mix-end[data-mix="100"]').addEventListener('click', () => {
      this.slider.value = 100; this.setMix(1);
    });

    this.scrubber = this.el.querySelector('.scrubber');
    this.scrubber.addEventListener('click', (e) => {
      if (!this.isLoaded) return;
      const rect = this.scrubber.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      this.seek(frac * this.buffers.raw.duration);
    });

    this.setMix(1);
  }

  async toggle() {
    if (this.isPlaying) { this.pause(); return; }
    if (activeClip && activeClip !== this) activeClip.pause();
    await this.play();
  }

  async load() {
    if (this.isLoaded || this.isLoading) return;
    this.isLoading = true;
    setSwarm(this.el.querySelector('.swarm'), true);
    if (!clipsAudioCtx) clipsAudioCtx = new (window.AudioContext || window.webkitAudioContext)();

    try {
      const buffers = await Promise.all(TRACKS.map(async (t) => {
        const res = await fetch(`${CLIPS_BASE}/${t}/${this.clip.id}.wav`);
        if (!res.ok) throw new Error(`fetch ${t} ${res.status}`);
        const arr = await res.arrayBuffer();
        return clipsAudioCtx.decodeAudioData(arr);
      }));
      TRACKS.forEach((t, i) => { this.buffers[t] = buffers[i]; });
      this.isLoaded = true;
      this.el.querySelector('.tot').textContent = fmtTime(this.buffers.raw.duration);
    } catch (e) {
      console.error(`[demo] failed to load ${this.clip.id}:`, e);
      this.el.querySelector('.tot').textContent = 'failed';
    } finally {
      this.isLoading = false;
      setSwarm(this.el.querySelector('.swarm'), false);
    }
  }

  async play() {
    await this.load();
    if (!this.isLoaded) return;

    activeClip = this;
    if (clipsAudioCtx.state === 'suspended') await clipsAudioCtx.resume();

    const startAt = clipsAudioCtx.currentTime + 0.05;
    TRACKS.forEach((t) => {
      const src = clipsAudioCtx.createBufferSource();
      src.buffer = this.buffers[t];
      src.loop = true;
      const gain = clipsAudioCtx.createGain();
      gain.gain.value = 0;
      src.connect(gain).connect(clipsAudioCtx.destination);
      src.start(startAt, this.offset);
      this.sources[t] = src;
      this.gains[t] = gain;
    });

    this.startCtxTime = startAt;
    this.isPlaying = true;
    this.applyMix();
    this.playBtn.innerHTML = ICON_PAUSE;
    this.tick();
  }

  pause() {
    if (!this.isPlaying) return;
    const elapsed = clipsAudioCtx.currentTime - this.startCtxTime;
    this.offset = (this.offset + elapsed) % this.buffers.raw.duration;
    TRACKS.forEach((t) => {
      try { this.sources[t]?.stop(); } catch {}
    });
    this.sources = {};
    this.gains = {};
    this.isPlaying = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.playBtn.innerHTML = ICON_PLAY;
  }

  seek(seconds) {
    const wasPlaying = this.isPlaying;
    if (wasPlaying) this.pause();
    this.offset = Math.max(0, seconds);
    this.updateScrubber(this.offset);
    if (wasPlaying) this.play();
  }

  setMix(value) {
    this.mix = value;
    this.applyMix();
    this.slider.style.setProperty('--p', `${value * 100}%`);
  }

  setEnhanced(track) {
    this.activeEnhanced = track;
    this.el.querySelectorAll('.pill').forEach((p) => {
      p.classList.toggle('active', p.dataset.track === track);
    });
    this.applyMix();
  }

  applyMix() {
    if (!this.isPlaying) return;
    this.gains.raw.gain.value = 1 - this.mix;
    this.gains.studio.gain.value = this.activeEnhanced === 'studio' ? this.mix : 0;
    this.gains.natural.gain.value = this.activeEnhanced === 'natural' ? this.mix : 0;
  }

  updateScrubber(playhead) {
    const dur = this.buffers.raw?.duration || 0;
    if (!dur) return;
    this.el.querySelector('.cur').textContent = fmtTime(playhead);
    this.el.querySelector('.progress').style.width = `${(playhead / dur) * 100}%`;
  }

  tick() {
    if (!this.isPlaying) return;
    const elapsed = clipsAudioCtx.currentTime - this.startCtxTime;
    const dur = this.buffers.raw.duration;
    const playhead = (this.offset + elapsed) % dur;
    this.updateScrubber(playhead);
    this.rafId = requestAnimationFrame(() => this.tick());
  }
}

function buildClipCard(clip, n) {
  const card = document.createElement('article');
  card.className = 'clip';
  card.innerHTML = `
    <div class="clip-index"><span class="num">${String(n).padStart(2, '0')}</span> <span class="sep">·</span> <span class="challenge">${htmlEscape(clip.challenge)}</span></div>
    <div class="controls-row">
      <div class="track-pills">
        <button class="pill active" data-track="studio">Studio</button>
        <button class="pill" data-track="natural">Natural</button>
      </div>
      <div class="mix-row">
        <button class="mix-end" data-mix="0">Raw</button>
        <input type="range" class="mix-slider" min="0" max="100" value="100" aria-label="Raw to enhanced mix" />
        <button class="mix-end accent" data-mix="100">Enhanced</button>
      </div>
    </div>
    <p class="quote">${clip.quote ? `&ldquo;${htmlEscape(clip.quote)}&rdquo;` : ''}</p>
    <div class="transport">
      <button class="play play-lg" aria-label="Play / pause"></button>
      <div class="track-area">
        <div class="scrubber"><div class="progress accent"></div></div>
        <div class="time-row"><span class="cur">0:00</span><span class="tot">…</span></div>
      </div>
      <span class="swarm clip-swarm" aria-hidden="true"><svg viewBox="0 0 24 24"></svg></span>
    </div>
  `;
  paintSwarm(card.querySelector('.swarm svg'));
  new ClipPlayer(clip, card);
  return card;
}

function renderClips() {
  const list = $('clip-list');
  if (!list) { console.error('[demo] missing #clip-list'); return; }
  CLIPS.forEach((clip, i) => {
    try {
      list.appendChild(buildClipCard(clip, i + 1));
    } catch (e) {
      console.error(`[demo] failed to build clip ${clip.id}:`, e);
    }
  });
}

try { renderClips(); }
catch (e) {
  console.error('[demo] renderClips failed:', e);
  const list = $('clip-list');
  if (list) {
    list.innerHTML = `<p class="meta" style="text-align:center; padding: 2rem 0; color: var(--danger);">Failed to render clips: ${e.message}</p>`;
  }
}

// ─────────────── Release the WASM heap before page unload ────────────────
// Without this, repeated reloads in the same tab eventually hit a wasm
// RangeError because the previous session's memory pool isn't reclaimed
// before the new page initializes its own.
window.addEventListener('pagehide', () => {
  if (clear) { try { clear.dispose(); } catch {} clear = null; }
  if (abPlayer) { try { abPlayer.dispose(); } catch {} abPlayer = null; }
  if (clipsAudioCtx) { try { clipsAudioCtx.close(); } catch {} clipsAudioCtx = null; }
});
