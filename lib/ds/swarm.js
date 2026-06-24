// Desert Ant Labs swarm loader.
//
// A circular swarm of obsidian dots converging on a cobalt core. Used
// across product demos as the loading indicator (the brand's "ants
// navigating by skylight" motif, animated).
//
// Pair with tokens/swarm.css. Usage:
//   <span class="swarm" id="myLoader" aria-hidden="true">
//     <svg viewBox="0 0 24 24"></svg>
//   </span>
//
//   import { paintSwarm, setSwarm } from './lib/ds/swarm.js';
//   paintSwarm(document.querySelector('#myLoader svg'));
//   setSwarm(document.getElementById('myLoader'), true);   // show
//   setSwarm(document.getElementById('myLoader'), false);  // hide
//
// paintSwarm is idempotent (safe to call multiple times on the same svg).

export function paintSwarm(svg) {
  if (!svg || svg.dataset.painted) return;
  const C = 12, R = 8, r = 1.5, count = 8, speed = 1.6;
  const rnd = (i, k) => {
    const v = Math.sin((i + 1) * 12.9898 + k * 78.233) * 43758.5453;
    return v - Math.floor(v);
  };
  let style = '', dots = '';
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2 - Math.PI / 2;
    const x = (C + R * Math.cos(a)).toFixed(2);
    const y = (C + R * Math.sin(a)).toFixed(2);
    const inwardX = C - x, inwardY = C - y;
    const perpX = -(C - y), perpY = C - x;
    const pull = 0.45 + rnd(i, 1) * 0.6;
    const wob = (rnd(i, 2) - 0.5) * 0.5;
    const tx = (inwardX * pull + perpX * wob).toFixed(2);
    const ty = (inwardY * pull + perpY * wob).toFixed(2);
    const mid = (0.45 + rnd(i, 3) * 0.3).toFixed(2);
    const dur = (speed * (0.74 + rnd(i, 4) * 0.62)).toFixed(2);
    const delay = (-rnd(i, 5) * speed * 1.4).toFixed(2);
    dots += `<circle class="dot d${i}" cx="${x}" cy="${y}" r="${r}"></circle>`;
    style += `@keyframes sw${i}{0%{transform:translate(0,0) scale(1);opacity:1}50%{transform:translate(${tx}px,${ty}px) scale(${mid});opacity:.45}100%{transform:translate(0,0) scale(1);opacity:1}}.d${i}{animation:sw${i} ${dur}s cubic-bezier(.45,0,.55,1) ${delay}s infinite}`;
  }
  svg.innerHTML = `<g>${dots}</g><circle class="core" cx="${C}" cy="${C}" r="1.7"></circle><style>${style}</style>`;
  svg.dataset.painted = '1';
}

export function setSwarm(el, on) {
  if (el) el.classList.toggle('visible', !!on);
}
