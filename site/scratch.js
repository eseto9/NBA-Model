/* Late scratches on the site: tap a player in a game's rotation to mark him out (or back in).
   The game is re-priced instantly with the default model's own lineup math and coefficients:
   the lineup features are recomputed for both teams and the change in the logistic index is
   added to the published log-odds. Saved in this browser for the current slate. */
"use strict";

const SCRATCH_KEY = "tipoff-scratches";
const LINEUP_KEYS = { pie: "d_lu_pie", net: "d_lu_net", gmsc: "d_lu_gmsc", star: "d_lu_star", min: "d_lu_min", rapm: "d_lu_rapm" };
const RAPM_REPLACEMENT = -2;      // rapm.REPLACEMENT: players with no lineup data yet

function loadScratches() {
  try {
    const s = JSON.parse(localStorage.getItem(SCRATCH_KEY));
    return s && s.date === D.slate.date ? s.players : {};
  } catch (e) { return {}; }
}
function saveScratches() {
  try { localStorage.setItem(SCRATCH_KEY, JSON.stringify({ date: D.slate.date, players: state.scratch })); } catch (e) { /* private mode */ }
}
const scratchKey = (side, p) => `${side.abbr}|${p.name}`;

/* players.lineup_strength in JS: top 10 by expected minutes x availability, minutes-weighted means */
function lineupOf(side, useScratches) {
  const rows = side.rotation.map((p) => {
    const o = useScratches ? state.scratch[scratchKey(side, p)] : null;
    const avail = o === "out" ? 0 : o === "in" ? 1 : p.status === "Out" ? 0 : p.status === "Day-To-Day" ? 0.5 : 1;
    return { ...p, rapm: p.rapm ?? RAPM_REPLACEMENT, w: p.min * avail };
  }).sort((a, b) => b.w - a.w).slice(0, 10);
  const W = rows.reduce((s, p) => s + p.w, 0) || 1;
  const avg = (k) => rows.reduce((s, p) => s + p.w * p[k], 0) / W;
  const stars = rows.filter((p) => p.w >= 20).map((p) => p.gmsc);
  return { pie: avg("pie"), net: avg("net"), gmsc: avg("gmsc"), rapm: avg("rapm"), star: stars.length ? Math.max(...stars) : avg("gmsc"), min: W };
}

function applyScratches() {
  const coef = Object.fromEntries(D.meta.model.features.map((f) => [f.name, f]));
  for (const g of D.slate.games) {
    if (g.p_base == null) g.p_base = g.p_home;
    const [h0, a0, h1, a1] = [lineupOf(g.home, false), lineupOf(g.away, false), lineupOf(g.home, true), lineupOf(g.away, true)];
    let dz = 0;
    for (const [k, f] of Object.entries(LINEUP_KEYS)) {
      const c = coef[f];
      if (c) dz += (c.coef * ((h1[k] - a1[k]) - (h0[k] - a0[k]))) / c.sd;
    }
    const z = Math.log(g.p_base / (1 - g.p_base)) + dz;
    g.p_home = 1 / (1 + Math.exp(-z));
    g.pick = g.p_home >= 0.5 ? g.home.abbr : g.away.abbr;
    g.scratched = [g.away, g.home].flatMap((s) => s.rotation.filter((p) => state.scratch[scratchKey(s, p)]).map((p) => ({ side: s, p, how: state.scratch[scratchKey(s, p)] })));
  }
}

function scratchNote(g) {
  if (!g.scratched || !g.scratched.length) return "";
  return `<div class="scratch-note"><span>Your changes: ${g.scratched.map((x) => `<b>${esc(x.p.name)}</b> ${x.how === "out" ? "out" : "playing"}`).join(", ")}
    · ${esc(g.home.abbr)} ${pct(g.p_base)} → <b>${pct(g.p_home)}</b></span><button class="chip" data-unscratch="${esc(g.id)}">Reset</button></div>`;
}

document.addEventListener("click", (e) => {
  const t = e.target;
  const pl = t.closest("[data-scratch]");
  if (pl) {
    const k = pl.dataset.scratch, cur = state.scratch[k], base = pl.dataset.status;
    // healthy: normal -> out -> normal; listed out or day-to-day: normal -> playing -> out -> normal
    const next = base === "Healthy" ? (cur ? null : "out") : !cur ? "in" : cur === "in" ? "out" : null;
    if (next) state.scratch[k] = next; else delete state.scratch[k];
    saveScratches();
    applyScratches();
    render();
    return;
  }
  const un = t.closest("[data-unscratch]");
  if (un) {
    const g = D.slate.games.find((x) => x.id === un.dataset.unscratch);
    for (const s of [g.away, g.home]) for (const p of s.rotation) delete state.scratch[scratchKey(s, p)];
    saveScratches();
    applyScratches();
    render();
  }
}, true);
