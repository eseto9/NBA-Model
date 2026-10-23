/* Tipoff: renders the model's daily JSON (data/*.json) into an interactive site. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (p, d = 0) => (p * 100).toFixed(d) + "%";
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const D = {};               // loaded data
const state = { filter: "all", sort: "tip", open: null, hlTeam: null, pSort: "gmsc", pTeam: "", pQuery: "", pLimit: 60 };

/* ---------------------------------------------------------------- data */

async function load() {
  const get = async (f, v) => {
    const r = await fetch(`data/${f}?v=${encodeURIComponent(v)}`);
    if (!r.ok) throw new Error(`${f}: ${r.status}`);
    return r.json();
  };
  D.meta = await get("meta.json", Date.now());
  const v = D.meta.generated_at;
  [D.slate, D.teams, D.players, D.record, D.backtest] = await Promise.all(
    ["slate.json", "teams.json", "players.json", "record.json", "backtest.json"].map((f) => get(f, v)));
  D.teamBy = Object.fromEntries(D.teams.teams.map((t) => [t.abbr, t]));
}

/* ---------------------------------------------------------------- helpers */

function etDate(s) {                       // "2026-10-23T19:30" US Eastern wall time -> Date
  if (!s) return null;
  const [d, t = "00:00"] = s.split("T");
  const [Y, M, Dd] = d.split("-").map(Number);
  const [h, mi] = t.split(":").map(Number);
  const guess = Date.UTC(Y, M - 1, Dd, h, mi);
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hourCycle: "h23", year: "numeric",
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).formatToParts(new Date(guess));
  const g = (k) => +parts.find((p) => p.type === k).value;
  return new Date(guess + (guess - Date.UTC(g("year"), g("month") - 1, g("day"), g("hour"), g("minute"))));
}
function tipLabel(s) {
  if (!s || s.endsWith("00:00")) return "TBD";
  let [h, m] = s.split("T")[1].split(":").map(Number);
  const ap = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  return `${h}:${String(m).padStart(2, "0")} ${ap} ET`;
}
function dayLabel(iso, opts = { weekday: "long", month: "long", day: "numeric" }) {
  const [Y, M, Dd] = iso.split("-").map(Number);
  return new Date(Y, M - 1, Dd).toLocaleDateString("en-US", opts);
}
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function colorsClash(a, b) {
  const [x, y] = [hexRgb(a), hexRgb(b)];
  return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) < 110;
}
const surname = (name) => name.replace(/\s+(Jr\.?|Sr\.?|II|III|IV)$/, "").split(" ").slice(-1)[0];
const badge = (abbr, color) => `<span class="badge" style="background:${esc(color)}">${esc(abbr)}</span>`;
const confidence = (g) => Math.max(g.p_home, 1 - g.p_home);
function ratingColor(gmsc) {                // sequential: light -> dark orange by player rating
  const t = clamp((gmsc - 6) / 16, 0, 1);
  return `color-mix(in oklab, var(--orange) ${Math.round(25 + t * 75)}%, var(--card-2))`;
}

/* tooltip */
const tipEl = () => $("#tip");
function showTip(html, e) {
  const t = tipEl();
  t.innerHTML = html;
  t.classList.add("on");
  const r = t.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  t.style.left = x + "px";
  t.style.top = y + "px";
}
const hideTip = () => tipEl().classList.remove("on");

/* animate widths / counters once visible */
const io = new IntersectionObserver((entries) => {
  for (const en of entries) {
    if (!en.isIntersecting) continue;
    const el = en.target;
    io.unobserve(el);
    if (el.dataset.w) el.style.width = el.dataset.w;
    if (el.dataset.flex) el.style.flexBasis = el.dataset.flex;
    if (el.dataset.count) countUp(el);
    if (el.dataset.dash) el.style.strokeDashoffset = el.dataset.dash;
  }
}, { threshold: 0.1 });
function animateIn(root) { $$("[data-w],[data-flex],[data-count],[data-dash]", root).forEach((el) => io.observe(el)); }
function countUp(el) {
  const end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), suf = el.dataset.suf || "";
  const t0 = performance.now(), dur = 1100;
  const step = (t) => {
    const k = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = (end * e).toFixed(dec) + suf;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------------------------------------------------------- slate */

function gameTags(g) {
  const c = confidence(g), tags = [];
  if (c >= 0.75) tags.push(`<span class="tag hot">Lock</span>`);
  else if (c < 0.56) tags.push(`<span class="tag toss">Toss-up</span>`);
  if ((g.p_home - 0.5) * (g.p_elo - 0.5) < 0) tags.push(`<span class="tag">Model ≠ Elo</span>`);
  if (g.neutral) tags.push(`<span class="tag">Neutral site</span>`);
  return tags.join(" ");
}

function outsLine(g) {
  const outs = [];
  for (const side of [g.away, g.home]) {
    for (const p of side.rotation) {
      if (p.status !== "Healthy" && p.min >= 12) {
        const dtd = p.status === "Day-To-Day";
        outs.push(`<span class="out ${dtd ? "dtd" : ""}" title="${esc(p.name)}: ${esc(p.status)}">${esc(side.abbr)} ${esc(surname(p.name))}${dtd ? " ?" : ""}</span>`);
      }
    }
  }
  return outs.length ? outs.join("") : `<span>Full-strength rotations</span>`;
}

function teamBlock(side, cls, picked) {
  const t = D.teamBy[side.abbr];
  const rec = t ? `${t.w}-${t.l} · #${t.rank}` : "";
  return `<div class="team ${cls} ${picked ? "picked" : ""}">${badge(side.abbr, side.color)}
    <div style="min-width:0"><div class="team-name">${esc(side.name)}</div><div class="team-sub">${rec}</div></div></div>`;
}

function gameCard(g) {
  const pa = 1 - g.p_home, open = state.open === g.id;
  const awayColor = colorsClash(g.away.color, g.home.color) ? "#8a857b" : g.away.color;
  return `<article class="card game ${open ? "open" : ""}" data-game="${g.id}">
    <div class="game-top"><span class="mono muted">${tipLabel(g.tip)}</span><span>${gameTags(g)}</span></div>
    <div class="matchup">${teamBlock(g.away, "away", g.pick === g.away.abbr)}<span class="at">@</span>${teamBlock(g.home, "home", g.pick === g.home.abbr)}</div>
    <div class="tug">
      <div class="tug-bar" role="img" aria-label="${esc(g.away.abbr)} ${pct(pa)}, ${esc(g.home.abbr)} ${pct(g.p_home)}">
        <i style="background:${esc(awayColor)}" data-flex="calc(${pct(pa, 1)} - 1px)"></i><i style="background:${esc(g.home.color)}" data-flex="calc(${pct(g.p_home, 1)} - 1px)"></i>
        <span class="tug-mid"></span></div>
      <div class="tug-nums num"><span class="${pa < 0.5 ? "lose" : ""}">${pct(pa)}</span><span class="${g.p_home < 0.5 ? "lose" : ""}">${pct(g.p_home)}</span></div>
    </div>
    <div class="outs">${outsLine(g)}</div>
    <span class="expand-hint">details →</span>
    ${open ? `<button class="close" aria-label="Close">✕</button>${gameDetail(g)}` : ""}
  </article>`;
}

function waterfall(g) {
  const W = 560, rowH = 34, L = 112, R = 24, top = 26;
  const rows = [...g.steps, { label: "Final", final: true, to: g.p_home }];
  const H = top + rows.length * rowH + 8;
  // symmetric window around 50/50, zoomed to this game's range
  const span = clamp(Math.max(...g.steps.flatMap((r) => [r.from, r.to]).map((v) => Math.abs(v - 0.5))) + 0.06, 0.15, 0.5);
  const x = (p) => L + ((p - 0.5 + span) / (2 * span)) * (W - L - R - 60) + 30;
  const tick = Math.round((span / 2) * 100) / 100;
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="How each factor moves the home win probability">`;
  for (const p of [0.5 - tick, 0.5, 0.5 + tick]) {
    s += `<line class="grid" x1="${x(p)}" x2="${x(p)}" y1="${top - 6}" y2="${H - 6}" ${p === 0.5 ? 'stroke-dasharray="3 3" style="stroke:var(--line)"' : ""}/>`;
  }
  s += `<text x="${x(0.5 - tick)}" y="12" text-anchor="middle">${esc(g.away.abbr)} ${pct(0.5 + tick)}</text><text x="${x(0.5)}" y="12" text-anchor="middle">50/50</text><text x="${x(0.5 + tick)}" y="12" text-anchor="middle">${esc(g.home.abbr)} ${pct(0.5 + tick)}</text>`;
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    s += `<text x="0" y="${y + rowH / 2 + 4}" class="lbl-ink" style="font-family:var(--font);font-size:12.5px">${esc(r.label)}</text>`;
    if (r.final) {
      const fav = g.p_home >= 0.5 ? g.home.abbr : g.away.abbr;
      s += `<circle class="pop" style="animation-delay:${0.15 * i}s" cx="${x(r.to)}" cy="${y + rowH / 2}" r="7" fill="var(--ink)"/>`;
      s += `<text x="${x(r.to) + 12}" y="${y + rowH / 2 + 4}" class="lbl-ink" style="font-weight:600">${esc(fav)} ${pct(confidence(g))}</text>`;
      return;
    }
    const a = x(r.from), b = x(r.to), up = r.to >= r.from;
    const w = Math.max(Math.abs(b - a), 2);
    const delta = (r.to - r.from) * 100;
    s += `<rect class="grow" style="animation-delay:${0.15 * i}s;transform-origin:${up ? "left" : "right"} center" x="${Math.min(a, b)}" y="${y + 9}" width="${w}" height="${rowH - 18}" rx="4" fill="${up ? "var(--orange)" : "var(--slate)"}" data-tip="${esc(r.label)}: ${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts toward ${esc(up ? g.home.abbr : g.away.abbr)}"/>`;
    if (Math.abs(delta) >= 0.05) {
      const tx = up ? Math.max(a, b) + 6 : Math.min(a, b) - 6;
      s += `<text x="${tx}" y="${y + rowH / 2 + 4}" text-anchor="${up ? "start" : "end"}">${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)}</text>`;
    }
    if (i < rows.length - 1) s += `<line x1="${b}" x2="${b}" y1="${y + rowH - 9}" y2="${y + rowH + 9}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 2"/>`;
  });
  return s + "</svg>";
}

const METRICS = [
  { k: "elo", label: "Elo", lo: 1300, hi: 1800, f: (v) => v.toFixed(0) },
  { k: "form_net", label: "Net rtg", lo: -15, hi: 15, f: (v) => (v > 0 ? "+" : "") + v.toFixed(1) },
  { k: "form_ortg", label: "Off rtg", lo: 100, hi: 125, f: (v) => v.toFixed(1) },
  { k: "form_drtg", label: "Def rtg", lo: 125, hi: 100, f: (v) => v.toFixed(1) },
  { k: "lu_gmsc", label: "Lineup", lo: 8, hi: 17, f: (v) => v.toFixed(1), get: (s) => s.lineup.gmsc },
  { k: "lu_star", label: "Best player", lo: 10, hi: 26, f: (v) => v.toFixed(1), get: (s) => s.lineup.star },
  { k: "rest", label: "Rest days", lo: 0, hi: 4, f: (v) => (v >= 5 ? "5+" : String(v)) },
];

function compare(g) {
  return `<div class="compare">${METRICS.map((m) => {
    const get = m.get || ((s) => s[m.k]);
    const [a, h] = [get(g.away), get(g.home)];
    if (a == null || h == null) return "";
    const w = (v) => pct(clamp((v - m.lo) / (m.hi - m.lo), 0.04, 1), 1);
    return `<div class="cmp-row">
      <div class="cmp-side"><span class="cmp-val num">${m.f(a)}</span><div class="cmp-bar l"><i data-w="${w(a)}"></i></div></div>
      <div class="lbl">${m.label}</div>
      <div class="cmp-side r"><div class="cmp-bar r"><i data-w="${w(h)}"></i></div><span class="cmp-val num">${m.f(h)}</span></div></div>`;
  }).join("")}
  <div class="legend"><span><i style="background:var(--slate)"></i>${esc(g.away.abbr)} (away)</span><span><i style="background:var(--orange)"></i>${esc(g.home.abbr)} (home)</span><span>longer = better</span></div></div>`;
}

function rotationCol(side) {
  return `<div><div class="rot-head">${badge(side.abbr, side.color)}${esc(side.name)}</div>
    ${side.rotation.map((p) => `<div class="pl ${esc(p.status)}" data-tip="<b>${esc(p.name)}</b><br>${esc(p.status)} · ${p.min} exp. min<br>Game Score/36 ${p.gmsc} · PIE ${pct(p.pie, 1)} · on-court net ${p.net > 0 ? "+" : ""}${p.net}">
      <span class="pl-name">${esc(p.name)}</span>
      <div class="pl-bar"><i style="background:${ratingColor(p.gmsc)}" data-w="${pct(clamp(p.min / 38, 0.05, 1))}"></i></div></div>`).join("")}</div>`;
}

function gameDetail(g) {
  return `<div class="detail">
    <div><h4>Why the model leans ${esc(g.pick)}</h4>${waterfall(g)}
      <h4 style="margin-top:22px">Tale of the tape</h4>${compare(g)}</div>
    <div><h4>Projected rotations</h4><div class="rot">${rotationCol(g.away)}${rotationCol(g.home)}</div>
      <div class="legend"><span><i style="background:var(--orange)"></i>bar = expected minutes, darker = better rating</span><span><i style="background:repeating-linear-gradient(135deg,var(--muted) 0 2px,transparent 2px 5px)"></i>out</span><span><i style="background:repeating-linear-gradient(135deg,var(--orange) 0 2px,transparent 2px 5px)"></i>day-to-day</span></div>
      <p class="mono muted" style="margin-top:16px">Elo alone: ${esc(g.home.abbr)} ${pct(g.p_elo)} · model: ${esc(g.home.abbr)} ${pct(g.p_home)}</p></div>
  </div>`;
}

function filteredGames() {
  let gs = [...D.slate.games];
  if (state.filter === "locks") gs = gs.filter((g) => confidence(g) >= 0.7);
  if (state.filter === "toss") gs = gs.filter((g) => confidence(g) < 0.6);
  if (state.sort === "conf") gs.sort((a, b) => confidence(b) - confidence(a));
  return gs;
}

function lockCard(games) {
  if (!games.length) return `<div class="lock"><div class="label">No games scheduled</div><p>Check back on the next game day.</p></div>`;
  const g = games.reduce((a, b) => (confidence(b) > confidence(a) ? b : a));
  const c = confidence(g), side = g.p_home >= 0.5 ? g.home : g.away, opp = side === g.home ? g.away : g.home;
  const R = 56, C = 2 * Math.PI * R;
  const first = games.map((x) => x.tip).filter(Boolean).sort()[0];
  const toss = games.reduce((a, b) => (confidence(b) < confidence(a) ? b : a));
  return `<div class="lock">
    <div class="label">Strongest pick · ${dayLabel(D.slate.date, { weekday: "short", month: "short", day: "numeric" })}</div>
    <div class="lock-body">
      <div class="gauge"><svg viewBox="0 0 132 132" width="132" height="132">
        <circle cx="66" cy="66" r="${R}" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="11"/>
        <circle class="arc" cx="66" cy="66" r="${R}" fill="none" stroke="var(--orange)" stroke-width="11" stroke-linecap="round"
          stroke-dasharray="${C}" stroke-dashoffset="${C}" data-dash="${C * (1 - c)}"/></svg>
        <div class="gauge-val"><div><b data-count="${(c * 100).toFixed(0)}" data-suf="%">0%</b><small>to win</small></div></div></div>
      <div><div class="lock-team">${esc(side.name)}</div>
        <div class="lock-sub">${side === g.home ? "vs" : "at"} ${esc(opp.name)} · ${tipLabel(g.tip)}</div></div>
    </div>
    <div class="lock-row"><span>Closest call</span><a href="#/game/${toss.id}">${esc(toss.away.abbr)} @ ${esc(toss.home.abbr)} · ${pct(confidence(toss))} ${esc(toss.pick)}</a></div>
    <div class="lock-row" style="border:0;padding-top:8px"><span>First tip</span><span class="countdown" data-tip-at="${esc(first || "")}">${tipLabel(first)}</span></div>
  </div>`;
}

function lastResults() {
  const graded = D.record.picks.filter((p) => p.score);
  if (!graded.length) {
    return `<div class="empty" style="padding:24px 8px">The live record starts with tonight's games.<br>Last season's backtest: <b>${pct(D.backtest.accuracy, 1)}</b> correct (${D.backtest.correct}/${D.backtest.n}).</div>`;
  }
  const day = graded[0].date, rows = graded.filter((p) => p.date === day);
  const hits = rows.filter(isHit).length;
  return `<p class="mono muted" style="margin:0 0 8px">${dayLabel(day)} · ${hits}/${rows.length} correct</p>
    <table><tbody>${rows.map(pickRow).join("")}</tbody></table>`;
}
const isHit = (p) => (p.p_home >= 0.5) === (p.score[1] > p.score[0]);
function pickRow(p) {
  const pick = p.p_home >= 0.5 ? p.home : p.away, conf = Math.max(p.p_home, 1 - p.p_home);
  const res = p.score ? (isHit(p) ? `<span class="hit">✓</span>` : `<span class="miss">✗</span>`) : `<span class="muted mono">pending</span>`;
  return `<tr><td class="mono muted">${p.date.slice(5)}</td><td>${esc(p.away)} @ ${esc(p.home)}</td><td><b>${esc(pick)}</b> <span class="muted">${pct(conf)}</span></td>
    <td class="mono">${p.score ? `${p.score[0]}–${p.score[1]}` : ""}</td><td class="r">${res}</td></tr>`;
}

function viewSlate() {
  const games = D.slate.games, live = D.record.live;
  const nLocks = games.filter((g) => confidence(g) >= 0.7).length;
  const top = D.teams.teams.slice(0, 6);
  return `<section class="hero">
    <div>
      <span class="pill"><span class="dot"></span>${esc(D.meta.season)} season · updated ${updatedLabel()}</span>
      <h1>Every NBA game,<br>called <em>before tipoff.</em></h1>
      <p class="lede">A model that knows who is actually playing. It rates every player on advanced metrics, projects minutes, reads the injury report, and publishes a win probability for each game the morning of.</p>
      <div class="stats">
        <div class="stat"><b data-count="${games.length}">0</b><span>games ${D.slate.date ? dayLabel(D.slate.date, { weekday: "short" }) : ""}</span></div>
        <div class="stat"><b data-count="${nLocks}">0</b><span>picks ≥ 70%</span></div>
        <div class="stat"><b data-count="${(D.backtest.accuracy * 100).toFixed(1)}" data-dec="1" data-suf="%">0</b><span>right in ${esc(D.backtest.season)} backtest</span></div>
        <div class="stat"><b>${live.n ? `${live.correct}-${live.n - live.correct}` : "0-0"}</b><span>live record</span></div>
      </div>
    </div>
    ${lockCard(games)}
  </section>
  <section>
    <div class="section-head">
      <div><h2>${D.slate.date ? dayLabel(D.slate.date) : "No upcoming games"}</h2><p>Bars show each team's win probability. Tap a game to see why.</p></div>
      <div class="chips">
        ${[["all", "All games"], ["locks", "Strong picks"], ["toss", "Close calls"]].map(([k, l]) => `<button class="chip ${state.filter === k ? "on" : ""}" data-filter="${k}">${l}</button>`).join("")}
        <span style="width:8px"></span>
        ${[["tip", "By tip time"], ["conf", "By confidence"]].map(([k, l]) => `<button class="chip ${state.sort === k ? "on" : ""}" data-sort="${k}">${l}</button>`).join("")}
      </div>
    </div>
    <div class="games">${filteredGames().map(gameCard).join("") || `<div class="card empty">No games match this filter.</div>`}</div>
  </section>
  <section class="two">
    <div class="card panel"><div class="panel-head"><h3>Latest results</h3><a href="#/record">Full record →</a></div>${lastResults()}</div>
    <div class="card panel"><div class="panel-head"><h3>Power rankings</h3><a href="#/rankings">All 30 →</a></div>
      <table><tbody>${top.map((t) => `<tr><td class="mono muted" style="width:28px">${t.rank}</td><td><div class="tm">${badge(t.abbr, t.color)}${esc(t.name)}</div></td>
        <td class="r" style="width:45%"><div class="bar-cell"><div class="track"><i data-w="${pct(t.power, 1)}"></i></div><span class="mono">${pct(t.power)}</span></div></td></tr>`).join("")}</tbody></table></div>
  </section>`;
}

function updatedLabel() {
  const d = new Date(D.meta.generated_at);
  return d.toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET";
}

/* ---------------------------------------------------------------- rankings */

function eloChart() {
  const teams = D.teams.teams;
  const dates = [...new Set(teams.flatMap((t) => t.elo_history.map((p) => p[0])))].sort();
  if (dates.length < 2) return `<div class="empty">Not enough games yet.</div>`;
  const idx = Object.fromEntries(dates.map((d, i) => [d, i]));
  const seasonOf = (d) => (+d.slice(5, 7) >= 8 ? +d.slice(0, 4) : +d.slice(0, 4) - 1);
  const series = teams.map((t) => {
    const arr = new Array(dates.length).fill(null), lastIdx = {};
    t.elo_history.forEach(([d, e]) => { arr[idx[d]] = e; lastIdx[seasonOf(d)] = idx[d]; });
    // carry a rating between a team's games, but not past its last game of a season
    for (let i = 1; i < arr.length; i++) {
      const s = seasonOf(dates[i]);
      if (arr[i] == null && seasonOf(dates[i - 1]) === s && i <= (lastIdx[s] ?? -1)) arr[i] = arr[i - 1];
    }
    return { t, arr };
  });
  const vals = series.flatMap((s) => s.arr.filter((v) => v != null));
  const [lo, hi] = [Math.floor((Math.min(...vals) - 20) / 50) * 50, Math.ceil((Math.max(...vals) + 20) / 50) * 50];
  const W = 1100, H = 340, m = { l: 44, r: 64, t: 12, b: 26 };
  const x = (i) => m.l + (i / (dates.length - 1)) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" id="elo" viewBox="0 0 ${W} ${H}" role="img" aria-label="Elo rating of every team over time">`;
  for (let v = lo; v <= hi; v += 100) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  // season breaks (gap > 60 days) and month ticks
  let lastMonth = "";
  dates.forEach((d, i) => {
    if (i && (new Date(d) - new Date(dates[i - 1])) / 864e5 > 60) {
      const bx = x(i - 0.5), right = bx > W * 0.75;
      s += `<line x1="${bx}" x2="${bx}" y1="${m.t}" y2="${H - m.b}" stroke="var(--line)" stroke-dasharray="4 4"/><text x="${bx + (right ? -6 : 6)}" y="${m.t + 12}" text-anchor="${right ? "end" : "start"}">${d.slice(0, 4)}-${String(+d.slice(2, 4) + 1).padStart(2, "0")} season ${right ? "→" : ""}</text>`;
    }
    const mo = d.slice(0, 7);
    if (mo !== lastMonth && i % 1 === 0 && (i === 0 || d.slice(8) <= "07")) {
      s += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${dayLabel(d, { month: "short" })}</text>`;
    }
    lastMonth = mo;
  });
  const path = (arr) => arr.map((v, i) => (v == null ? "" : `${arr[i - 1] == null ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)).join("");
  for (const { t, arr } of series) s += `<path class="elo-line" data-team="${t.abbr}" d="${path(arr)}" fill="none" stroke="var(--line)" stroke-width="1.3"/>`;
  s += `<g id="elo-hl"></g><line id="elo-x" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-dasharray="2 3" opacity="0"/>`;
  s += `<rect id="elo-hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
  eloChart.ctx = { series, dates, x, y, path, W, H, m };
  return s;
}

function eloHighlight(abbr, animate = false) {
  const c = eloChart.ctx;
  if (!c) return;
  const s = c.series.find((q) => q.t.abbr === abbr) || c.series[0];
  const last = s.arr.length - 1;
  $("#elo-hl").innerHTML = `<path class="${animate ? "draw" : ""}" d="${c.path(s.arr)}" fill="none" stroke="${s.t.color}" stroke-width="2.6" stroke-linejoin="round"/>
    <circle cx="${c.x(last)}" cy="${c.y(s.arr[last])}" r="5" fill="${s.t.color}" stroke="var(--card)" stroke-width="2"/>
    <text x="${c.x(last) + 9}" y="${c.y(s.arr[last]) + 4}" class="lbl-ink" style="font-weight:600">${s.t.abbr} ${s.arr[last]}</text>`;
  $$("#rank-table tbody tr").forEach((tr) => tr.classList.toggle("hl", tr.dataset.team === s.t.abbr));
}

function bindEloChart() {
  const svg = $("#elo"), c = eloChart.ctx;
  if (!svg || !c) return;
  eloHighlight(state.hlTeam || D.teams.teams[0].abbr, true);
  const hit = $("#elo-hit");
  hit.addEventListener("mousemove", (e) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const i = clamp(Math.round(((pt.x - c.m.l) / (c.W - c.m.l - c.m.r)) * (c.dates.length - 1)), 0, c.dates.length - 1);
    let best = null, bd = Infinity;
    for (const s of c.series) {
      const v = s.arr[i];
      if (v == null) continue;
      const d = Math.abs(c.y(v) - pt.y);
      if (d < bd) [bd, best] = [d, s];
    }
    if (!best) return;
    eloHighlight(best.t.abbr);
    const xl = $("#elo-x");
    xl.setAttribute("x1", c.x(i)); xl.setAttribute("x2", c.x(i)); xl.setAttribute("opacity", 1);
    showTip(`<b>${best.t.abbr}</b> ${best.arr[i]} Elo<br>${dayLabel(c.dates[i], { month: "short", day: "numeric", year: "numeric" })}`, e);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); $("#elo-x").setAttribute("opacity", 0); eloHighlight(state.hlTeam || D.teams.teams[0].abbr); });
}

function movers() {
  const season = D.teams.season.slice(0, 4);
  const rows = D.teams.teams.map((t) => {
    const cur = t.elo_history.filter((p) => p[0] >= `${season}-08-01`);
    const prev = t.elo_history.filter((p) => p[0] < `${season}-08-01`).slice(-1)[0];
    if (!cur.length) return null;
    // 10 games back; early in the season, the preseason rating (last season's, regressed like the model does)
    const base = cur.length > 10 ? cur[cur.length - 11][1] : prev ? 0.75 * prev[1] + 0.25 * 1505 : cur[0][1];
    return { t, delta: cur[cur.length - 1][1] - base, games: Math.min(cur.length, 10) };
  }).filter(Boolean).sort((a, b) => b.delta - a.delta);
  const list = (arr) => `<table><tbody>${arr.map((r) => `<tr><td><div class="tm">${badge(r.t.abbr, r.t.color)}${esc(r.t.name)}</div></td>
    <td class="r mono">${r.t.elo}</td><td class="r"><span class="move ${r.delta >= 0 ? "up" : "down"}">${r.delta >= 0 ? "▲ +" : "▼ "}${Math.round(r.delta)}</span></td></tr>`).join("")}</tbody></table>`;
  return `<section class="two" style="margin-top:14px">
    <div class="card panel"><div class="panel-head"><h3>Rising</h3><span class="mono muted">Elo change, last 10 games</span></div>${list(rows.slice(0, 5))}</div>
    <div class="card panel"><div class="panel-head"><h3>Falling</h3><span class="mono muted">Elo change, last 10 games</span></div>${list(rows.slice(-5).reverse())}</div>
  </section>`;
}

function viewRankings() {
  const ts = D.teams.teams;
  const leader = ts[0];
  return `<section class="hero" style="margin-bottom:28px">
    <div><span class="pill"><span class="dot"></span>Power rankings · ${dayLabel(D.teams.date, { month: "short", day: "numeric" })}</span>
      <h1 style="font-size:clamp(36px,5vw,56px)">Who would win<br>on a <em>neutral floor?</em></h1>
      <p class="lede">Each team's chance to beat a league-average opponent with the players it has available today: its Elo, recent form, and the rated strength of its healthy rotation.</p></div>
    <div class="lock"><div class="label">Current leaders</div>
      ${ts.slice(0, 3).map((t, i) => `<div class="lock-row" style="${i ? "" : "border:0"};align-items:center;padding:12px 0">
        <span style="opacity:1;display:flex;align-items:center;gap:10px"><b style="color:var(--orange);font:700 20px var(--display)">${t.rank}</b>${badge(t.abbr, t.color)}<span style="font:600 16px var(--font)">${esc(t.name)}</span></span>
        <span style="opacity:1;font:700 22px var(--display)" data-count="${(t.power * 100).toFixed(0)}" data-suf="%">0%</span></div>`).join("")}
    </div>
  </section>
  <section class="card panel"><div class="panel-head"><h3>Elo rating over time</h3><span class="mono muted">hover a line · click a team below</span></div>${eloChart()}</section>
  ${movers()}
  <section class="card" style="margin-top:14px"><div class="table-wrap"><table id="rank-table">
    <thead><tr><th>#</th><th></th><th>Team</th><th class="r">Record</th><th>vs. average team</th><th class="r">Elo</th><th class="r">Net rtg</th><th class="r">Lineup</th><th>Missing</th></tr></thead>
    <tbody>${ts.map((t) => {
      const mv = t.prev_rank == null ? "" : t.prev_rank > t.rank ? `<span class="move up">▲${t.prev_rank - t.rank}</span>` : t.prev_rank < t.rank ? `<span class="move down">▼${t.rank - t.prev_rank}</span>` : `<span class="move muted">–</span>`;
      return `<tr data-team="${t.abbr}" style="cursor:pointer"><td class="mono">${t.rank}</td><td>${mv}</td>
        <td><div class="tm">${badge(t.abbr, t.color)}<span>${esc(t.name)}</span></div></td>
        <td class="r mono">${t.w}-${t.l}</td>
        <td><div class="bar-cell"><div class="track"><i data-w="${pct(t.power, 1)}"></i></div><span class="mono">${pct(t.power)}</span></div></td>
        <td class="r mono">${t.elo}</td><td class="r mono" style="color:${t.form_net >= 0 ? "var(--good)" : "var(--bad)"}">${t.form_net > 0 ? "+" : ""}${t.form_net}</td>
        <td class="r mono">${t.lineup_gmsc?.toFixed(1)}</td>
        <td class="muted" style="font-size:12.5px">${t.missing.map((p) => esc(surname(p.name)) +(p.status === "Day-To-Day" ? "?" : "")).join(", ") || "—"}</td></tr>`;
    }).join("")}</tbody></table></div></section>`;
}

/* ---------------------------------------------------------------- players */

function spark(vals) {
  if (!vals || vals.length < 2) return "";
  const W = 90, H = 24, lo = Math.min(0, ...vals), hi = Math.max(...vals, 1);
  const x = (i) => 2 + (i / (vals.length - 1)) * (W - 6), y = (v) => H - 3 - ((v - lo) / (hi - lo)) * (H - 6);
  const pts = vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-linejoin="round"/>
    <circle cx="${x(vals.length - 1)}" cy="${y(vals[vals.length - 1])}" r="3" fill="var(--orange)"/></svg>`;
}

function playerRows() {
  const q = state.pQuery.toLowerCase();
  let ps = D.players.filter((p) => (!state.pTeam || p.team === state.pTeam) && (!q || p.name.toLowerCase().includes(q)));
  const k = state.pSort;
  ps.sort((a, b) => (b[k] ?? -1e9) - (a[k] ?? -1e9));
  const maxG = Math.max(...D.players.map((p) => p.gmsc));
  const shown = ps.slice(0, state.pLimit);
  return `${shown.map((p, i) => `<tr>
    <td class="mono muted">${i + 1}</td>
    <td><b style="font-weight:600">${esc(p.name)}</b>${p.status !== "Healthy" ? ` <span class="out ${p.status === "Day-To-Day" ? "dtd" : ""}">${p.status === "Out" ? "OUT" : "DTD"}</span>` : ""}</td>
    <td>${D.teamBy[p.team] ? `<div class="tm">${badge(p.team, D.teamBy[p.team].color)}</div>` : esc(p.team)}</td>
    <td class="r mono">${p.min}</td>
    <td><div class="bar-cell"><div class="track"><i data-w="${pct(clamp(p.gmsc / maxG, 0, 1), 1)}"></i></div><span class="mono">${p.gmsc}</span></div></td>
    <td class="r mono">${pct(p.pie, 1)}</td>
    <td class="r mono" style="color:${p.net >= 0 ? "var(--good)" : "var(--bad)"}">${p.net > 0 ? "+" : ""}${p.net}</td>
    <td>${spark(p.spark)}</td></tr>`).join("")}
    ${ps.length > shown.length ? `<tr><td colspan="8" style="text-align:center"><button class="chip" id="more">Show more (${ps.length - shown.length})</button></td></tr>` : ""}
    ${!ps.length ? `<tr><td colspan="8" class="empty">No players match.</td></tr>` : ""}`;
}

function viewPlayers() {
  const teams = [...new Set(D.players.map((p) => p.team))].filter(Boolean).sort();
  const cols = [["min", "Exp. min"], ["gmsc", "Game Score/36"], ["pie", "PIE"], ["net", "On-court net"]];
  return `<section class="section-head" style="margin-bottom:20px">
    <div><h2>Player ratings</h2><p>What the model thinks of every rotation player, with recent games weighted most. These ratings drive the lineup strength in each game.</p></div>
    <div class="search"><input id="pq" type="search" placeholder="Search a player" value="${esc(state.pQuery)}" aria-label="Search players">
      <select id="pt" aria-label="Filter by team"><option value="">All teams</option>${teams.map((t) => `<option ${t === state.pTeam ? "selected" : ""}>${t}</option>`).join("")}</select></div>
  </section>
  <section class="card"><div class="table-wrap"><table id="ptable">
    <thead><tr><th>#</th><th>Player</th><th>Team</th>${cols.map(([k, l]) => `<th class="sort ${k === "gmsc" ? "" : "r"} ${state.pSort === k ? "sorted" : ""}" data-psort="${k}">${l}</th>`).join("")}<th>Last 15 games</th></tr></thead>
    <tbody>${playerRows()}</tbody></table></div></section>`;
}

/* ---------------------------------------------------------------- record */

function calibrationChart() {
  const W = 540, H = 330, m = { l: 46, r: 16, t: 16, b: 38 };
  const x = (v) => m.l + ((v - 0.5) / 0.5) * (W - m.l - m.r), y = (v) => m.t + (1 - (v - 0.3) / 0.7) * (H - m.t - m.b);
  let s = `<svg class="chart" id="calib" viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration: predicted confidence versus actual win rate">`;
  for (let v = 0.3; v <= 1.001; v += 0.1) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  for (let v = 0.5; v <= 1.001; v += 0.1) s += `<text x="${x(v)}" y="${H - 18}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">model's confidence in its pick</text>`;
  s += `<line x1="${x(0.5)}" y1="${y(0.5)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/><text x="${x(0.93)}" y="${y(0.97) - 4}" text-anchor="end">perfect calibration</text>`;
  const series = [["Backtest " + D.backtest.season, D.backtest.calibration, "var(--slate)"], ["Live " + D.meta.season, D.record.live.calibration || [], "var(--orange)"]];
  series.forEach(([name, pts, col], si) => {
    if (pts.length > 1) s += `<polyline class="draw" points="${pts.map((p) => `${x(p.pred)},${y(p.actual)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
    pts.forEach((p, i) => {
      s += `<circle class="pop" style="animation-delay:${0.08 * i + si * 0.3}s" cx="${x(p.pred)}" cy="${y(p.actual)}" r="${clamp(Math.sqrt(p.n) * 0.75, 4, 14)}" fill="${col}" stroke="var(--card)" stroke-width="2"
        data-tip="<b>${esc(name)}</b><br>Picks at ${Math.round(p.lo * 100)}–${Math.round(p.hi * 100)}%: ${p.n}<br>Predicted ${pct(p.pred, 1)} · won ${pct(p.actual, 1)}"/>`;
    });
  });
  s += `</svg><div class="legend">${series.map(([n, pts, c]) => `<span><i style="background:${c};border-radius:50%;width:9px;height:9px"></i>${esc(n)}${pts.length ? "" : " (no games yet)"}</span>`).join("")}<span>dot size = number of games</span></div>`;
  return s;
}

function rollingChart() {
  const daily = D.backtest.daily, N = 150;
  const games = [];
  daily.forEach(([d, h, c]) => { for (let i = 0; i < c; i++) games.push([d, i < h ? 1 : 0]); });
  const pts = [];
  let sum = 0;
  games.forEach((g, i) => {
    sum += g[1];
    if (i >= N) sum -= games[i - N][1];
    if (i >= N - 1 && (i === games.length - 1 || games[i + 1][0] !== g[0])) pts.push([g[0], sum / N]);
  });
  const W = 540, H = 330, m = { l: 46, r: 16, t: 16, b: 38 }, lo = 0.5, hi = 0.85;
  const x = (i) => m.l + (i / (pts.length - 1)) * (W - m.l - m.r), y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" id="roll" viewBox="0 0 ${W} ${H}" role="img" aria-label="Rolling accuracy over last season">`;
  for (let v = lo; v <= hi + 0.001; v += 0.05) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  let lastMo = "";
  pts.forEach(([d], i) => { if (d.slice(0, 7) !== lastMo) { lastMo = d.slice(0, 7); if (i < pts.length - 8) s += `<text x="${x(i)}" y="${H - 18}" text-anchor="start">${dayLabel(d, { month: "short" })}</text>`; } });
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  s += `<path d="${line}L${x(pts.length - 1)},${y(lo)}L${x(0)},${y(lo)}Z" fill="var(--orange)" opacity=".1"/>`;
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(D.backtest.accuracy)}" y2="${y(D.backtest.accuracy)}" stroke="var(--ink-2)" stroke-dasharray="4 4"/><text x="${W - m.r}" y="${y(D.backtest.accuracy) - 6}" text-anchor="end" class="lbl-ink">season ${pct(D.backtest.accuracy, 1)}</text>`;
  s += `<path class="draw" d="${line}" fill="none" stroke="var(--orange)" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<line id="roll-x" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-dasharray="2 3" opacity="0"/><circle id="roll-dot" r="5" fill="var(--orange)" stroke="var(--card)" stroke-width="2" opacity="0"/>`;
  s += `<rect id="roll-hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">accuracy over the previous ${N} games</text></svg>`;
  rollingChart.ctx = { pts, x, y, W, m };
  return s;
}

function bindRolling() {
  const c = rollingChart.ctx, svg = $("#roll");
  if (!svg) return;
  $("#roll-hit").addEventListener("mousemove", (e) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const i = clamp(Math.round(((pt.x - c.m.l) / (c.W - c.m.l - c.m.r)) * (c.pts.length - 1)), 0, c.pts.length - 1);
    const [d, v] = c.pts[i];
    for (const [id, attrs] of [["#roll-x", { x1: c.x(i), x2: c.x(i), opacity: 1 }], ["#roll-dot", { cx: c.x(i), cy: c.y(v), opacity: 1 }]]) {
      for (const [k, val] of Object.entries(attrs)) $(id).setAttribute(k, val);
    }
    showTip(`<b>${pct(v, 1)}</b> correct<br>150 games through ${dayLabel(d, { month: "short", day: "numeric" })}`, e);
  });
  $("#roll-hit").addEventListener("mouseleave", () => { hideTip(); $("#roll-x").setAttribute("opacity", 0); $("#roll-dot").setAttribute("opacity", 0); });
}

function viewRecord() {
  const live = D.record.live, bt = D.backtest;
  const pending = D.record.picks.filter((p) => !p.score).length;
  return `<section class="section-head" style="margin-bottom:20px"><div><h2>Track record</h2>
    <p>Every pick is frozen when it is published, before tipoff, and graded the next morning. The backtest shows how the same model did on last season without having seen it.</p></div></section>
  <section class="big-stats">
    <div class="card big-stat"><span>Live record</span><b>${live.n ? `${live.correct}–${live.n - live.correct}` : "0–0"}</b><small>${pending} pick${pending === 1 ? "" : "s"} pending</small></div>
    <div class="card big-stat"><span>Live accuracy</span><b ${live.n ? `data-count="${(live.accuracy * 100).toFixed(1)}" data-dec="1" data-suf="%"` : ""}>${live.n ? "0" : "—"}</b><small>${live.n ? `log loss ${live.log_loss}` : "starts tonight"}</small></div>
    <div class="card big-stat"><span>${esc(bt.season)} backtest</span><b data-count="${(bt.accuracy * 100).toFixed(1)}" data-dec="1" data-suf="%">0</b><small>${bt.correct} of ${bt.n} games</small></div>
    <div class="card big-stat"><span>Vegas benchmark</span><b>67–69%</b><small>closing-line favorites</small></div>
  </section>
  <section class="two" style="margin-top:0">
    <div class="card panel"><div class="panel-head"><h3>When it says 70%, does it win 70%?</h3></div>${calibrationChart()}</div>
    <div class="card panel"><div class="panel-head"><h3>Accuracy through ${esc(bt.season)}</h3><span class="mono muted">backtest</span></div>${rollingChart()}</div>
  </section>
  <section class="card" style="margin-top:14px"><div class="panel-head" style="padding:18px 18px 0"><h3>Every pick</h3></div>
    <div class="table-wrap"><table><thead><tr><th>Date</th><th>Game</th><th>Pick</th><th>Score</th><th class="r">Result</th></tr></thead>
    <tbody>${D.record.picks.map(pickRow).join("") || `<tr><td colspan="5" class="empty">No picks yet.</td></tr>`}</tbody></table></div></section>`;
}

/* ---------------------------------------------------------------- how */

function viewHow() {
  return `<section class="hero" style="margin-bottom:0"><div><span class="pill"><span class="dot"></span>Methodology</span>
    <h1 style="font-size:clamp(36px,5vw,56px)">Who's playing<br><em>matters most.</em></h1>
    <p class="lede">Team ratings say how good a team has been. Tipoff also asks who will actually be on the floor tonight, and for how long.</p></div></section>
  <section class="steps">
    <div class="step s1"><span class="n">01</span><h3>Rate every player</h3><p>PIE, on-court net rating and Game Score per 36, weighted by minutes and recency, pulled toward a newcomer's level until a player has a real sample.</p></div>
    <div class="step s2"><span class="n">02</span><h3>Build tonight's lineup</h3><p>Take today's roster, drop players listed as out, count day-to-day players at half their minutes, and weight everyone by projected minutes.</p></div>
    <div class="step s4"><span class="n">03</span><h3>Blend with team form</h3><p>Margin-adjusted Elo, recent offensive and defensive efficiency, four factors and rest feed a logistic regression trained on 19,000+ games.</p></div>
  </section>
  <div class="prose">
    <h2>The daily cycle</h2>
    <p>Every morning, after the last game of the night is final, the model pulls the new box scores and retrains on every completed game since 2011. It then fetches the latest rosters and injury report and publishes a probability for each game on the next slate. Picks are frozen at publication and graded the following morning.</p>
    <h2>Reading a game</h2>
    <p>The <b>tug-of-war bar</b> shows each team's win probability. Open a game for the <b>factor chart</b>, which starts at 50/50 and shows how home court, Elo, recent form, the available players and rest each move the line. These steps add up exactly to the final number. The <b>rotation</b> panel shows projected minutes; darker bars are better-rated players and hatched bars are players who are out or questionable.</p>
    <h2>Backtest, last six seasons</h2>
    <div class="card" style="margin-top:8px"><table><thead><tr><th>Model</th><th class="r">Accuracy</th><th class="r">Log loss</th></tr></thead><tbody>
      <tr><td><b>Team + players (Tipoff)</b></td><td class="r mono">67.1%</td><td class="r mono">0.606</td></tr>
      <tr><td>Team stats only</td><td class="r mono">65.3%</td><td class="r mono">0.624</td></tr>
      <tr><td>Elo only</td><td class="r mono">64.2%</td><td class="r mono">0.638</td></tr>
      <tr><td>Always pick the home team</td><td class="r mono">55.4%</td><td class="r mono">0.689</td></tr></tbody></table></div>
    <h2>Limits</h2>
    <p>Late scratches after publication, minutes restrictions and surprise rest days aren't captured. The backtest knew exactly who played in each past game, so live accuracy will run a little lower. This is a statistical model, not betting advice.</p>
  </div>`;
}

/* ---------------------------------------------------------------- router */

function render() {
  const [, route = "", arg] = location.hash.split("/");
  if (route === "game") { state.open = arg; }
  const view = { "": viewSlate, game: viewSlate, rankings: viewRankings, players: viewPlayers, record: viewRecord, how: viewHow }[route] || viewSlate;
  const name = route === "game" ? "" : route;
  $$("nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === (name || "slate")));
  const app = $("#app");
  app.innerHTML = `<div class="view">${view()}</div>`;
  animateIn(app);
  if (route === "rankings") bindEloChart();
  if (route === "record") bindRolling();
  if (route === "game" && arg) requestAnimationFrame(() => $(`[data-game="${arg}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
}

function rerenderGames() {
  const grid = $(".games");
  grid.innerHTML = filteredGames().map(gameCard).join("") || `<div class="card empty">No games match this filter.</div>`;
  animateIn(grid);
}

document.addEventListener("click", (e) => {
  const t = e.target;
  const f = t.closest("[data-filter]"), so = t.closest("[data-sort]");
  if (f || so) {
    if (f) state.filter = f.dataset.filter;
    if (so) state.sort = so.dataset.sort;
    $$("[data-filter]").forEach((b) => b.classList.toggle("on", b.dataset.filter === state.filter));
    $$("[data-sort]").forEach((b) => b.classList.toggle("on", b.dataset.sort === state.sort));
    rerenderGames();
    return;
  }
  if (t.closest(".close")) { state.open = null; history.replaceState(null, "", "#/"); rerenderGames(); return; }
  const card = t.closest(".game");
  if (card && !card.classList.contains("open")) {
    state.open = card.dataset.game;
    history.replaceState(null, "", `#/game/${state.open}`);
    rerenderGames();
    requestAnimationFrame(() => $(`[data-game="${state.open}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    return;
  }
  const row = t.closest("#rank-table tbody tr");
  if (row) { state.hlTeam = row.dataset.team; eloHighlight(state.hlTeam, true); $("#elo")?.scrollIntoView({ behavior: "smooth", block: "center" }); return; }
  const ps = t.closest("[data-psort]");
  if (ps) { state.pSort = ps.dataset.psort; $$("[data-psort]").forEach((h) => h.classList.toggle("sorted", h.dataset.psort === state.pSort)); refreshPlayers(); return; }
  if (t.id === "more") { state.pLimit += 60; refreshPlayers(); }
});
function refreshPlayers() { const tb = $("#ptable tbody"); tb.innerHTML = playerRows(); animateIn(tb); }
document.addEventListener("input", (e) => {
  if (e.target.id === "pq") { state.pQuery = e.target.value; state.pLimit = 60; refreshPlayers(); }
  if (e.target.id === "pt") { state.pTeam = e.target.value; state.pLimit = 60; refreshPlayers(); }
});
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-tip]");
  if (el) showTip(el.dataset.tip, e);
});
document.addEventListener("mousemove", (e) => { const el = e.target.closest("[data-tip]"); if (el) showTip(el.dataset.tip, e); });
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });

$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (err) { /* private mode */ }
});

function tickCountdown() {
  for (const el of $$("[data-tip-at]")) {
    const at = etDate(el.dataset.tipAt);
    if (!at) continue;
    const ms = at - Date.now();
    if (ms <= 0) { el.textContent = "Games under way"; continue; }
    const h = Math.floor(ms / 36e5), m = Math.floor((ms % 36e5) / 6e4), s = Math.floor((ms % 6e4) / 1e3);
    el.textContent = `${tipLabel(el.dataset.tipAt)} · in ${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  }
}

function ticker() {
  const items = D.slate.games.map((g) => `<span>${esc(g.away.abbr)} @ ${esc(g.home.abbr)} <b>${esc(g.pick)} ${pct(confidence(g))}</b></span>`);
  items.push(`<span>${esc(D.backtest.season)} backtest <b>${pct(D.backtest.accuracy, 1)}</b></span>`);
  const lead = D.teams.teams[0];
  items.push(`<span>No. 1 power ranking <b>${esc(lead.abbr)}</b></span>`);
  $("#ticker").innerHTML = items.join("");
}

load().then(() => {
  ticker();
  $("#foot-meta").textContent = `Updated ${updatedLabel()} · trained on ${D.meta.trained_games.toLocaleString()} games`;
  render();
  addEventListener("hashchange", render);
  setInterval(tickCountdown, 1000);
  tickCountdown();
}).catch((err) => {
  $("#app").innerHTML = `<div class="empty">Couldn't load today's data (${esc(err.message)}).</div>`;
});
