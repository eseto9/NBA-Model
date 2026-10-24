/* Tipoff: renders the model's daily JSON (data/*.json) into an interactive site. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (p, d = 0) => (p * 100).toFixed(d) + "%";
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const D = {};               // loaded data
const state = { filter: "all", sort: "tip", open: null, hlTeam: null, pSort: "gmsc", pTeam: "", pQuery: "", pLimit: 60, scratch: {} };

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
  // betting lines refresh hourly, separately from the morning model run
  D.odds = await fetch(`data/odds.json?v=${Date.now()}`).then((r) => (r.ok ? r.json() : null)).catch(() => null) || { games: {} };
  for (const g of D.slate.games) {
    const key = `${g.date}|${g.away.abbr}|${g.home.abbr}`;
    g.market = D.odds.games[key] || null;
    g.spread = (D.odds.spreads || {})[key] || null;      // display only; never a model input
    republish(g);
  }
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
const ml = (v) => (v == null ? "—" : v > 0 ? `+${v}` : `${v}`);
const fairMl = (p) => (p >= 0.5 ? -Math.round((100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p));
const btSpan = () => `${D.backtest.first}–${D.backtest.through.slice(-5)}`;
/* Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9) */
function normInv(p) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const q = Math.min(Math.max(p, 1e-9), 1 - 1e-9), lo = 0.02425;
  if (q < lo) { const r = Math.sqrt(-2 * Math.log(q)); return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1); }
  if (q > 1 - lo) return -normInv(1 - q);
  const r = q - 0.5, t = r * r;
  return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * r / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1);
}
/* Spread implied by a win probability, assuming the final margin is normal with SD 12 points */
const MARGIN_SD = 12;
function impliedSpread(g, p) {
  const margin = MARGIN_SD * normInv(p);            // expected home margin
  const fav = margin >= 0 ? g.home.abbr : g.away.abbr;
  return Math.abs(margin) < 0.25 ? "PK" : `${fav} −${Math.abs(margin).toFixed(1)}`;
}

/* model minus market, in points, from the side the model likes more than the market does */
function edge(g) {
  if (!g.market) return null;
  const d = modelP(g) - g.market.p_home;
  return { team: d >= 0 ? g.home.abbr : g.away.abbr, pts: Math.abs(d) * 100 };
}
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
function animateIn(root) {
  $$("[data-w],[data-flex],[data-count],[data-dash]", root).forEach((el) => io.observe(el));
  // draw-in lines: dash length must match each path's real length, or long lines stop partway
  $$(".draw", root).forEach((el) => {
    const len = Math.ceil(el.getTotalLength ? el.getTotalLength() : 3000) + 2;
    el.style.strokeDasharray = len;
    el.style.strokeDashoffset = len;
  });
}
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
  const e = edge(g);
  if (disagrees(g)) tags.push(`<span class="tag split" title="The model and the betting market pick different winners">Model ≠ market</span>`);
  if (e && e.pts >= 5) tags.push(`<span class="tag edge" title="The model rates ${esc(e.team)} ${e.pts.toFixed(1)} percentage points more likely to win than the betting market does">Model +${e.pts.toFixed(0)}% on ${esc(e.team)}</span>`);
  if (g.neutral) tags.push(`<span class="tag">Neutral site</span>`);
  return tags.join(" ");
}

function teamBlock(side, cls, picked) {
  const t = D.teamBy[side.abbr];
  // records and ranks are as of today, so only show them on today's games
  const rec = t && (typeof viewDay !== "function" || isToday(viewDay())) ? `${t.w}-${t.l} · #${t.rank}` : "";
  return `<div class="team ${cls} ${picked ? "picked" : ""}">${badge(side.abbr, side.color)}
    <div style="min-width:0"><div class="team-name">${esc(side.name)}</div><div class="team-sub">${rec}</div></div></div>`;
}

function marketLine(g) {
  const m = g.market;
  const sp = g.spread ? ` · <span title="${esc(g.spread.provider || "")} spread and total, display only">${esc(g.spread.text)}${g.spread.total ? `, O/U ${g.spread.total}` : ""}</span>` : "";
  if (!m) return `<div class="mkt-line muted">${g.spread ? `Spread ${esc(g.spread.text)}${g.spread.total ? `, O/U ${g.spread.total}` : ""}` : "Betting line not posted yet"}</div>`;
  const fav = m.p_home >= 0.5 ? g.home.abbr : g.away.abbr;
  return `<div class="mkt-line"><span class="mkt-dot"></span>Market <b>${esc(fav)} ${pct(Math.max(m.p_home, 1 - m.p_home))}</b>
    <span class="muted">${m.home_ml != null ? `${esc(g.away.abbr)} ${ml(m.away_ml)} · ${esc(g.home.abbr)} ${ml(m.home_ml)}` : ""}${sp}</span></div>`;
}

/* the model and the market pick different winners */
const disagrees = (g) => !!g.market && (modelP(g) >= 0.5) !== (g.market.p_home >= 0.5);

/* a finished game: the score and whether the pick was right, nothing else */
function finalCard(g, open) {
  const x = g.live, homeWon = x.hs > x.as, hit = homeWon === (g.p_home >= 0.5);
  const pick = (p) => `${esc(p >= 0.5 ? g.home.abbr : g.away.abbr)} ${pct(Math.max(p, 1 - p))}`;
  const row = (side, pts, won) => `<div class="fc-row ${won ? "win" : ""}">${badge(side.abbr, side.color)}<span class="fc-name">${esc(side.name)}</span><b class="fc-pts">${pts}</b></div>`;
  return `<article class="card game is-final compact-final ${hit ? "call-hit" : "call-miss"} ${open ? "open" : ""} ${g.lite ? "lite" : ""}" data-game="${g.id}">
    <div class="fc-top"><span class="mono muted">FINAL${x.period > 4 ? ` · ${x.period - 4 > 1 ? x.period - 4 : ""}OT` : ""}</span>
      <span class="verdict big ${hit ? "v-hit" : "v-miss"}"><i>${hit ? "✓" : "✗"}</i>${hit ? "Called it" : "Missed"}</span></div>
    ${row(g.away, x.as, !homeWon)}${row(g.home, x.hs, homeWon)}
    <div class="fc-sub mono">Pick <b>${pick(g.p_home)}</b>${g.market ? ` · Market <b>${pick(g.market.p_home)}</b>` : ""}</div>
    ${open ? gameDetail(g) : ""}
  </article>`;
}

function gameCard(g) {
  if (g.live && g.live.state === "post") return finalCard(g, state.open === g.id);
  const pa = 1 - g.p_home, open = state.open === g.id;
  const awayColor = colorsClash(g.away.color, g.home.color) ? "#8a857b" : g.away.color;
  return `<article class="card game ${open ? "open" : ""} ${g.lite ? "lite" : ""} ${g.live && g.live.state === "post" ? `is-final ${(g.live.hs > g.live.as) === (g.p_home >= 0.5) ? "call-hit" : "call-miss"}` : ""} ${disagrees(g) ? "disagree" : ""}" data-game="${g.id}">
    <div class="game-top"><span class="mono muted">${g.lite ? "" : tipLabel(g.tip)}</span><span>${gameTags(g)}</span></div>
    <div class="matchup">${teamBlock(g.away, "away", g.pick === g.away.abbr)}<span class="at">@</span>${teamBlock(g.home, "home", g.pick === g.home.abbr)}</div>
    <div class="tug">
      <div class="tug-wrap">
        <div class="tug-bar" role="img" aria-label="${esc(g.away.abbr)} ${pct(pa)}, ${esc(g.home.abbr)} ${pct(g.p_home)}">
          <i style="background:${esc(awayColor)}" data-flex="calc(${pct(pa, 1)} - 1px)"></i><i style="background:${esc(g.home.color)}" data-flex="calc(${pct(g.p_home, 1)} - 1px)"></i>
</div>
        ${g.market && !(g.live && g.live.state !== "pre") ? `<span class="tug-mkt" style="left:${pct(1 - g.market.p_home, 1)}" data-tip="<b>Betting market</b><br>${esc(g.home.abbr)} ${pct(g.market.p_home)} · ${esc(g.market.provider || "DraftKings")} via ESPN, margin removed"></span>` : ""}
      </div>
      <div class="tug-nums num"><span class="${pa < 0.5 ? "lose" : ""}">${pct(pa)}</span><span class="tug-cap">${g.live && g.live.state === "post" ? "pregame" : ""}</span><span class="${g.p_home < 0.5 ? "lose" : ""}">${pct(g.p_home)}</span></div>
    </div>
    <div class="live-slot">${typeof liveBlock === "function" ? liveBlock(g) : ""}</div>
    ${marketLine(g)}
    ${scratchNote(g)}
    ${typeof injuryNote === "function" ? injuryNote(g) : ""}
    ${g.lite ? `<div class="outs"><span>Backfilled pick${g.source === "backfill" ? " (before launch)" : ""}</span></div>` : ""}
    <span class="expand-hint">details →</span>
    ${open ? gameDetail(g) : ""}
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
  { k: "lu_rapm", label: "Lineup RAPM", lo: -4, hi: 4, f: (v) => (v > 0 ? "+" : "") + v.toFixed(1), get: (s) => s.lineup.rapm },
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
    ${side.rotation.map((p) => {
      const mine = state.scratch[`${side.abbr}|${p.name}`];
      const status = mine === "out" ? "Out" : mine === "in" ? "Healthy" : p.status_live || p.status;
      return `<div class="pl ${esc(status)} ${mine ? "mine" : ""}" data-scratch="${esc(side.abbr)}|${esc(p.name)}" data-status="${esc(p.status)}" data-tip="<b>${esc(p.name)}</b><br>${esc(p.status)}${p.manual ? " (late scratch)" : ""}${mine ? `, you marked ${mine === "out" ? "out" : "playing"}` : ""} · ${p.min} exp. min<br>Game Score/36 ${p.gmsc} · PIE ${pct(p.pie, 1)} · on-court net ${p.net > 0 ? "+" : ""}${p.net}<br>RAPM ${p.rapm > 0 ? "+" : ""}${p.rapm} per 100 poss.<br><i>tap to change</i>">
      <span class="pl-name">${esc(p.name)}${p.manual ? ` <span class="tag scratch">scratch</span>` : ""}${mine ? ` <span class="tag mine-tag">${mine === "out" ? "you: out" : "you: in"}</span>` : ""}</span>
      <div class="pl-bar"><i style="background:${ratingColor(p.gmsc)}" data-w="${pct(clamp(p.min / 38, 0.05, 1))}"></i></div></div>`;
    }).join("")}</div>`;
}

function lineMovement(g) {
  const h = g.market.history;
  if (h.length < 2) return `<p class="mono muted" style="margin:8px 0 0">Line movement appears after the next odds update.</p>`;
  const W = 520, H = 130, m = { l: 40, r: 70, t: 10, b: 22 };
  const ts = h.map(([t]) => Date.parse(t)), ps = h.map(([, p]) => p);
  const lo = Math.min(...ps, g.p_home) - 0.03, hi = Math.max(...ps, g.p_home) + 0.03;
  const x = (t) => m.l + ((t - ts[0]) / Math.max(ts[ts.length - 1] - ts[0], 1)) * (W - m.l - m.r);
  const y = (p) => m.t + (1 - (p - lo) / (hi - lo)) * (H - m.t - m.b);
  const hr = (t) => new Date(t).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric" });
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Market probability over time">`;
  for (const p of [lo + 0.03, hi - 0.03]) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(p)}" y2="${y(p)}"/><text x="${m.l - 6}" y="${y(p) + 4}" text-anchor="end">${pct(p)}</text>`;
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(g.p_home)}" y2="${y(g.p_home)}" stroke="var(--orange)" stroke-width="2" stroke-dasharray="5 4"/>
    <text x="${W - m.r + 6}" y="${y(g.p_home) + 4}" style="fill:var(--orange)">model</text>`;
  s += `<polyline class="draw" points="${h.map(([t, p]) => `${x(Date.parse(t))},${y(p)}`).join(" ")}" fill="none" stroke="var(--slate)" stroke-width="2"/>`;
  h.forEach(([t, p]) => (s += `<circle cx="${x(Date.parse(t))}" cy="${y(p)}" r="3.5" fill="var(--slate)" data-tip="${hr(Date.parse(t))} ET · market ${esc(g.home.abbr)} ${pct(p, 1)}"/>`));
  s += `<text x="${W - m.r + 6}" y="${y(ps[ps.length - 1]) + 4}" style="fill:var(--slate)">market</text>`;
  s += `<text x="${m.l}" y="${H - 4}">${hr(ts[0])}</text><text x="${W - m.r}" y="${H - 4}" text-anchor="end">${hr(ts[ts.length - 1])}</text></svg>`;
  return s;
}

/* The published pick blends the model with the betting market: the walk-forward stack from the
   backtest, logit p = a + b*logit(market) + c*logit(model). The model's own number is p_model. */
const modelP = (g) => (g.p_model != null ? g.p_model : g.p_home);
function blendP(pModel, pMarket) {
  const b = D.backtest.blend && D.backtest.blend.coef;
  if (!b || pMarket == null) return pModel;
  const lg = (p) => Math.log(p / (1 - p));
  return 1 / (1 + Math.exp(-(b.a + b.market * lg(pMarket) + b.model * lg(pModel))));
}
/* re-blend with the latest line (odds refresh after the morning model run), until tipoff */
function republish(g) {
  if (g.live && g.live.state !== "pre") return;
  const pm = g.p_model_adj != null ? g.p_model_adj : modelP(g);
  g.p_home = g.market ? blendP(pm, g.market.p_home) : pm;
  g.pick = g.p_home >= 0.5 ? g.home.abbr : g.away.abbr;
  const last = g.steps && g.steps[g.steps.length - 1];
  if (last && last.label === "Betting market") last.to = g.p_home;
  else if (g.steps && g.steps.length && g.market && Math.abs(g.p_home - pm) > 1e-9) g.steps.push({ label: "Betting market", from: pm, to: g.p_home });
}

/* The side with positive expected value at the posted moneyline, if any */
function valueLine(g) {
  const m = g.market;
  const sides = [[g.away, 1 - g.p_home, m.away_ml], [g.home, g.p_home, m.home_ml]]
    .map(([t, p, price]) => ({ t, p, price, ev: p * (1 + payout(price)) - 1 })).sort((a, b) => b.ev - a.ev);
  const b = sides[0];
  if (b.ev <= 0) return `<p class="mono" style="margin:10px 0 0">Value: <b>none</b>, neither side's price beats the model's probability.</p>`;
  return `<p class="mono value-line" style="margin:10px 0 0">Value: <b>${esc(b.t.abbr)} moneyline ${ml(b.price)}</b>, model ${pct(b.p)} vs ${pct(1 / (1 + payout(b.price)), 1)} break-even, +${(b.ev * 100).toFixed(1)}¢ per $1</p>`;
}

function marketPanel(g) {
  const m = g.market;
  if (!m) return "";
  const pmod = modelP(g);
  const row = (side, p, pm, line, best) => `<tr><td><div class="tm">${badge(side.abbr, side.color)}${esc(side.abbr)}</div></td>
    <td class="r mono">${pct(p)}</td><td class="r mono">${pct(pm)}</td><td class="r mono">${ml(fairMl(p))}</td><td class="r mono">${ml(line)}</td></tr>`;
  const e = edge(g);
  return `<h4>Model vs betting market</h4>
    <table class="mkt-table"><thead><tr><th></th><th class="r">Model</th><th class="r">Market</th><th class="r">Model's fair odds</th><th class="r">Market odds</th></tr></thead><tbody>
      ${row(g.away, 1 - pmod, 1 - m.p_home, m.away_ml, m.away_best)}${row(g.home, pmod, m.p_home, m.home_ml, m.home_best)}</tbody></table>
    ${valueLine(g)}
    ${D.backtest.blend ? `<p class="mono" style="margin:10px 0 0">Published pick: <b>${esc(g.p_home >= 0.5 ? g.home.abbr : g.away.abbr)} ${pct(Math.max(g.p_home, 1 - g.p_home))}</b> = model ${pct(Math.max(pmod, 1 - pmod))} blended with the market <span class="muted" title="Logistic stack fit walk-forward on past seasons: ${(D.backtest.blend.coef.market).toFixed(2)} x market log-odds + ${(D.backtest.blend.coef.model).toFixed(2)} x model log-odds. It matches the market's accuracy in the backtest; the model alone is about 0.4 points behind.">ⓘ</span></p>` : ""}
    ${g.spread ? `<p class="mono" style="margin:10px 0 0">Spread <b>${esc(g.spread.text)}</b>${g.spread.total ? ` · total ${g.spread.total}` : ""} <span class="muted">(${esc(g.spread.provider || "")}, display only)</span> · model ≈ <b>${impliedSpread(g, g.p_home)}</b> <span class="muted" title="Converts the model's win probability to a point spread, assuming the final margin is normal with SD ${MARGIN_SD}">ⓘ</span></p>` : ""}
    <p class="mono muted" style="margin:8px 0 4px">${e.pts < 1 ? "Model and market agree." : `Model is ${e.pts.toFixed(1)} pts higher on ${esc(e.team)} than the market.`} Market = ${esc(m.provider || "DraftKings")} moneyline via ESPN, margin removed.${m.home_open != null ? ` Opened ${esc(g.away.abbr)} ${ml(m.away_open)} / ${esc(g.home.abbr)} ${ml(m.home_open)}.` : ""}</p>
    ${lineMovement(g)}`;
}

function gameDetail(g) {
  return `<div class="detail">
    <div><h4>Why the ${g.p_default != null ? "default " : ""}model leans ${esc(g.p_default != null ? (g.p_default >= 0.5 ? g.home.abbr : g.away.abbr) : g.pick)}</h4>${waterfall(g)}
      ${g.p_default != null ? `<p class="mono muted">Your custom model says ${esc(g.home.abbr)} ${pct(g.p_home)}; this breakdown is for the default model (${pct(g.p_default)}).</p>` : ""}
      <h4 style="margin-top:22px">Tale of the tape</h4>${compare(g)}</div>
    <div>${g.market ? `${marketPanel(g)}<div style="height:22px"></div>` : ""}<h4>Projected rotations <span class="muted" style="text-transform:none;letter-spacing:0">· tap a player to mark him out</span></h4><div class="rot">${rotationCol(g.away)}${rotationCol(g.home)}</div>
      <div class="legend"><span><i style="background:var(--orange)"></i>bar = expected minutes, darker = better rating</span><span><i style="background:repeating-linear-gradient(135deg,var(--muted) 0 2px,transparent 2px 5px)"></i>out</span><span><i style="background:repeating-linear-gradient(135deg,var(--orange) 0 2px,transparent 2px 5px)"></i>day-to-day</span></div>
      <p class="mono muted" style="margin-top:16px">Elo alone: ${esc(g.home.abbr)} ${pct(g.p_elo)} · model: ${esc(g.home.abbr)} ${pct(g.p_home)}</p></div>
  </div>`;
}

function currentGames() {
  return typeof dayGames === "function" ? dayGames(viewDay()) : D.slate.games;
}

function filteredGames() {
  let gs = [...(currentGames() || [])];
  if (state.filter === "locks") gs = gs.filter((g) => confidence(g) >= 0.7);
  if (state.filter === "toss") gs = gs.filter((g) => confidence(g) < 0.6);
  if (state.sort === "conf") gs.sort((a, b) => confidence(b) - confidence(a));
  return gs;
}

/* profit per $1 staked at American odds */
const payout = (price) => (price > 0 ? price / 100 : 100 / -price);

/* Best value: highest expected return per $1 at the best available price, using the model's probability */
function bestValue(games) {
  let best = null;
  for (const g of games) {
    if (!g.market) continue;
    for (const [side, p, price, pm] of [[g.home, g.p_home, g.market.home_best, g.market.p_home], [g.away, 1 - g.p_home, g.market.away_best, 1 - g.market.p_home]]) {
      if (price == null) continue;
      const dec = price > 0 ? 1 + price / 100 : 1 + 100 / -price;
      const ev = p * dec - 1;
      if (!best || ev > best.ev) best = { g, side, p, pm, price, ev };
    }
  }
  return best;
}

/* Hero picks: the best-value bet (or the strongest pick when nothing beats the price), and the
   most likely winner as its own card. Both show the result once the game is over. */
function heroPicks(games, day = D.slate.date) {
  if (!games || !games.length) return `<div class="lock"><div class="label">No games ${games ? "scheduled" : "loaded yet"}</div><p>Pick another day.</p></div>`;
  const conf = games.reduce((a, b) => (confidence(b) > confidence(a) ? b : a));
  const v = bestValue(games);
  const useValue = v && v.ev > 0;
  return valueCard(useValue ? v : null, conf, games, day) + (useValue && v.g !== conf ? winnerCard(conf, day) : "");
}

function valueCard(v, conf, games, day) {
  const g = v ? v.g : conf;
  const side = v ? v.side : g.p_home >= 0.5 ? g.home : g.away, opp = side === g.home ? g.away : g.home;
  const c = v ? v.p : confidence(g);
  const R = 56, C = 2 * Math.PI * R, r2 = R - (v ? 13 : 0), C2 = 2 * Math.PI * r2;
  const label = v ? "Best value vs the odds" : bestValue(games) ? "Strongest pick · no bet beats the odds" : isToday(day) ? "Strongest pick · odds not posted yet" : "Strongest pick";
  const breakEven = v ? 1 / (1 + payout(v.price)) : null;
  return `<div class="lock">
    <div class="label">${label} · ${dayLabel(day, { weekday: "short", month: "short", day: "numeric" })}</div>
    <div class="lock-body">
      <div class="gauge"><svg viewBox="0 0 132 132" width="132" height="132">
        <circle cx="66" cy="66" r="${R}" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="11"/>
        ${v ? `<circle cx="66" cy="66" r="${R}" fill="none" stroke="var(--slate)" stroke-width="11" stroke-dasharray="${C}" stroke-dashoffset="${C}" data-dash="${C * (1 - breakEven)}" class="arc" opacity=".9"/>` : ""}
        <circle class="arc" cx="66" cy="66" r="${r2}" fill="none" stroke="var(--orange)" stroke-width="11" stroke-linecap="round"
          stroke-dasharray="${C2}" stroke-dashoffset="${C2}" data-dash="${C2 * (1 - c)}"/></svg>
        <div class="gauge-val"><div><b data-count="${(c * 100).toFixed(0)}" data-suf="%">0%</b><small>${v ? "Tipoff" : "to win"}</small></div></div></div>
      <div>${v ? `<div class="lock-position">Bet: ${esc(side.abbr)} moneyline ${ml(v.price)}</div>` : ""}
        <div class="lock-team">${esc(side.name)}${v ? " to win" : ""}</div>
        <div class="lock-sub">${side === g.home ? "vs" : "at"} ${esc(opp.name)}</div>
      </div>
    </div>
    ${v ? `<div class="lock-explain">
      <div><span class="dotk" style="background:var(--orange)"></span>Tipoff gives ${esc(side.abbr)} <b>${pct(v.p, 1)}</b></div>
      <div><span class="dotk" style="background:var(--slate)"></span>At ${ml(v.price)} they only need <b>${pct(breakEven, 1)}</b> to break even</div>
    </div>` : ""}
    ${lockResult(g, side)}
  </div>`;
}

function winnerCard(g, day) {
  const side = g.p_home >= 0.5 ? g.home : g.away, opp = side === g.home ? g.away : g.home;
  const w = winnerOf(g);
  return `<a class="card winner-card ${w ? (w === side.abbr ? "won" : "lost") : ""}" href="${isToday(day) ? `#/game/${g.id}` : `#/day/${day}`}">
    <span class="mono muted wc-label">Most likely winner</span>
    <span class="wc-team">${badge(side.abbr, side.color)}<span><b>${esc(side.name)}</b><small class="muted">${side === g.home ? "vs" : "at"} ${esc(opp.name)}</small></span></span>
    <b class="wc-pct">${pct(confidence(g))}</b>
    ${w ? `<span class="verdict ${w === side.abbr ? "v-hit" : "v-miss"}"><i>${w === side.abbr ? "✓" : "✗"}</i>${w === side.abbr ? "Called it" : "Missed"}</span>` : ""}
  </a>`;
}

/* redraw the hero picks when one of their games finishes (not on every score update) */
function paintHeroPicks() {
  const el = document.getElementById("hero-picks");
  if (!el || !isToday(viewDay())) return;
  const sig = D.slate.games.filter((g) => winnerOf(g)).map((g) => g.id).join();
  if (sig === paintHeroPicks.sig) return;
  paintHeroPicks.sig = sig;
  el.innerHTML = heroPicks(D.slate.games, D.slate.date);
  animateIn(el);
}

/* how a pick turned out: winner of a finished game, or null */
const winnerOf = (g) => (g.live && g.live.state === "post" ? (g.live.hs > g.live.as ? g.home.abbr : g.away.abbr) : null);
const resultMark = (g, side) => (winnerOf(g) ? (winnerOf(g) === side.abbr ? " ✓" : " ✗") : "");

function lockResult(g, side) {
  const w = winnerOf(g);
  if (!w) return "";
  const won = w === side.abbr;
  return `<div class="lock-result ${won ? "won" : "lost"}"><i>${won ? "✓" : "✗"}</i>${won ? "Called it" : "Missed"}
    <span>final ${esc(g.away.abbr)} ${g.live.as} – ${g.live.hs} ${esc(g.home.abbr)}</span></div>`;
}

/* "games today", or the next day with games when today has none (the slate is always the next game day) */
function slateDayLabel() {
  if (!D.slate.date) return "games scheduled";
  const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  return D.slate.date === today ? "games today" : `games ${dayLabel(D.slate.date, { weekday: "short" })} · next game day`;
}

/* season record, counting tonight's finished games before the morning update grades them */
function seasonRecord() {
  let hit = 0, n = 0, mHit = 0, mN = 0;
  const graded = new Set();
  for (const p of D.record.picks) {
    if (!p.score) continue;
    n++; hit += isHit(p); graded.add(`${p.date}|${p.away}|${p.home}`);
    if (p.p_market != null) { mN++; mHit += (p.p_market >= 0.5) === (p.score[1] > p.score[0]); }
  }
  for (const g of D.slate.games) {
    if (!g.live || g.live.state !== "post" || graded.has(`${D.slate.date}|${g.away.abbr}|${g.home.abbr}`)) continue;
    const homeWon = g.live.hs > g.live.as;
    n++; hit += homeWon === (g.p_home >= 0.5);
    if (g.market) { mN++; mHit += homeWon === (g.market.p_home >= 0.5); }
  }
  return { hit, n, mHit, mN };
}

function splitPanel(games) {
  if (!games) return "";
  const split = games.filter(disagrees);
  const past = D.record.picks.filter((p) => p.score && p.p_market != null && (p.p_home >= 0.5) !== (p.p_market >= 0.5));
  const modelRight = past.filter(isHit).length;
  if (!split.length && !past.length) return "";
  return `<div class="split-panel card">
    <div class="split-head"><b>${split.length ? `Model and market disagree on ${split.length} game${split.length > 1 ? "s" : ""}` : "Model and market agree on every game"}</b>
      ${past.length ? `<span class="mono muted">when they've disagreed this season: model ${modelRight}–${past.length - modelRight}</span>` : ""}</div>
    ${split.length ? `<div class="split-list">${split.map((g) => {
      const mSide = g.p_home >= 0.5 ? g.home : g.away, kSide = mSide === g.home ? g.away : g.home;
      const w = winnerOf(g);
      return `<a class="split-item ${w ? (w === mSide.abbr ? "won" : "lost") : ""}" href="${isToday(viewDay()) ? `#/game/${g.id}` : `#/day/${viewDay()}`}">
        <span>${badge(g.away.abbr, g.away.color)}<span class="at">@</span>${badge(g.home.abbr, g.home.color)}</span>
        <span>Model <b>${esc(mSide.abbr)} ${pct(Math.max(g.p_home, 1 - g.p_home))}</b><br>Market <b>${esc(kSide.abbr)} ${pct(Math.max(g.market.p_home, 1 - g.market.p_home))}</b></span>
        ${w ? `<span class="split-res">${w === mSide.abbr ? "✓ model" : "✓ market"}</span>` : ""}</a>`;
    }).join("")}</div>` : ""}
  </div>`;
}

const isHit = (p) => (p.p_home >= 0.5) === (p.score[1] > p.score[0]);

function viewSlate() {
  const games = D.slate.games, rec = seasonRecord();
  const top = D.teams.teams.slice(0, 6);
  return `<section class="hero">
    <div>
      <span class="pill"><span class="dot"></span>${esc(D.meta.season)} season · updated ${updatedLabel()}</span>
      <h1>Every NBA game,<br>called <em>before tipoff.</em></h1>
      <p class="lede">A model that knows who is actually playing. It rates every player on advanced metrics, projects minutes, reads the injury report, and publishes a win probability for each game the morning of.</p>
      <div class="stats">
        <div class="stat"><b data-count="${games.length}">0</b><span>${slateDayLabel()}</span></div>
        <div class="stat"><b id="season-pct">${rec.n ? pct(rec.hit / rec.n, 1) : "—"}</b><span>Tipoff right this season</span></div>
        <div class="stat"><b id="season-mkt" class="mkt-stat">${rec.mN ? pct(rec.mHit / rec.mN, 1) : "—"}</b><span>market right this season</span></div>
        <div class="stat"><b id="season-rec">${rec.hit}–${rec.n - rec.hit}</b><span>record this season</span></div>
      </div>
    </div>
    <div class="hero-picks" id="hero-picks">${heroPicks(currentGames(), viewDay())}</div>
  </section>
  ${D.custom ? `<div class="custom-banner card"><span><b>Showing your custom model</b> · ${esc(describe({ ...D.custom }))}</span><a href="#/lab">Edit in Model Lab →</a></div>` : ""}
  <section id="slate">
    ${typeof dayStrip === "function" ? dayStrip() : ""}
    <div class="section-head">
      <div><h2>${viewDay() ? dayLabel(viewDay()) : "No upcoming games"}</h2>${isToday(viewDay()) ? `<div class="tonight" id="tonight"></div>` : dayHeading(viewDay(), currentGames())}<p>Bars show each team's win probability; the dark tick is the betting market. Tap a game to see why. <span class="muted">Late scratch? Open a game and tap the player.</span></p></div>
      <div class="chips">
        ${[["all", "All games"], ["locks", "Strong picks"], ["toss", "Close calls"]].map(([k, l]) => `<button class="chip ${state.filter === k ? "on" : ""}" data-filter="${k}">${l}</button>`).join("")}
        <span style="width:8px"></span>
        ${[["tip", "By tip time"], ["conf", "By confidence"]].map(([k, l]) => `<button class="chip ${state.sort === k ? "on" : ""}" data-sort="${k}">${l}</button>`).join("")}
      </div>
    </div>
    ${isToday(viewDay()) ? `<div class="card news" id="inj-news"></div>` : ""}
    <div id="split-wrap">${splitPanel(currentGames())}</div>
    <div class="games">${currentGames() == null ? `<div class="card empty"><span class="bounce" style="display:inline-block;width:18px;height:18px"></span></div>`
      : filteredGames().map(gameCard).join("") || `<div class="card empty">No games match this filter.</div>`}</div>
  </section>`;
}

function updatedLabel() {
  const d = new Date(D.meta.generated_at);
  return d.toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET";
}

/* ---------------------------------------------------------------- rankings */

function eloChart() {
  const teams = D.teams.teams;
  // playoff games only involve half the league, so by default they're left out
  const hist = (t) => t.elo_history.filter((p) => state.eloPlayoffs || !p[2]);
  const dates = [...new Set(teams.flatMap((t) => hist(t).map((p) => p[0])))].sort();
  if (dates.length < 2) return `<div class="empty">Not enough games yet.</div>`;
  const idx = Object.fromEntries(dates.map((d, i) => [d, i]));
  const seasonOf = (d) => (+d.slice(5, 7) >= 8 ? +d.slice(0, 4) : +d.slice(0, 4) - 1);
  const series = teams.map((t) => {
    const arr = new Array(dates.length).fill(null), lastIdx = {};
    hist(t).forEach(([d, e]) => { arr[idx[d]] = e; lastIdx[seasonOf(d)] = idx[d]; });
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
  let last = s.arr.length - 1;
  while (last > 0 && s.arr[last] == null) last--;          // the team's most recent game on the chart
  $("#elo-hl").innerHTML = `<path class="${animate ? "draw" : ""}" d="${c.path(s.arr)}" fill="none" stroke="${s.t.color}" stroke-width="2.6" stroke-linejoin="round"/>
    <circle cx="${c.x(last)}" cy="${c.y(s.arr[last])}" r="5" fill="${s.t.color}" stroke="var(--card)" stroke-width="2"/>
    <text x="${c.x(last) + 9}" y="${c.y(s.arr[last]) + 4}" class="lbl-ink" style="font-weight:600">${s.t.abbr} ${s.arr[last]}</text>`;
  if (animate) animateIn($("#elo-hl"));
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
  <section class="card panel"><div class="panel-head"><h3>Elo rating over time</h3>
    <div class="chips"><span class="mono muted" style="margin-right:6px">hover a line · click a team below</span>
      <button class="chip ${state.eloPlayoffs ? "" : "on"}" data-elo-po="0">Regular season</button><button class="chip ${state.eloPlayoffs ? "on" : ""}" data-elo-po="1">Including playoffs</button></div></div>
    <div id="elo-wrap">${eloChart()}</div></section>
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
    <td class="r mono" style="color:${(p.rapm ?? 0) >= 0 ? "var(--good)" : "var(--bad)"}">${p.rapm == null ? "—" : (p.rapm > 0 ? "+" : "") + p.rapm.toFixed(1)}</td>
    <td>${spark(p.spark)}</td></tr>`).join("")}
    ${ps.length > shown.length ? `<tr><td colspan="9" style="text-align:center"><button class="chip" id="more">Show more (${ps.length - shown.length})</button></td></tr>` : ""}
    ${!ps.length ? `<tr><td colspan="9" class="empty">No players match.</td></tr>` : ""}`;
}

function viewPlayers() {
  const teams = [...new Set(D.players.map((p) => p.team))].filter(Boolean).sort();
  const cols = [["min", "Exp. min"], ["gmsc", "Game Score/36"], ["pie", "PIE"], ["net", "On-court net"], ["rapm", "RAPM"]];
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
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">confidence in the favorite</text>`;
  s += `<line x1="${x(0.5)}" y1="${y(0.5)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/><text x="${x(0.93)}" y="${y(0.97) - 4}" text-anchor="end">perfect calibration</text>`;
  const series = [
    ["Model, backtest", D.backtest.overall.calibration, "var(--orange)"],
    ["Betting market", D.backtest.vs_market.market?.calibration || [], "var(--slate)"],
    ["Model, " + D.meta.season + " so far", D.record.season.calibration || [], "var(--ink)"],
  ];
  // bins with fewer than 10 games are too noisy to plot
  series.forEach((sr) => (sr[1] = sr[1].filter((p) => p.n >= 10).map((p) => ({ ...p, actual: clamp(p.actual, 0.3, 1) }))));
  series.forEach(([name, pts, col], si) => {
    if (pts.length > 1) s += `<polyline class="draw" points="${pts.map((p) => `${x(p.pred)},${y(p.actual)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
    pts.forEach((p, i) => {
      s += `<circle class="pop" style="animation-delay:${0.08 * i + si * 0.3}s" cx="${x(p.pred)}" cy="${y(p.actual)}" r="${clamp(Math.sqrt(p.n) * 0.3, 4, 12)}" fill="${col}" stroke="var(--card)" stroke-width="2"
        data-tip="<b>${esc(name)}</b><br>Favorites at ${Math.round(p.lo * 100)}–${Math.round(p.hi * 100)}%: ${p.n.toLocaleString()} games<br>Predicted ${pct(p.pred, 1)} · won ${pct(p.actual, 1)}"/>`;
    });
  });
  s += `</svg><div class="legend">${series.map(([n, pts, c]) => `<span><i style="background:${c};border-radius:50%;width:9px;height:9px"></i>${esc(n)}${pts.length ? "" : " (not enough games yet)"}</span>`).join("")}<span>dot size = games · bins with 10+ games</span></div>`;
  return s;
}

/* Model vs market accuracy, season by season */
function seasonChart() {
  const ss = D.backtest.seasons;
  const W = 1100, H = 300, m = { l: 46, r: 16, t: 18, b: 34 }, lo = 0.55, hi = 0.75;
  const band = (W - m.l - m.r) / ss.length, bw = Math.min(26, band / 2 - 4);
  const y = (v) => m.t + (1 - (clamp(v, lo, hi) - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by season, model versus betting market">`;
  for (let v = lo; v <= hi + 0.001; v += 0.05) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  ss.forEach((q, i) => {
    const cx = m.l + band * (i + 0.5), vm = q.vs_market;
    const bars = [["Model", q.accuracy, "var(--orange)"]];
    if (vm.n) bars.push(["Market", vm.market.accuracy, "var(--slate)"]);
    const tip = `<b>${esc(q.season)}</b><br>Model ${pct(q.accuracy, 1)} of ${q.n} games` + (vm.n
      ? `<br>On the ${vm.n} games with a line:<br>model ${pct(vm.model.accuracy, 1)} · market ${pct(vm.market.accuracy, 1)}<br>disagreed ${vm.disagree_n}×, model right ${vm.disagree_model_right}`
      : "<br>No betting lines this season");
    bars.forEach(([, v, col], j) => {
      const bx = bars.length === 1 ? cx - bw / 2 : cx + (j ? 1 : -bw - 1);
      s += `<rect class="growy" style="animation-delay:${0.04 * i}s" x="${bx}" y="${y(v)}" width="${bw}" height="${y(lo) - y(v)}" rx="4" fill="${col}" data-tip="${tip}"/>`;
    });
    s += `<text x="${cx}" y="${H - 14}" text-anchor="middle">${q.season.slice(2, 4)}-${q.season.slice(-2)}</text>`;
  });
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(lo)}" y2="${y(lo)}" class="axis"/></svg>
    <div class="legend"><span><i style="background:var(--orange)"></i>Tipoff model (every game)</span><span><i style="background:var(--slate)"></i>Betting favorite (games with an ESPN line)</span><span>hover a season for the head-to-head</span></div>`;
  return s;
}

function rollingChart() {
  const sel = state.rollSeason || "all";
  let daily = D.backtest.daily;
  if (sel !== "all") {
    const y0 = +sel.slice(0, 4);
    daily = daily.filter(([d]) => d >= `${y0}-08-01` && d < `${y0 + 1}-08-01`);
  }
  const N = sel === "all" ? 400 : 150;
  const games = [];
  daily.forEach(([d, h, c]) => { for (let i = 0; i < c; i++) games.push([d, i < h ? 1 : 0]); });
  const pts = [];
  let sum = 0;
  games.forEach((g, i) => {
    sum += g[1];
    if (i >= N) sum -= games[i - N][1];
    if (i >= N - 1 && (i === games.length - 1 || games[i + 1][0] !== g[0])) pts.push([g[0], sum / N]);
  });
  const acc = games.reduce((a, g) => a + g[1], 0) / games.length;
  const W = 1100, H = 280, m = { l: 46, r: 16, t: 16, b: 34 }, lo = 0.5, hi = 0.85;
  const x = (i) => m.l + (i / Math.max(pts.length - 1, 1)) * (W - m.l - m.r), y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" id="roll" viewBox="0 0 ${W} ${H}" role="img" aria-label="Rolling accuracy">`;
  for (let v = lo; v <= hi + 0.001; v += 0.05) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  // one label per season (all seasons) or per month, skipping any that would crowd the previous one
  let last = "", lastX = -Infinity;
  pts.forEach(([d], i) => {
    const yr = +d.slice(0, 4), season = +d.slice(5, 7) >= 8 ? yr : yr - 1;
    const key = sel === "all" ? String(season) : d.slice(0, 7);
    if (key === last) return;
    last = key;
    if (x(i) - lastX < 44 || i > pts.length - 6) return;
    lastX = x(i);
    s += `<text x="${x(i)}" y="${H - 12}">${sel === "all" ? `${String(season).slice(2)}-${String(season + 1).slice(2)}` : dayLabel(d, { month: "short" })}</text>`;
  });
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  s += `<path d="${line}L${x(pts.length - 1)},${y(lo)}L${x(0)},${y(lo)}Z" fill="var(--orange)" opacity=".1"/>`;
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(acc)}" y2="${y(acc)}" stroke="var(--ink-2)" stroke-dasharray="4 4"/><text x="${W - m.r}" y="${y(acc) - 6}" text-anchor="end" class="lbl-ink">average ${pct(acc, 1)}</text>`;
  s += `<path class="draw" d="${line}" fill="none" stroke="var(--orange)" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<line id="roll-x" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-dasharray="2 3" opacity="0"/><circle id="roll-dot" r="5" fill="var(--orange)" stroke="var(--card)" stroke-width="2" opacity="0"/>`;
  s += `<rect id="roll-hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
  rollingChart.ctx = { pts, x, y, W, m, N };
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
    showTip(`<b>${pct(v, 1)}</b> correct<br>${c.N} games through ${dayLabel(d, { month: "short", day: "numeric", year: "numeric" })}`, e);
  });
  $("#roll-hit").addEventListener("mouseleave", () => { hideTip(); $("#roll-x").setAttribute("opacity", 0); $("#roll-dot").setAttribute("opacity", 0); });
}

function viewRecord() {
  const rec = seasonRecord();          // same live count as the Today page
  const sea = D.record.season, live = D.record.live, o = D.backtest.overall, vm = D.backtest.vs_market, svm = D.record.vs_market || { n: 0 };
  const pending = D.record.picks.filter((p) => !p.score).length;
  const seasons = D.backtest.seasons.map((q) => q.season);
  const right = vm.n ? vm.disagree_model_right / vm.disagree_n : 0;
  const nBack = D.record.picks.filter((p) => p.score && p.source === "backfill").length;
  return `<section class="section-head" style="margin-bottom:20px"><div><h2>Track record</h2>
    <p>Every game since opening night of ${esc(D.meta.season)}, graded against the final score and the closing betting line. Picks published on the site are revisable until tipoff, then frozen. The ${nBack} games before the site launched${D.record.first_live ? ` (${dayLabel(D.record.first_live, { month: "short", day: "numeric" })})` : ""} are <b>backfilled</b>: predicted afterwards from pre-game data by a model trained only on earlier seasons. The backtest replays every season since ${esc(D.backtest.first)} the same way.</p></div></section>
  <section class="big-stats">
    <div class="card big-stat"><span>${esc(D.meta.season)} record</span><b>${rec.hit}–${rec.n - rec.hit}</b><small>${rec.n ? `${pct(rec.hit / rec.n, 1)} right · ` : ""}${nBack} backfilled · ${pending} pending</small></div>
    <div class="card big-stat"><span>Season vs market</span><b>${rec.n ? pct(rec.hit / rec.n, 1) : "—"} <span class="mkt-stat">/ ${rec.mN ? pct(rec.mHit / rec.mN, 1) : "—"}</span></b><small>Tipoff / betting favorite, same as the Today page, incl. tonight</small></div>
    <div class="card big-stat"><span>Backtest ${btSpan()}</span><b data-count="${(o.accuracy * 100).toFixed(1)}" data-dec="1" data-suf="%">0</b><small>${o.correct.toLocaleString()} of ${o.n.toLocaleString()} games</small></div>
    <div class="card big-stat"><span>Market, same games</span><b ${vm.n ? `data-count="${(vm.market.accuracy * 100).toFixed(1)}" data-dec="1" data-suf="%"` : ""}>${vm.n ? "0" : "—"}</b><small>${vm.n ? `model ${pct(vm.model.accuracy, 1)} on these ${vm.n.toLocaleString()}` : "no lines"}</small></div>
  </section>
  ${vm.n ? `<section class="card panel vs-strip">
    <div class="vs-head"><span class="mono muted">When the model and the market disagreed on the favorite</span><b>${vm.disagree_n.toLocaleString()} games</b></div>
    <div class="vs-bar"><i style="background:var(--orange)" data-w="${pct(right, 1)}"></i><i style="background:var(--slate)" data-w="${pct(1 - right, 1)}"></i></div>
    <div class="vs-labels mono"><span>model right ${pct(right, 1)}</span><span>market right ${pct(1 - right, 1)}</span></div>
    <div class="mono muted">Log loss (lower is better): model ${vm.model.log_loss} · market ${vm.market.log_loss} · they pick the same favorite in ${pct(vm.agree, 0)} of games</div>
    ${D.backtest.blend ? `<div class="mono muted">Tipoff × Market blend, walk-forward on ${D.backtest.blend.vs_market.n.toLocaleString()} games: ${pct(D.backtest.blend.vs_market.model.accuracy, 1)} accuracy, log loss ${D.backtest.blend.vs_market.model.log_loss} (market ${pct(D.backtest.blend.vs_market.market.accuracy, 1)}, ${D.backtest.blend.vs_market.market.log_loss}). <a href="#/how" class="scratch-link">Can it beat the market?</a></div>` : ""}
  </section>` : ""}
  <section class="card panel" style="margin-top:14px"><div class="panel-head"><h3>Model vs betting market, season by season</h3><span class="mono muted">walk-forward backtest</span></div>${seasonChart()}</section>
  <section class="two" style="margin-top:14px">
    <div class="card panel"><div class="panel-head"><h3>When it says 70%, does it win 70%?</h3></div>${calibrationChart()}</div>
    <div class="card panel"><div class="panel-head"><h3>Seasons at a glance</h3></div>
      <div class="table-wrap"><table><thead><tr><th>Season</th><th class="r">Games</th><th class="r">Model</th><th class="r">Market</th><th class="r">Log loss</th></tr></thead><tbody>
      ${D.backtest.seasons.slice().reverse().map((q) => `<tr><td class="mono">${esc(q.season)}</td><td class="r mono">${q.n}</td><td class="r mono"><b>${pct(q.accuracy, 1)}</b></td>
        <td class="r mono">${q.vs_market.n ? `${pct(q.vs_market.market.accuracy, 1)} <span class="muted">(${q.vs_market.n})</span>` : "—"}</td><td class="r mono">${q.log_loss}</td></tr>`).join("")}</tbody></table></div></div>
  </section>
  <section class="card panel" style="margin-top:14px"><div class="panel-head"><h3>Rolling accuracy</h3>
    <div class="chips">${["all", ...seasons].map((k) => `<button class="chip ${(state.rollSeason || "all") === k ? "on" : ""}" data-roll="${k}">${k === "all" ? "All seasons" : k}</button>`).join("")}</div></div>
    <div id="roll-wrap">${rollingChart()}</div></section>
  ${pickLog()}`;
}

/* ---------------------------------------------------------------- pick log */

const pickSide = (p) => (p.p_home >= 0.5 ? p.home : p.away);
const pickConf = (p) => Math.max(p.p_home, 1 - p.p_home);
const mktFor = (p) => (p.p_market == null ? null : pickSide(p) === p.home ? p.p_market : 1 - p.p_market);
const mktHit = (p) => (p.p_market >= 0.5) === (p.score[1] > p.score[0]);
const teamColor = (abbr) => (D.teamBy[abbr] ? D.teamBy[abbr].color : "#888");

/* every pick as a square, oldest first: the season at a glance */
function pickStrip(picks) {
  const ordered = picks.slice().reverse();
  return `<div class="strip">${ordered.map((p) => {
    const cls = !p.score ? "pend" : isHit(p) ? "hit" : "miss";
    const tip = `<b>${esc(p.away)} @ ${esc(p.home)}</b> · ${p.date}<br>Pick ${esc(pickSide(p))} ${pct(pickConf(p))}${p.score ? `<br>Final ${p.score[0]}–${p.score[1]} · ${isHit(p) ? "right" : "wrong"}` : "<br>pending"}${p.source === "backfill" ? "<br><i>backfilled</i>" : ""}`;
    return `<i class="sq ${cls} ${p.source === "backfill" ? "bf" : ""}" data-tip="${tip}"></i>`;
  }).join("")}</div>
  <div class="legend"><span><i style="background:var(--good)"></i>right</span><span><i style="background:var(--bad)"></i>wrong</span><span><i style="background:var(--line)"></i>pending</span><span><i class="bf-key"></i>backfilled (before launch)</span></div>`;
}

/* running accuracy through the season, model vs the betting favorite on the same games */
function cumulativeChart(picks) {
  const g = picks.filter((p) => p.score).slice().reverse();
  if (g.length < 3) return `<p class="lab-hint">The running record appears after a few graded games.</p>`;
  let hm = 0, hk = 0, nk = 0;
  const pts = g.map((p, i) => {
    hm += isHit(p);
    if (p.p_market != null) { nk++; hk += mktHit(p); }
    return { i, model: hm / (i + 1), market: nk ? hk / nk : null, date: p.date };
  });
  const W = 1100, H = 220, m = { l: 46, r: 70, t: 14, b: 28 };
  const lo = Math.max(0.3, Math.min(...pts.map((q) => Math.min(q.model, q.market ?? 1))) - 0.05), hi = Math.min(1, Math.max(...pts.map((q) => Math.max(q.model, q.market ?? 0))) + 0.05);
  const x = (i) => m.l + (i / Math.max(pts.length - 1, 1)) * (W - m.l - m.r), y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Running accuracy this season, model versus market">`;
  for (let v = Math.ceil(lo * 20) / 20; v <= hi + 1e-9; v += 0.05) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  const path = (k) => pts.filter((q) => q[k] != null).map((q, j) => `${j ? "L" : "M"}${x(q.i).toFixed(1)},${y(q[k]).toFixed(1)}`).join("");
  s += `<path class="draw" d="${path("market")}" fill="none" stroke="var(--slate)" stroke-width="2"/>`;
  s += `<path class="draw" d="${path("model")}" fill="none" stroke="var(--orange)" stroke-width="2.5"/>`;
  const lastQ = pts[pts.length - 1];
  s += `<circle cx="${x(lastQ.i)}" cy="${y(lastQ.model)}" r="5" fill="var(--orange)" stroke="var(--card)" stroke-width="2"/><text x="${x(lastQ.i) + 9}" y="${y(lastQ.model) + 4}" style="fill:var(--orange)">model ${pct(lastQ.model, 0)}</text>`;
  if (lastQ.market != null) s += `<circle cx="${x(lastQ.i)}" cy="${y(lastQ.market)}" r="4" fill="var(--slate)"/><text x="${x(lastQ.i) + 9}" y="${y(lastQ.market) + (lastQ.market > lastQ.model ? -6 : 14)}" style="fill:var(--slate)">market ${pct(lastQ.market, 0)}</text>`;
  let lastDay = "";
  pts.forEach((q) => { if (q.date !== lastDay) { lastDay = q.date; s += `<text x="${x(q.i)}" y="${H - 8}">${q.date.slice(5)}</text>`; } });
  return s + `</svg>`;
}

/* one pick: matchup, confidence bar with the market marked on it, score, result */
function pickCard(p) {
  const side = pickSide(p), c = pickConf(p), mk = mktFor(p);
  const res = !p.score ? `<span class="res pend">pending</span>` : isHit(p) ? `<span class="verdict v-hit"><i>✓</i>Called it</span>` : `<span class="verdict v-miss"><i>✗</i>Missed</span>`;
  const winner = p.score ? (p.score[1] > p.score[0] ? p.home : p.away) : null;
  const score = p.score ? `<span class="${winner === p.away ? "w" : ""}">${esc(p.away)} ${p.score[0]}</span> – <span class="${winner === p.home ? "w" : ""}">${p.score[1]} ${esc(p.home)}</span>` : "";
  return `<div class="pick ${!p.score ? "" : isHit(p) ? "is-hit" : "is-miss"}">
    <div class="pick-match">${badge(p.away, teamColor(p.away))}<span class="at">@</span>${badge(p.home, teamColor(p.home))}
      ${p.source === "backfill" ? `<span class="tag" title="Before the site launched: predicted afterwards by a model trained only on earlier seasons">backfill</span>` : ""}</div>
    <div class="pick-bar"><div class="pick-lbl"><b>${esc(side)}</b> ${pct(c)}${mk != null ? ` <span class="muted">· market ${pct(mk)}</span>` : ""}</div>
      <div class="pb-track"><i style="background:${teamColor(side)}" data-w="${pct(c, 1)}"></i>${mk != null ? `<span class="pb-mkt" style="left:${pct(mk, 1)}" title="market ${pct(mk)}"></span>` : ""}<span class="pb-half"></span></div></div>
    <div class="pick-score mono">${score}</div>
    ${res}
  </div>`;
}

function pickLog() {
  const picks = D.record.picks;
  if (!picks.length) return `<section class="card panel" style="margin-top:14px"><h3>Every pick this season</h3><p class="empty">No picks yet.</p></section>`;
  const days = [...new Set(picks.map((p) => p.date))];
  return `<section class="card panel" style="margin-top:14px">
    <div class="panel-head"><h3>Every pick this season</h3><span class="mono muted">${picks.length} games</span></div>
    ${pickStrip(picks)}
    <h4 class="sub-h">Running accuracy</h4>${cumulativeChart(picks)}
    <div class="days">${days.map((d) => {
      const ps = picks.filter((p) => p.date === d), graded = ps.filter((p) => p.score), hits = graded.filter(isHit).length;
      return `<div class="day"><div class="day-head"><span>${dayLabel(d, { weekday: "short", month: "short", day: "numeric" })}</span>
        <span class="mono">${graded.length ? `${hits}–${graded.length - hits}` : `${ps.length} pending`}</span>
        ${graded.length ? `<span class="day-bar"><i style="width:${pct(hits / graded.length, 1)}"></i></span>` : ""}</div>
        ${ps.map(pickCard).join("")}</div>`;
    }).join("")}</div>
  </section>`;
}

/* ---------------------------------------------------------------- how */

/* ---------------------------------------------------------------- router */

function render() {
  const [, route = "", arg] = location.hash.split("/");
  if (route === "game") { state.open = arg; state.day = null; }
  if (route === "day") { state.day = arg === D.slate.date ? null : arg; state.open = null; }
  if (!route) state.day = null;
  const view = { "": viewSlate, game: viewSlate, day: viewSlate, rankings: viewRankings, players: viewPlayers, record: viewRecord, how: viewMethod, lab: viewLab }[route] || viewSlate;
  const name = route === "game" || route === "day" ? "" : route;
  $$("nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === (name || "slate")));
  const app = $("#app");
  app.innerHTML = `<div class="view">${view()}</div>`;
  animateIn(app);
  if (route === "rankings") bindEloChart();
  if (route === "record") bindRolling();
  if (route === "how") renderMath(app);
  if (route === "lab") mountLab();
  if (!route || route === "game" || route === "day") requestAnimationFrame(() => {
    if (typeof centerDayStrip === "function") centerDayStrip();
    if (isToday(viewDay()) && typeof repaintLive === "function") repaintLive();
    if (typeof paintNews === "function") paintNews();
  });
  if (route === "day" && window.scrollY > 400) requestAnimationFrame(() => $("#slate")?.scrollIntoView({ block: "start" }));
  if (route === "game" && arg) requestAnimationFrame(() => $(`[data-game="${arg}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
}

function rerenderGames() {
  const grid = $(".games");
  grid.innerHTML = filteredGames().map(gameCard).join("") || `<div class="card empty">No games match this filter.</div>`;
  animateIn(grid);
  requestAnimationFrame(() => isToday(viewDay()) && typeof repaintLive === "function" && repaintLive());
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
  const openCard = t.closest(".game.open");
  if (openCard && !t.closest(".detail, a, button, input, select")) {        // click the card again to close it
    state.open = null;
    if (isToday(viewDay())) history.replaceState(null, "", "#/");
    rerenderGames();
    return;
  }
  const card = t.closest(".game");
  if (card && card.classList.contains("lite")) return;
  if (card && !card.classList.contains("open")) {
    state.open = card.dataset.game;
    if (isToday(viewDay())) history.replaceState(null, "", `#/game/${state.open}`);
    rerenderGames();
    requestAnimationFrame(() => $(`[data-game="${state.open}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    return;
  }
  const row = t.closest("#rank-table tbody tr");
  if (row) { state.hlTeam = row.dataset.team; eloHighlight(state.hlTeam, true); $("#elo")?.scrollIntoView({ behavior: "smooth", block: "center" }); return; }
  const ps = t.closest("[data-psort]");
  if (ps) { state.pSort = ps.dataset.psort; $$("[data-psort]").forEach((h) => h.classList.toggle("sorted", h.dataset.psort === state.pSort)); refreshPlayers(); return; }
  if (t.id === "more") { state.pLimit += 60; refreshPlayers(); return; }
  const po = t.closest("[data-elo-po]");
  if (po) {
    state.eloPlayoffs = po.dataset.eloPo === "1";
    $$("[data-elo-po]").forEach((b) => b.classList.toggle("on", b === po));
    $("#elo-wrap").innerHTML = eloChart();
    animateIn($("#elo-wrap"));
    bindEloChart();
    return;
  }
  const jump = t.closest("[data-jump]");
  if (jump) { e.preventDefault(); document.getElementById(jump.dataset.jump)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const roll = t.closest("[data-roll]");
  if (roll) {
    state.rollSeason = roll.dataset.roll;
    $$("[data-roll]").forEach((b) => b.classList.toggle("on", b.dataset.roll === state.rollSeason));
    $("#roll-wrap").innerHTML = rollingChart();
    animateIn($("#roll-wrap"));
    bindRolling();
  }
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
  items.push(`<span>Backtest ${btSpan()} <b>${pct(D.backtest.overall.accuracy, 1)}</b></span>`);
  if (D.backtest.vs_market.n) items.push(`<span>Betting favorites, same games <b>${pct(D.backtest.vs_market.market.accuracy, 1)}</b></span>`);
  const edges = D.slate.games.map((g) => ({ g, e: edge(g) })).filter((x) => x.e && x.e.pts >= 5);
  for (const { g, e } of edges) items.push(`<span>${esc(g.away.abbr)} @ ${esc(g.home.abbr)} edge <b>${esc(e.team)} +${e.pts.toFixed(0)}</b> vs market</span>`);
  const lead = D.teams.teams[0];
  items.push(`<span>No. 1 power ranking <b>${esc(lead.abbr)}</b></span>`);
  $("#ticker").innerHTML = items.join("");
}

load().then(async () => {
  D.custom = await applyCustomToToday().catch(() => null);
  state.scratch = loadScratches();
  applyScratches();
  ticker();
  $("#foot-meta").textContent = `Updated ${updatedLabel()} · trained on ${D.meta.trained_games.toLocaleString()} games`;
  render();
  addEventListener("hashchange", render);
  setInterval(tickCountdown, 1000);
  if (typeof pollLive === "function") pollLive();
  if (typeof pollInjuries === "function") pollInjuries();
  tickCountdown();
}).catch((err) => {
  $("#app").innerHTML = `<div class="empty">Couldn't load today's data (${esc(err.message)}).</div>`;
});
