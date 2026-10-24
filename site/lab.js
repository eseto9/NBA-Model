/* Tipoff Model Lab: refit the model in the browser with your own model type and inputs.
   Fitting runs in a Web Worker: an L2 logistic regression by Newton's method (the same
   objective as the site's default scikit-learn model) or a histogram gradient-boosted tree
   ensemble with logistic loss. The walk-forward backtest uses the same season splits. */
"use strict";

const LAB_KEY = "tipoff-lab";
const LAB = { data: null, config: null, result: null, running: false, worker: null };

/* ---------------------------------------------------------------- worker (as source text) */

const WORKER_SRC = String.raw`
"use strict";
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function standardize(cols, idx) {
  return cols.map((c) => {
    let m = 0; for (const i of idx) m += c[i]; m /= idx.length;
    let v = 0; for (const i of idx) v += (c[i] - m) ** 2; const sd = Math.sqrt(v / idx.length) || 1;
    return { m, sd };
  });
}

/* L2 logistic regression: minimize sum(logloss) + ||w||^2 / (2C), intercept unpenalized (scikit-learn's objective) */
function fitLogistic(cols, y, idx, C) {
  const k = cols.length, st = standardize(cols, idx), lam = 1 / C;
  const w = new Float64Array(k + 1);
  const x = new Float64Array(k + 1);
  for (let it = 0; it < 30; it++) {
    const g = new Float64Array(k + 1), H = Array.from({ length: k + 1 }, () => new Float64Array(k + 1));
    for (const i of idx) {
      x[0] = 1; for (let j = 0; j < k; j++) x[j + 1] = (cols[j][i] - st[j].m) / st[j].sd;
      let z = 0; for (let j = 0; j <= k; j++) z += w[j] * x[j];
      const p = sigmoid(z), r = p - y[i], h = p * (1 - p);
      for (let a = 0; a <= k; a++) { g[a] += r * x[a]; const ha = h * x[a]; for (let b = a; b <= k; b++) H[a][b] += ha * x[b]; }
    }
    for (let a = 1; a <= k; a++) { g[a] += lam * w[a]; H[a][a] += lam; }
    for (let a = 0; a <= k; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];
    const step = solve(H, g);
    let mx = 0; for (let a = 0; a <= k; a++) { w[a] -= step[a]; mx = Math.max(mx, Math.abs(step[a])); }
    if (mx < 1e-9) break;
  }
  return { kind: "logistic", w: Array.from(w), st };
}
function solve(A, b) {           // Gaussian elimination with partial pivoting (small k)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
function predLogistic(m, cols, i) {
  let z = m.w[0]; for (let j = 0; j < cols.length; j++) z += m.w[j + 1] * (cols[j][i] - m.st[j].m) / m.st[j].sd;
  return sigmoid(z);
}

/* Histogram gradient boosting, logistic loss, Newton leaf values with L2 = 1 */
function fitGBM(cols, y, idx, o) {
  const k = cols.length, B = 32, lam = 1, n = idx.length;
  const edges = cols.map((c) => {                       // quantile bin edges from the training rows
    const v = idx.map((i) => c[i]).sort((a, b) => a - b), e = [];
    for (let q = 1; q < B; q++) e.push(v[Math.floor((q * n) / B)]);
    return [...new Set(e)];
  });
  const bin = (j, v) => { const e = edges[j]; let lo = 0, hi = e.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v > e[m]) lo = m + 1; else hi = m; } return lo; };
  const X = cols.map((c, j) => Uint8Array.from(idx, (i) => bin(j, c[i])));
  const yy = Float64Array.from(idx, (i) => y[i]);
  let ybar = 0; for (const v of yy) ybar += v; ybar /= n;
  const base = Math.log(ybar / (1 - ybar)), F = new Float64Array(n).fill(base);
  const g = new Float64Array(n), h = new Float64Array(n), gain = new Float64Array(k), trees = [];
  for (let t = 0; t < o.trees; t++) {
    for (let r = 0; r < n; r++) { const p = sigmoid(F[r]); g[r] = p - yy[r]; h[r] = p * (1 - p); }
    const tree = [];
    const grow = (rows, depth) => {
      let G = 0, Hs = 0; for (const r of rows) { G += g[r]; Hs += h[r]; }
      const node = { leaf: -G / (Hs + lam) };
      tree.push(node);
      if (depth >= o.depth || rows.length < 2 * o.minLeaf) return node;
      let best = null;
      for (let j = 0; j < k; j++) {
        const hg = new Float64Array(B), hh = new Float64Array(B), hc = new Int32Array(B);
        for (const r of rows) { const b = X[j][r]; hg[b] += g[r]; hh[b] += h[r]; hc[b]++; }
        let gl = 0, hl = 0, cl = 0;
        for (let b = 0; b < B - 1; b++) {
          gl += hg[b]; hl += hh[b]; cl += hc[b];
          if (cl < o.minLeaf || rows.length - cl < o.minLeaf) continue;
          const gr = G - gl, hr = Hs - hl;
          const s = (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam) - (G * G) / (Hs + lam);
          if (!best || s > best.s) best = { s, j, b };
        }
      }
      if (!best || best.s <= 0) return node;
      gain[best.j] += best.s;
      const L = [], R = []; for (const r of rows) (X[best.j][r] <= best.b ? L : R).push(r);
      node.j = best.j; node.b = best.b;
      node.l = grow(L, depth + 1); node.r = grow(R, depth + 1);
      return node;
    };
    const root = grow([...Array(n).keys()], 0);
    const walk = (nd, r) => (nd.l ? walk(X[nd.j][r] <= nd.b ? nd.l : nd.r, r) : nd.leaf);
    for (let r = 0; r < n; r++) F[r] += o.lr * walk(root, r);
    trees.push(root);
  }
  return { kind: "gbm", base, trees, lr: o.lr, bin, gain: Array.from(gain) };
}
function predGBM(m, cols, i) {
  let z = m.base;
  const walk = (nd) => (nd.l ? walk(m.bin(nd.j, cols[nd.j][i]) <= nd.b ? nd.l : nd.r) : nd.leaf);
  for (const t of m.trees) z += m.lr * walk(t);
  return sigmoid(z);
}

function fit(cfg, cols, y, idx) {
  if (cfg.model === "elo") return { kind: "elo" };
  return cfg.model === "gbm" ? fitGBM(cols, y, idx, cfg) : fitLogistic(cols, y, idx, cfg.C);
}
function predict(m, cols, elo, i) {
  return m.kind === "elo" ? elo[i] : m.kind === "gbm" ? predGBM(m, cols, i) : predLogistic(m, cols, i);
}

onmessage = (ev) => {
  const { cfg, D } = ev.data;          // D: { season, y, elo, cols, names, slate: {cols, elo} }
  const seasons = [...new Set(D.season)].sort((a, b) => a - b);
  const last = seasons[seasons.length - 1];
  const tests = seasons.slice(1);       // the first season only trains
  const p = new Float64Array(D.y.length).fill(NaN);
  tests.forEach((s, k) => {
    const train = [], test = [];
    D.season.forEach((q, i) => (q < s ? train : q === s ? test : null)?.push(i));
    const m = fit(cfg, D.cols, D.y, train);
    for (const i of test) p[i] = predict(m, D.cols, D.elo, i);
    postMessage({ progress: (k + 1) / (tests.length + 1), season: s });
  });
  // final model on everything, for today's slate and the coefficient / importance chart
  const all = [...D.y.keys()];
  const m = fit(cfg, D.cols, D.y, all);
  const slate = D.slate ? D.slate.elo.map((_, i) => predict(m, D.slate.cols, D.slate.elo, i)) : [];
  const explain = m.kind === "logistic" ? D.names.map((n, j) => ({ name: n, v: m.w[j + 1] }))
    : m.kind === "gbm" ? D.names.map((n, j) => ({ name: n, v: m.gain[j] })) : [];
  postMessage({ done: true, p: Array.from(p), slate, explain, kind: m.kind, last });
};
`;

/* ---------------------------------------------------------------- data & config */

async function labLoad() {
  if (LAB.data) return LAB.data;
  const [h, l] = await Promise.all(["lab_hist.json", "lab_live.json"].map((f) =>
    fetch(`data/${f}?v=${encodeURIComponent(D.meta.generated_at)}`).then((r) => r.json())));
  const cat = (k) => [...h[k], ...l[k]];
  LAB.data = {
    groups: l.groups, unavailable: l.unavailable, def: l.default, currentSeason: l.season,
    season: cat("season"), y: cat("y"), elo: cat("elo_prob"), market: cat("p_market"), date: cat("date"),
    X: Object.fromEntries(Object.keys(l.X).map((c) => [c, [...h.X[c], ...l.X[c]]])), slate: l.slate || null,
  };
  return LAB.data;
}

function defaultConfig(d) { return { model: "logistic", C: d.def.C, trees: 150, depth: 3, lr: 0.05, minLeaf: 100, groups: [...d.def.groups] }; }
function savedConfig() { try { return JSON.parse(localStorage.getItem(LAB_KEY)) || null; } catch (e) { return null; } }
function saveConfig(cfg) { try { localStorage.setItem(LAB_KEY, JSON.stringify(cfg)); } catch (e) { /* private mode */ } }
function isDefault(cfg, d) {
  const def = defaultConfig(d);
  return cfg.model === def.model && cfg.C === def.C && [...cfg.groups].sort().join() === [...def.groups].sort().join();
}
function describe(cfg) {
  return cfg.model === "elo" ? "Elo only" : `${cfg.model === "gbm" ? "Gradient boosting" : "Logistic"} · ${cfg.groups.length} input groups`;
}

function runModel(cfg, d, onProgress) {
  return new Promise((resolve) => {
    const names = cfg.model === "elo" ? [] : d.groups.filter((g) => cfg.groups.includes(g.key)).flatMap((g) => g.features);
    const payload = {
      season: d.season, y: d.y, elo: d.elo, names, cols: names.map((n) => d.X[n]),
      slate: d.slate ? { cols: names.map((n) => d.slate.X[n]), elo: d.slate.elo_prob } : null,
    };
    if (LAB.worker) { LAB.worker.terminate(); if (LAB.cancel) LAB.cancel(null); }   // a newer run supersedes it
    LAB.cancel = resolve;
    LAB.worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" })));
    LAB.worker.onmessage = (ev) => {
      if (!ev.data.done) return onProgress && onProgress(ev.data);
      LAB.cancel = null;
      resolve(ev.data);
    };
    LAB.worker.postMessage({ cfg, D: payload });
  });
}

/* ---------------------------------------------------------------- scoring */

function scoreSet(p, y, idx) {
  let n = 0, hit = 0, ll = 0, br = 0;
  for (const i of idx) {
    const q = Math.min(Math.max(p[i], 1e-9), 1 - 1e-9);
    n++; hit += (q >= 0.5) === (y[i] === 1); ll -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q); br += (q - y[i]) ** 2;
  }
  return n ? { n, acc: hit / n, ll: ll / n, brier: br / n } : { n: 0 };
}
function bySeason(p, d) {
  const seasons = [...new Set(d.season)].sort((a, b) => a - b).slice(1);
  return seasons.map((s) => {
    const idx = d.season.flatMap((q, i) => (q === s && !Number.isNaN(p[i]) ? [i] : []));
    const mk = idx.filter((i) => d.market[i] != null);
    return { s, ...scoreSet(p, d.y, idx), mkt: mk.length ? scoreSet(d.market, d.y, mk) : null, mine_mk: mk.length ? scoreSet(p, d.y, mk) : null };
  });
}
function calib(p, d) {
  const bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0001], out = [];
  for (let b = 0; b < bins.length - 1; b++) {
    let n = 0, sp = 0, sh = 0;
    p.forEach((q, i) => {
      if (Number.isNaN(q)) return;
      const c = Math.max(q, 1 - q);
      if (c >= bins[b] && c < bins[b + 1]) { n++; sp += c; sh += q >= 0.5 ? d.y[i] : 1 - d.y[i]; }
    });
    if (n >= 10) out.push({ pred: sp / n, actual: sh / n, n, lo: bins[b], hi: bins[b + 1] });
  }
  return out;
}
const seasonLabel = (s) => `${String(s).slice(2)}-${String(s + 1).slice(2)}`;

/* ---------------------------------------------------------------- view */

function viewLab() {
  return `<section class="hero" style="margin-bottom:24px"><div><span class="pill"><span class="dot"></span>Model Lab</span>
    <h1 style="font-size:clamp(36px,5vw,56px)">Build your<br><em>own model.</em></h1>
    <p class="lede">Pick a model and its inputs. Your browser refits it on every season since 2011-12, replays the walk-forward backtest, and prices today's games, side by side with the default model and the betting market. Nothing leaves your device.</p></div>
    <div class="lock lab-summary" id="lab-summary"><div class="label">Your model</div><div class="lab-sum-body"><span class="mono" style="opacity:.7">Loading data…</span></div></div></section>
  <section class="lab">
    <aside class="card panel lab-controls" id="lab-controls"><div class="loading" style="height:200px"><span class="bounce"></span></div></aside>
    <div class="lab-results" id="lab-results"></div>
  </section>`;
}

function controlsHTML(cfg, d) {
  const models = [["logistic", "Logistic regression"], ["gbm", "Gradient boosting"], ["elo", "Elo only"]];
  const slider = (key, label, min, max, step, val, fmt) => `<label class="lab-slider"><span>${label} <b class="mono" id="v-${key}">${fmt(val)}</b></span>
    <input type="range" data-cfg="${key}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const logC = Math.log10(cfg.C);
  return `<h3>Model</h3>
    <div class="seg">${models.map(([k, l]) => `<button class="${cfg.model === k ? "on" : ""}" data-model="${k}">${l}</button>`).join("")}</div>
    ${cfg.model === "logistic" ? slider("logC", "Regularization C", -3, 1, 0.1, logC, (v) => (10 ** v).toPrecision(2)) +
      `<p class="lab-hint">Smaller C means a stronger L2 penalty, which shrinks coefficients toward zero. The default is 0.1.</p>` : ""}
    ${cfg.model === "gbm" ? slider("trees", "Trees", 25, 400, 25, cfg.trees, String) + slider("depth", "Depth", 1, 5, 1, cfg.depth, String)
      + slider("lr", "Learning rate", 0.01, 0.3, 0.01, cfg.lr, (v) => (+v).toFixed(2)) + slider("minLeaf", "Min games per leaf", 20, 400, 20, cfg.minLeaf, String)
      + `<p class="lab-hint">Histogram boosting with logistic loss and 32 quantile bins. A 14-season backtest takes a few seconds.</p>` : ""}
    ${cfg.model === "elo" ? `<p class="lab-hint">The Elo formula's own probability, with no fitting. A baseline.</p>` : ""}
    <h3 style="margin-top:18px">Inputs</h3>
    <div class="lab-groups ${cfg.model === "elo" ? "disabled" : ""}">${d.groups.map((g) => `<label class="lab-check">
      <input type="checkbox" data-group="${g.key}" ${cfg.groups.includes(g.key) ? "checked" : ""} ${cfg.model === "elo" ? "disabled" : ""}>
      <span><b>${esc(g.label)}</b>${g.default ? "" : ` <span class="tag">optional</span>`}<small>${esc(g.desc)}</small></span></label>`).join("")}
      ${d.unavailable.map((u) => `<label class="lab-check off" title="${esc(u.why)}"><input type="checkbox" disabled><span><b>${esc(u.label)}</b> <span class="tag">unavailable</span><small>${esc(u.why)}</small></span></label>`).join("")}
    </div>
    <div class="lab-actions">
      <button class="btn" id="lab-reset">Reset to default</button>
      <span class="lab-hint" style="margin:0;align-self:center">Updates as you change anything.</span>
    </div>
    <label class="lab-check use-today"><input type="checkbox" id="lab-use" ${cfg.useOnToday ? "checked" : ""}><span><b>Use my model on the Today page</b><small>Saved in this browser only.</small></span></label>`;
}

function labSummary(res, def) {
  const a = res.overall, b = def.overall, dl = a.ll - b.ll, da = a.acc - b.acc;
  return `<div class="label">${esc(describe(LAB.config))}</div>
    <div class="lab-sum-body">
      <div><b class="lab-big" data-count="${(a.acc * 100).toFixed(1)}" data-dec="1" data-suf="%">0</b><span class="mono">accuracy, ${a.n.toLocaleString()} games</span>
        <span class="mono ${da >= 0 ? "up" : "down"}">${da >= 0 ? "▲" : "▼"} ${(Math.abs(da) * 100).toFixed(2)} pts vs default</span></div>
      <div><b class="lab-big">${a.ll.toFixed(4)}</b><span class="mono">log loss (lower is better)</span>
        <span class="mono ${dl <= 0 ? "up" : "down"}">${dl <= 0 ? "▲ better" : "▼ worse"} by ${Math.abs(dl).toFixed(4)}</span></div>
    </div>`;
}

function labSeasonChart(mine, def) {
  const W = 760, H = 280, m = { l: 44, r: 14, t: 14, b: 30 }, lo = 0.58, hi = 0.74;
  const n = mine.length, x = (i) => m.l + (i + 0.5) * ((W - m.l - m.r) / n), y = (v) => m.t + (1 - (clamp(v, lo, hi) - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by season: your model, default model, market">`;
  for (let v = 0.6; v <= hi + 0.001; v += 0.04) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  const series = [["Market", mine.map((q) => q.mkt && q.mkt.acc), "var(--slate)"], ["Default", def.map((q) => q.acc), "var(--muted)"], ["Yours", mine.map((q) => q.acc), "var(--orange)"]];
  for (const [name, vals, col] of series) {
    const pts = vals.map((v, i) => (v == null ? null : [x(i), y(v)]));
    const d = pts.map((p, i) => (p ? `${pts[i - 1] ? "L" : "M"}${p[0]},${p[1]}` : "")).join("");
    s += `<path class="draw" d="${d}" fill="none" stroke="${col}" stroke-width="${name === "Yours" ? 2.5 : 1.8}" ${name === "Default" ? 'stroke-dasharray="5 4"' : ""}/>`;
    pts.forEach((p, i) => p && (s += `<circle cx="${p[0]}" cy="${p[1]}" r="${name === "Yours" ? 4.5 : 3.5}" fill="${col}" stroke="var(--card)" stroke-width="1.5"
      data-tip="<b>${seasonLabel(mine[i].s)}</b><br>Yours ${pct(mine[i].acc, 1)} · default ${pct(def[i].acc, 1)}${mine[i].mkt ? `<br>Market ${pct(mine[i].mkt.acc, 1)} (yours ${pct(mine[i].mine_mk.acc, 1)} on the same ${mine[i].mkt.n})` : ""}<br>Log loss ${mine[i].ll.toFixed(3)} vs ${def[i].ll.toFixed(3)}"/>`));
  }
  mine.forEach((q, i) => (s += `<text x="${x(i)}" y="${H - 10}" text-anchor="middle">${seasonLabel(q.s)}</text>`));
  return s + `</svg><div class="legend"><span><i style="background:var(--orange)"></i>Your model</span><span><i style="background:var(--muted)"></i>Default model</span><span><i style="background:var(--slate)"></i>Betting favorite (seasons with lines)</span></div>`;
}

function labExplain(res) {
  if (!res.explain.length) return `<p class="lab-hint">Elo has no fitted parameters: \(p = 1/(1 + 10^{-\Delta/400})\), where \(\Delta\) is the rating gap including home court.</p>`;
  const f = res.explain.slice().sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const max = Math.max(...f.map((d) => Math.abs(d.v))) || 1, gbm = res.kind === "gbm";
  return `<div class="lab-bars">${f.map((d) => `<div class="lab-bar"><span>${esc(FEATURE_LABELS[d.name] || d.name)}</span>
    <div class="track ${gbm ? "" : "div"}"><i style="${gbm ? "left:0" : d.v >= 0 ? "left:50%" : "right:50%"};background:${gbm || d.v >= 0 ? "var(--orange)" : "var(--slate)"}" data-w="${pct((Math.abs(d.v) / max) * (gbm ? 1 : 0.5), 1)}"></i></div>
    <span class="mono">${gbm ? pct(d.v / f.reduce((a, b) => a + b.v, 0), 1) : (d.v > 0 ? "+" : "") + d.v.toFixed(3)}</span></div>`).join("")}</div>
    <p class="lab-hint">${gbm ? "Share of total split gain, a rough importance measure." : "Standardized coefficients: change in log-odds per one-SD change, positive favoring the home team."}</p>`;
}

function labSlate(res, d) {
  if (!d.slate || !res.slate.length) return `<p class="lab-hint">No upcoming games.</p>`;
  const byKey = Object.fromEntries(D.slate.games.map((g) => [`${g.away.abbr}|${g.home.abbr}`, g]));
  return `<div class="table-wrap"><table><thead><tr><th>Game</th><th class="r">Default</th><th class="r">Yours</th><th class="r">Market</th><th class="r">Δ vs default</th></tr></thead><tbody>
    ${d.slate.home.map((h, i) => {
      const g = byKey[`${d.slate.away[i]}|${h}`], mine = res.slate[i], def = g ? g.p_default ?? g.p_home : null, mk = g && g.market ? g.market.p_home : null;
      const delta = def == null ? null : (mine - def) * 100;
      return `<tr><td>${esc(d.slate.away[i])} @ <b>${esc(h)}</b></td><td class="r mono">${def == null ? "—" : pct(def)}</td><td class="r mono"><b>${pct(mine)}</b></td>
        <td class="r mono">${mk == null ? "—" : pct(mk)}</td><td class="r mono" style="color:${delta == null || Math.abs(delta) < 0.5 ? "var(--muted)" : delta > 0 ? "var(--good)" : "var(--bad)"}">${delta == null ? "" : (delta > 0 ? "+" : "") + delta.toFixed(1)}</td></tr>`;
    }).join("")}</tbody></table></div><p class="lab-hint">Home-team win probability. Δ is your model minus the default.</p>`;
}

function labCalib(mine, def) {
  const W = 420, H = 300, m = { l: 40, r: 10, t: 10, b: 32 };
  const x = (v) => m.l + ((v - 0.5) / 0.5) * (W - m.l - m.r), y = (v) => m.t + (1 - (v - 0.4) / 0.6) * (H - m.t - m.b);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration of your model versus default">`;
  for (let v = 0.4; v <= 1.001; v += 0.1) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  for (let v = 0.5; v <= 1.001; v += 0.1) s += `<text x="${x(v)}" y="${H - 12}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
  s += `<line x1="${x(0.5)}" y1="${y(0.5)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  for (const [pts, col, name] of [[def, "var(--muted)", "Default"], [mine, "var(--orange)", "Yours"]]) {
    s += `<polyline points="${pts.map((p) => `${x(p.pred)},${y(clamp(p.actual, 0.4, 1))}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
    pts.forEach((p) => (s += `<circle class="pop" cx="${x(p.pred)}" cy="${y(clamp(p.actual, 0.4, 1))}" r="${clamp(Math.sqrt(p.n) * 0.25, 3.5, 10)}" fill="${col}" stroke="var(--card)" stroke-width="1.5" data-tip="<b>${name}</b><br>${Math.round(p.lo * 100)}–${Math.round(p.hi * 100)}%: predicted ${pct(p.pred, 1)}, won ${pct(p.actual, 1)} (${p.n})"/>`));
  }
  return s + `</svg><div class="legend"><span><i style="background:var(--orange)"></i>Yours</span><span><i style="background:var(--muted)"></i>Default</span></div>`;
}

function renderResults(res, def, d) {
  // a season needs a real sample to be worth plotting (the one in progress may have a handful of games)
  const full = (rows) => rows.filter((q) => q.n >= 200);
  const mine = full(bySeason(res.p, d)), dflt = full(bySeason(def.p, d));
  const idx = res.p.flatMap((q, i) => (Number.isNaN(q) ? [] : [i]));
  res.overall = scoreSet(res.p, d.y, idx);
  def.overall = scoreSet(def.p, d.y, idx);
  $("#lab-summary").innerHTML = labSummary(res, def);
  $("#lab-results").innerHTML = `
    <div class="card panel"><div class="panel-head"><h3>Accuracy by season</h3><span class="mono muted">walk-forward: each season fit on earlier seasons only</span></div>${labSeasonChart(mine, dflt)}</div>
    <div class="two" style="margin-top:14px">
      <div class="card panel"><div class="panel-head"><h3>${res.kind === "gbm" ? "Feature importance" : res.kind === "elo" ? "Parameters" : "Coefficients"}</h3></div>${labExplain(res)}</div>
      <div class="card panel"><div class="panel-head"><h3>Calibration</h3></div>${labCalib(calib(res.p, d), calib(def.p, d))}</div>
    </div>
    <div class="card panel" style="margin-top:14px"><div class="panel-head"><h3>Today's games, your model</h3></div>${labSlate(res, d)}</div>`;
  animateIn($("#lab-results"));
  animateIn($("#lab-summary"));
  if (window.renderMathInElement) renderMath($("#lab-results"));
}

/* every change reruns the backtest: changes during a run cancel it, and finished configurations are cached */
let LAB_RUN = 0;
function scheduleRun(delay) {
  clearTimeout(LAB.debounce);
  LAB.debounce = setTimeout(labRun, delay);
}
const cfgKey = (cfg) => JSON.stringify({ ...cfg, useOnToday: undefined, groups: cfg.model === "elo" ? [] : [...cfg.groups].sort(),
  ...(cfg.model === "gbm" ? { C: undefined } : { trees: undefined, depth: undefined, lr: undefined, minLeaf: undefined }) });

async function labRun() {
  const token = ++LAB_RUN, d = LAB.data, cfg = { ...LAB.config };
  saveConfig(LAB.config);
  LAB.results = LAB.results || {};
  const bar = (p, label) => token === LAB_RUN && ($("#lab-summary").innerHTML = `<div class="label">${esc(describe(cfg))}</div>
    <div class="lab-progress"><i style="width:${(p * 100).toFixed(0)}%"></i></div><span class="mono" style="opacity:.75">${label}</span>`);
  $("#lab-results").classList.add("stale");
  if (!LAB.defaultResult) {
    bar(0, "Fitting the default model…");
    const r = await runModel(defaultConfig(d), d, (m) => bar(m.progress * 0.3, `Default model: ${seasonLabel(m.season)}`));
    if (!r || token !== LAB_RUN) return;
    LAB.defaultResult = LAB.results[cfgKey(defaultConfig(d))] = r;
  }
  let res = LAB.results[cfgKey(cfg)];
  if (!res) {
    bar(0.3, "Backtesting your model…");
    res = await runModel(cfg, d, (m) => bar(0.3 + m.progress * 0.7, `Your model: backtesting ${seasonLabel(m.season)}`));
    if (!res || token !== LAB_RUN) return;
    LAB.results[cfgKey(cfg)] = res;
  }
  LAB.result = res;
  $("#lab-results").classList.remove("stale");
  renderResults({ ...res }, { ...LAB.defaultResult }, d);
}

async function mountLab() {
  const d = await labLoad();
  LAB.config = { ...defaultConfig(d), ...(savedConfig() || {}) };
  $("#lab-controls").innerHTML = controlsHTML(LAB.config, d);
  labRun();
}

function labControlsChanged() { $("#lab-controls").innerHTML = controlsHTML(LAB.config, LAB.data); }

document.addEventListener("click", (e) => {
  const t = e.target;
  if (!LAB.data || !t.closest("#lab-controls")) return;
  const m = t.closest("[data-model]");
  if (m) { LAB.config.model = m.dataset.model; labControlsChanged(); scheduleRun(0); return; }
  if (t.id === "lab-reset") { LAB.config = { ...defaultConfig(LAB.data), useOnToday: LAB.config.useOnToday }; labControlsChanged(); scheduleRun(0); }
});
document.addEventListener("input", (e) => {
  const t = e.target;
  if (!LAB.data || !t.closest || !t.closest("#lab-controls")) return;
  if (t.dataset.cfg) {
    const k = t.dataset.cfg, v = +t.value;
    if (k === "logC") { LAB.config.C = +(10 ** v).toPrecision(2); $("#v-logC").textContent = LAB.config.C; }
    else { LAB.config[k] = v; $(`#v-${k}`).textContent = k === "lr" ? v.toFixed(2) : v; }
    scheduleRun(400);                    // wait for the slider to settle
  }
  if (t.dataset.group) {
    const g = t.dataset.group;
    LAB.config.groups = t.checked ? [...new Set([...LAB.config.groups, g])] : LAB.config.groups.filter((x) => x !== g);
    scheduleRun(150);
  }
  if (t.id === "lab-use") { LAB.config.useOnToday = t.checked; saveConfig(LAB.config); }
});

/* Today page: swap in the saved custom model's probabilities */
async function applyCustomToToday() {
  const cfg = savedConfig();
  if (!cfg || !cfg.useOnToday || !D.slate.games.length) return null;
  const d = await labLoad();
  if (isDefault({ ...defaultConfig(d), ...cfg }, d)) return null;
  const res = await runModel({ ...defaultConfig(d), ...cfg }, { ...d, season: d.season.map(() => 0) }, null);   // no backtest needed: one fit on all games
  const byKey = Object.fromEntries(d.slate.home.map((h, i) => [`${d.slate.away[i]}|${h}`, res.slate[i]]));
  for (const g of D.slate.games) {
    const p = byKey[`${g.away.abbr}|${g.home.abbr}`];
    if (p == null) continue;
    g.p_default = g.p_home;
    g.p_model = p;                       // your model replaces the default model; the market blend still applies
    republish(g);
  }
  return cfg;
}
