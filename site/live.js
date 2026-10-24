/* Live scores: the page polls ESPN's public scoreboard directly (it allows cross-origin requests),
   every 30 seconds while a game is on or about to tip, every 5 minutes otherwise, and not at all
   while the tab is hidden. Cards update in place, so bars slide rather than redraw.

   Live win probability treats the rest of the game as Brownian motion (Stern 1994): the final
   margin is normal around the current lead plus the pregame expected margin scaled by the time
   left, with variance shrinking in proportion to the time left. */
"use strict";

const LIVE = { timer: null, games: {}, fetchedAt: null, history: {} };
const SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard";
const ESPN_ABBR = { GS: "GSW", NY: "NYK", SA: "SAS", NO: "NOP", UTAH: "UTA", WSH: "WAS", NJ: "BKN" };

function normCdf(x) {             // Abramowitz-Stegun 7.1.26 via erf, |error| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(x * x) / 2);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/* P(home wins | lead, fraction of regulation left), from the pregame probability */
function liveProb(pPre, lead, tau) {
  const mu0 = MARGIN_SD * normInv(pPre);              // pregame expected home margin
  if (tau <= 1e-4) return lead > 0 ? 1 : lead < 0 ? 0 : 0.5;
  return normCdf((lead + mu0 * tau) / (MARGIN_SD * Math.sqrt(tau)));
}

function secondsLeft(period, clock) {
  const [m, s] = String(clock || "0:00").includes(":") ? clock.split(":").map(Number) : [0, Number(clock) || 0];
  const inPeriod = m * 60 + s;
  if (period <= 4) return (4 - period) * 720 + inPeriod;
  return inPeriod;                                   // overtime: only this period's clock remains
}

function elapsed(period, clock) {
  return period <= 4 ? 2880 - secondsLeft(period, clock) : 2880 + 300 * (period - 5) + (300 - secondsLeft(period, clock));
}

function parseEvent(e) {
  const comp = e.competitions[0], st = e.status, type = st.type;
  const side = Object.fromEntries(comp.competitors.map((c) => [c.homeAway, c]));
  const abbr = (c) => ESPN_ABBR[c.team.abbreviation] || c.team.abbreviation;
  return {
    id: e.id, state: type.state,          // pre / in / post
    detail: type.shortDetail, period: st.period, clock: st.displayClock,
    home: abbr(side.home), away: abbr(side.away),
    hs: +side.home.score || 0, as: +side.away.score || 0,
    last: comp.situation && comp.situation.lastPlay ? comp.situation.lastPlay.text : "",
  };
}

async function pollLive() {
  if (!D.slate || !D.slate.games.length || document.hidden) return schedule(60);
  const day = D.slate.date.replaceAll("-", "");
  try {
    const r = await fetch(`${SCOREBOARD}?dates=${day}&_=${Date.now()}`);
    const events = (await r.json()).events || [];
    LIVE.fetchedAt = new Date();
    for (const e of events) {
      const x = parseEvent(e), g = D.slate.games.find((q) => q.id.endsWith(`E${e.id}`) || (q.home.abbr === x.home && q.away.abbr === x.away));
      if (!g) continue;
      const pre = g.p_home;                // pregame number, including any scratches you entered
      if (x.state === "in") {
        x.p = liveProb(pre, x.hs - x.as, Math.min(1, secondsLeft(x.period, x.clock) / 2880));
      } else if (x.state === "post") {
        x.p = x.hs > x.as ? 1 : 0;
      } else {
        x.p = pre;
      }
      const h = (LIVE.history[g.id] = LIVE.history[g.id] || []);
      if (x.state !== "pre" && (!h.length || h[h.length - 1][1] !== x.p)) h.push([elapsed(x.period, x.clock), x.p]);
      g.live = x;
      paintLive(g);
    }
    paintTonight();
    tickerLive();
  } catch (err) {
    console.warn("live scores unavailable", err);
  }
  const anyLive = D.slate.games.some((g) => g.live && g.live.state === "in");
  const soon = D.slate.games.some((g) => (!g.live || g.live.state === "pre") && g.tip && etDate(g.tip) - Date.now() < 20 * 60e3);
  const allDone = D.slate.games.every((g) => g.live && g.live.state === "post");
  schedule(allDone ? 0 : anyLive || soon ? 30 : 300);
}

function schedule(secs) {
  clearTimeout(LIVE.timer);
  if (secs) LIVE.timer = setTimeout(pollLive, secs * 1000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) pollLive(); });

/* ---------------------------------------------------------------- painting */

function liveBlock(g) {
  const x = g.live;
  if (!x || x.state === "pre") return "";
  const hit = x.state === "post" ? ((x.hs > x.as) === (g.p_home >= 0.5)) : null;
  const leadA = x.as > x.hs, leadH = x.hs > x.as;
  return `<div class="live ${x.state} ${hit == null ? "" : hit ? "call-hit" : "call-miss"}">
    <span class="live-tag">${x.state === "in" ? `<i class="live-dot"></i>LIVE · ${esc(x.detail)}` : `FINAL${x.period > 4 ? ` / ${x.period - 4 > 1 ? x.period - 4 : ""}OT` : ""}`}</span>
    <span class="live-score"><b class="${leadA ? "lead" : ""}">${esc(g.away.abbr)} ${x.as}</b><span class="muted">–</span><b class="${leadH ? "lead" : ""}">${x.hs} ${esc(g.home.abbr)}</b></span>
    ${hit == null ? `<span class="mono live-pre">pregame ${esc(g.p_home >= 0.5 ? g.home.abbr : g.away.abbr)} ${pct(Math.max(g.p_home, 1 - g.p_home))}</span>`
      : `<span class="verdict ${hit ? "v-hit" : "v-miss"}"><i>${hit ? "✓" : "✗"}</i>${hit ? "Called it" : "Missed"}</span>`}
    ${x.state === "post" ? finalCompare(g) : liveSpark(g)}
  </div>`;
}

function finalCompare(g) {
  const homeWon = g.live.hs > g.live.as;
  const pick = (p) => `${esc(p >= 0.5 ? g.home.abbr : g.away.abbr)} ${pct(Math.max(p, 1 - p))} ${(p >= 0.5) === homeWon ? "✓" : "✗"}`;
  return `<div class="final-compare mono"><span>Model <b>${pick(g.p_home)}</b></span>${g.market ? `<span>Market <b>${pick(g.market.p_home)}</b></span>` : ""}</div>`;
}

function liveSpark(g) {
  const h = LIVE.history[g.id] || [];
  if (h.length < 4) return "";                 // not until there's a trend to show
  const W = 220, H = 28, end = Math.max(2880, h[h.length - 1][0]);
  const pts = h.map(([t, p]) => `${(t / end) * W},${H - p * H}`).join(" ");
  return `<svg class="live-spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-label="Home win probability during the game">
    <line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--line)" stroke-dasharray="2 3"/>
    <polyline points="${pts}" fill="none" stroke="${esc(g.home.color)}" stroke-width="1.8"/></svg>`;
}

/* update one card in place: score block, and slide the bars to the live probability */
function paintLive(g) {
  if (typeof viewDay === "function" && !isToday(viewDay())) return;
  const card = document.querySelector(`[data-game="${g.id}"]`);
  if (!card) return;
  const slot = card.querySelector(".live-slot");
  if (slot) slot.innerHTML = liveBlock(g);
  if (!g.live || g.live.state === "pre" || g.live.p == null) return;
  // once a game is over the bars go back to what was predicted; the score panel shows what happened
  const p = g.live.state === "post" ? g.p_home : g.live.p, bars = card.querySelectorAll(".tug-bar i"), nums = card.querySelectorAll(".tug-nums span:not(.tug-cap)");
  if (bars.length === 2) {
    bars[0].dataset.flex = bars[0].style.flexBasis = `calc(${pct(1 - p, 1)} - 1px)`;
    bars[1].dataset.flex = bars[1].style.flexBasis = `calc(${pct(p, 1)} - 1px)`;
  }
  // not 100% until the buzzer
  const show = (q) => (g.live.state === "post" ? pct(q) : q > 0.995 ? ">99%" : q < 0.005 ? "<1%" : pct(q));
  const cap = card.querySelector(".tug-cap");
  if (cap) cap.textContent = g.live.state === "post" ? "pregame" : "live";
  if (nums.length === 2) {
    nums[0].textContent = show(1 - p); nums[0].className = p > 0.5 ? "lose" : "";
    nums[1].textContent = show(p); nums[1].className = p < 0.5 ? "lose" : "";
  }
  card.classList.toggle("is-live", g.live.state === "in");
  card.classList.toggle("is-final", g.live.state === "post");
  const hit = g.live.state === "post" ? (g.live.hs > g.live.as) === (g.p_home >= 0.5) : null;
  card.classList.toggle("call-hit", hit === true);
  card.classList.toggle("call-miss", hit === false);
}

function paintTonight() {
  const rec = typeof seasonRecord === "function" ? seasonRecord() : null, recEl = document.getElementById("season-rec");
  if (rec && recEl) {
    recEl.textContent = `${rec.hit}–${rec.n - rec.hit}`;
    document.getElementById("season-pct").textContent = rec.n ? pct(rec.hit / rec.n, 1) : "—";
  }
  const split = document.getElementById("split-wrap");
  if (split && typeof splitPanel === "function" && isToday(viewDay())) split.innerHTML = splitPanel(D.slate.games);
  const el = document.getElementById("tonight");
  if (!el) return;
  const games = D.slate.games, live = games.filter((g) => g.live && g.live.state === "in").length;
  const done = games.filter((g) => g.live && g.live.state === "post");
  const hits = done.filter((g) => (g.live.hs > g.live.as) === (g.p_home >= 0.5)).length;
  el.innerHTML = !live && !done.length ? "" : `${live ? `<span><i class="live-dot"></i>${live} live</span>` : ""}
    ${done.length ? `<span>${done.length} final · picks <b>${hits}–${done.length - hits}</b> tonight</span>` : ""}
    <span class="muted">updated ${LIVE.fetchedAt ? LIVE.fetchedAt.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" }) : ""}</span>`;
}

function tickerLive() {
  const t = document.getElementById("ticker");
  if (!t) return;
  const items = D.slate.games.map((g) => {
    const x = g.live;
    if (!x || x.state === "pre") return `<span>${esc(g.away.abbr)} @ ${esc(g.home.abbr)} <b>${esc(g.pick)} ${pct(confidence(g))}</b> · ${tipLabel(g.tip)}</span>`;
    return `<span>${x.state === "in" ? "● " : ""}${esc(g.away.abbr)} ${x.as} – ${x.hs} ${esc(g.home.abbr)} <b>${esc(x.detail)}</b></span>`;
  });
  t.innerHTML = items.join("");
}

/* after any re-render, put live state back on the fresh cards */
function repaintLive() {
  if (!D.slate) return;
  for (const g of D.slate.games) if (g.live) paintLive(g);
  paintTonight();
}
