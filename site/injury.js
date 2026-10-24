/* Injury news, every 5 minutes: ESPN's injury feed straight from the browser, plus the official
   NBA injury report as last fetched by the site (data/injuries.json, refreshed every 15 minutes in
   the evening). The official report wins where both list a player. Any status that differs from
   what the published pick assumed is applied to games that haven't started, which are re-priced
   with the model's own lineup math (scratch.js) and re-blended with the market. */
"use strict";

const INJ = { timer: null, changes: [], seen: {}, lastCheck: null, reportTime: null };
const ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries";

function nameKey(name) {
  return String(name).normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/-/g, " ")
    .replace(/[^a-z ]/g, "").split(" ").filter((w) => w && !["jr", "sr", "ii", "iii", "iv"].includes(w)).join("");
}

async function pollInjuries() {
  if (!D.slate || !D.slate.games.length || document.hidden) return scheduleInj(60);
  const status = {};
  try {
    const off = await fetch(`data/injuries.json?_=${Date.now()}`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const es = await fetch(`${ESPN_INJURIES}?_=${Date.now()}`).then((r) => r.json());
    for (const t of es.injuries || []) for (const i of t.injuries || []) {
      status[nameKey(i.athlete.displayName)] = { status: i.status, source: "ESPN", reason: (i.shortComment || "").slice(0, 160) };
    }
    if (off) {
      INJ.reportTime = off.report_time;
      for (const p of off.players) if (p.source !== "ESPN") status[p.key] = { status: p.status, source: "NBA injury report", reason: p.reason };
    }
  } catch (err) {
    console.warn("injury feed unavailable", err);
    return scheduleInj(300);
  }
  INJ.lastCheck = new Date();
  let changed = false;
  for (const g of D.slate.games) {
    if (g.live && g.live.state !== "pre") continue;                   // too late to matter
    for (const side of [g.away, g.home]) for (const p of side.rotation) {
      const s = status[nameKey(p.name)];
      const now = !s || s.status === "Available" ? "Healthy" : s.status;
      const was = p.status_live || p.status;
      if (now === p.status) { if (p.status_live) { delete p.status_live; changed = true; } continue; }
      if (now !== was) {
        p.status_live = now;
        changed = true;
        const id = `${p.name}|${now}`;
        if (!INJ.seen[id]) {
          INJ.seen[id] = true;
          INJ.changes.unshift({ at: new Date(), game: g, team: side.abbr, name: p.name, from: p.status, to: now, source: s ? s.source : "ESPN", reason: s ? s.reason : "" });
        }
      }
    }
  }
  if (changed) {
    applyScratches();
    if (isToday(viewDay())) rerenderGames();
  }
  paintNews();
  scheduleInj(D.slate.games.some((g) => !g.live || g.live.state === "pre") ? 300 : 0);
}

function scheduleInj(secs) {
  clearTimeout(INJ.timer);
  if (secs) INJ.timer = setTimeout(pollInjuries, secs * 1000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && INJ.lastCheck && Date.now() - INJ.lastCheck > 300e3) pollInjuries(); });

const timeET = (d) => d.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" });

/* on each card: what changed since the pick was published */
function injuryNote(g) {
  const ch = INJ.changes.filter((c) => c.game === g);
  if (!ch.length) return "";
  return `<div class="inj-note"><span class="inj-bolt">⚡</span>${ch.map((c) => `<span><b>${esc(c.name)}</b> ${esc(c.from === "Healthy" ? "" : c.from + " → ")}<b class="st-${esc(c.to)}">${esc(c.to === "Healthy" ? "available" : c.to.toLowerCase())}</b> <span class="muted">${timeET(c.at)} · ${esc(c.source)}</span></span>`).join("")}</div>`;
}

/* the feed on the Today page */
function paintNews() {
  const el = document.getElementById("inj-news");
  if (!el) return;
  const items = INJ.changes.slice(0, 8);
  el.innerHTML = `<div class="news-head"><b>Lineup news</b><span class="mono muted">${INJ.lastCheck ? `checked ${timeET(INJ.lastCheck)}` : "checking…"}${INJ.reportTime ? ` · NBA report ${timeET(new Date(INJ.reportTime))}` : ""} · every 5 min</span></div>
    ${items.length ? `<div class="news-list">${items.map((c) => `<a class="news-item" href="#/game/${c.game.id}">
      <span class="mono muted">${timeET(c.at)}</span><span>${badge(c.team, teamColor(c.team))}</span>
      <span><b>${esc(c.name)}</b> ${esc(c.from === "Healthy" ? "" : c.from.toLowerCase() + " → ")}<b class="st-${esc(c.to)}">${esc(c.to === "Healthy" ? "available" : c.to.toLowerCase())}</b>
      <span class="muted"> · ${esc(c.game.away.abbr)} @ ${esc(c.game.home.abbr)} now ${esc(c.game.pick)} ${pct(Math.max(c.game.p_home, 1 - c.game.p_home))}</span></span></a>`).join("")}</div>`
      : `<p class="mono muted" style="margin:6px 0 0">No changes since the picks were published.</p>`}`;
}
