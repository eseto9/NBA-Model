/* Day navigation on the Today page: any day this season and the week ahead.
   Past days show the card that was published before tipoff (site/data/days/<date>.json) with
   final scores and results from the season record; days from before the site launched are
   rebuilt from the record alone (backfilled picks). Future days are an early look. */
"use strict";

const DAYS = { cache: {}, loading: {} };

const viewDay = () => state.day || D.slate.date;
const isToday = (d) => d === D.slate.date;

/* games for a day, or null while its file loads (render() runs again when it arrives) */
function dayGames(day) {
  if (isToday(day)) return D.slate.games;
  if (DAYS.cache[day]) return DAYS.cache[day];
  if (!DAYS.loading[day] && !(D.meta.day_files || []).includes(day)) {      // before the site published cards
    DAYS.loading[day] = Promise.resolve();
    DAYS.cache[day] = withResults(liteGames(day), day);
    return DAYS.cache[day];
  }
  if (!DAYS.loading[day]) {
    DAYS.loading[day] = fetch(`data/days/${day}.json?v=${encodeURIComponent(D.meta.generated_at)}`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((j) => {
        DAYS.cache[day] = withResults(j ? j.games : liteGames(day), day);
        if (viewDay() === day) render();
      });
  }
  return null;
}

/* a pick from the season record as a card, for days before the site published cards */
function liteGames(day) {
  return D.record.picks.filter((p) => p.date === day).map((p) => {
    const side = (abbr) => ({ abbr, name: D.teamBy[abbr] ? D.teamBy[abbr].name : abbr, color: teamColor(abbr), rotation: [], lineup: {} });
    return { id: `lite-${day}-${p.home}`, date: day, tip: null, lite: true, neutral: false, p_home: p.p_home, p_elo: p.p_home,
      pick: p.p_home >= 0.5 ? p.home : p.away, steps: [], home: side(p.home), away: side(p.away) };
  });
}

/* final scores and the closing line from the season record; tomorrow's line from the odds feed */
function withResults(games, day) {
  const picks = Object.fromEntries(D.record.picks.filter((p) => p.date === day).map((p) => [`${p.away}|${p.home}`, p]));
  for (const g of games) {
    const p = picks[`${g.away.abbr}|${g.home.abbr}`];
    g.date = g.date || day;
    if (p && p.p_home != null) { g.p_home = p.p_home; g.pick = p.p_home >= 0.5 ? g.home.abbr : g.away.abbr; g.source = p.source; }
    if (p && p.score) g.live = { state: "post", as: p.score[0], hs: p.score[1], period: 4, detail: "Final" };
    const odds = (D.odds.games || {})[`${day}|${g.away.abbr}|${g.home.abbr}`];
    if (p && p.p_market != null) g.market = { p_home: p.p_market, home_ml: p.ml ? p.ml[1] : null, away_ml: p.ml ? p.ml[0] : null,
      home_best: p.ml ? p.ml[1] : null, away_best: p.ml ? p.ml[0] : null, books: 1, history: [], provider: "closing line" };
    else if (odds) g.market = odds;
    g.spread = (D.odds.spreads || {})[`${day}|${g.away.abbr}|${g.home.abbr}`] || null;
  }
  return games;
}

function dayRecord(day) {
  const ps = D.record.picks.filter((p) => p.date === day && p.score);
  return ps.length ? [ps.filter(isHit).length, ps.length] : null;
}

function dayStrip() {
  const days = D.meta.days || [D.slate.date], cur = viewDay(), i = days.indexOf(cur);
  const chip = (d) => {
    const rec = dayRecord(d), today = isToday(d), past = d < D.slate.date;
    const sub = today ? "today" : rec ? `${rec[0]}–${rec[1] - rec[0]}` : past ? "" : "upcoming";
    const [Y, M, Dd] = d.split("-").map(Number), dt = new Date(Y, M - 1, Dd);
    return `<a class="day-chip ${d === cur ? "on" : ""} ${today ? "today" : ""} ${rec ? (rec[0] / rec[1] >= 0.5 ? "good" : "bad") : ""}" href="#/day/${d}">
      <span class="dc-wd">${dt.toLocaleDateString("en-US", { weekday: "short" })}</span><b>${dt.toLocaleDateString("en-US", { month: "short", day: "numeric" })}</b><span class="dc-sub">${sub}</span></a>`;
  };
  return `<div class="day-nav">
    <a class="day-arrow ${i <= 0 ? "off" : ""}" href="#/day/${days[Math.max(0, i - 1)]}" aria-label="Previous day">‹</a>
    <div class="day-strip" id="day-strip">${days.map(chip).join("")}</div>
    <a class="day-arrow ${i >= days.length - 1 ? "off" : ""}" href="#/day/${days[Math.min(days.length - 1, i + 1)]}" aria-label="Next day">›</a>
  </div>`;
}

function dayHeading(day, games) {
  const rec = dayRecord(day);
  if (isToday(day)) return "";
  if (rec) return `<span class="day-note">${rec[0]}–${rec[1] - rec[0]} on picks${games && games.some((g) => g.lite) ? " · backfilled: predicted afterwards from pre-game data, before the site launched" : ""}</span>`;
  if (day > D.slate.date) return `<span class="day-note">Early look: injury reports and lines will change before tipoff. Final picks publish that morning.</span>`;
  return "";
}

/* keep the selected chip in view */
function centerDayStrip() {
  const strip = document.getElementById("day-strip"), on = strip && strip.querySelector(".on");
  if (on) strip.scrollLeft = on.offsetLeft - strip.clientWidth / 2 + on.clientWidth / 2;
}
