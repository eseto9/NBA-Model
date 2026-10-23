"""Live NBA moneylines from The Odds API (https://the-odds-api.com), paced for the free plan.

Each call costs 1 credit (US region, moneyline market only) and returns every upcoming game.
The free plan allows 500 credits a month, so calls are skipped when:
  - no game tips off in the next 24 hours, or
  - the credits left would not cover CALLS_PER_DAY for the rest of the month.

Point spreads and totals come free from ESPN's scoreboard on every run (display only).
Every call is appended to log/odds.csv (one row per game), which gives line movement and
the closing line (last snapshot before tipoff). site/data/odds.json is rebuilt from it.

  ODDS_API_KEY=... python odds.py [--force]
"""
import argparse
import calendar
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import espn

ROOT = Path(__file__).parent
API = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ET = ZoneInfo("America/New_York")
MONTHLY_CREDITS = 500
CALLS_PER_DAY = 15        # the schedule in .github/workflows/odds.yml: 14 hourly runs + the morning run
FLOOR = 10                # never spend the last few credits
STATE = ROOT / "log" / "odds_state.json"
SNAPSHOTS = ROOT / "log" / "odds.csv"
OUT = ROOT / "site" / "data" / "odds.json"
NICKNAMES = {  # The Odds API uses full names ("Los Angeles Clippers"); match on the nickname
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA", "Bulls": "CHI", "Cavaliers": "CLE",
    "Mavericks": "DAL", "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND",
    "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
    "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI", "Suns": "PHX",
    "Blazers": "POR", "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS",
}


def implied(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def no_vig(home_ml, away_ml):
    """Home win probability with the bookmaker's margin removed."""
    h, a = implied(home_ml), implied(away_ml)
    return h / (h + a)


def american(p):
    """Fair American odds for a probability."""
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


# ---------------------------------------------------------------- budget

def load_state(now):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    if s.get("month") != f"{now:%Y-%m}":           # credits reset each month
        s = {"month": f"{now:%Y-%m}", "remaining": MONTHLY_CREDITS, "calls": 0}
    return s


def budget_ok(state, now):
    days_after_today = calendar.monthrange(now.year, now.month)[1] - now.day
    reserve = max(FLOOR, CALLS_PER_DAY * days_after_today)
    return state["remaining"] - 1 >= reserve, reserve


def games_soon(now, hours=24):
    """True if any known game tips off within `hours` (from the published slate and odds)."""
    tips = []
    slate = ROOT / "site" / "data" / "slate.json"
    if slate.exists():
        for g in json.loads(slate.read_text(encoding="utf-8")).get("games", []):
            if g.get("tip"):
                tips.append(dt.datetime.fromisoformat(g["tip"]).replace(tzinfo=ET))
    if SNAPSHOTS.exists():
        s = pd.read_csv(SNAPSHOTS)
        tips += [dt.datetime.fromisoformat(c.replace("Z", "+00:00")) for c in s.commence.unique()]
    return any(now <= t <= now + dt.timedelta(hours=hours) for t in tips) or not tips


# ---------------------------------------------------------------- fetch

def fetch(key, now):
    r = requests.get(API, timeout=30, params={
        "apiKey": key, "regions": "us", "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso",
        "commenceTimeFrom": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    r.raise_for_status()
    remaining = int(float(r.headers.get("x-requests-remaining", 0)))
    rows = []
    for ev in r.json():
        home, away = NICKNAMES.get(ev["home_team"].split()[-1]), NICKNAMES.get(ev["away_team"].split()[-1])
        if not home or not away:
            continue
        lines = []
        for b in ev.get("bookmakers", []):
            m = next((m for m in b.get("markets", []) if m["key"] == "h2h"), None)
            if not m:
                continue
            price = {o["name"]: o["price"] for o in m["outcomes"]}
            if ev["home_team"] in price and ev["away_team"] in price:
                lines.append((price[ev["home_team"]], price[ev["away_team"]]))
        if not lines:
            continue
        commence = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        hm = pd.Series([h for h, _ in lines])
        am = pd.Series([a for _, a in lines])
        rows.append({
            "fetched_at": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_id": ev["id"], "commence": ev["commence_time"], "date": f"{commence.astimezone(ET):%Y-%m-%d}",
            "home": home, "away": away, "books": len(lines),
            "p_home": round(sum(no_vig(h, a) for h, a in lines) / len(lines), 4),
            "home_ml": int(hm.median()), "away_ml": int(am.median()),
            "home_best": int(hm.max()), "away_best": int(am.max()),
        })
    return rows, remaining


def spreads(now):
    """Point spread and total from ESPN's scoreboard for today and tomorrow. Free, so refreshed on
    every run. Display only: the model never sees them."""
    out = {}
    today = now.astimezone(ET).date()
    for day in (today, today + dt.timedelta(days=1)):
        for e in espn.scoreboard(day):
            odds_ = (e["competitions"][0].get("odds") or [None])[0]
            if not odds_ or odds_.get("spread") is None:
                continue
            sides = {c["homeAway"]: espn.nba_abbr(c["team"]["abbreviation"]) for c in e["competitions"][0]["competitors"]}
            out[f"{espn.et_day(e['date']):%Y-%m-%d}|{sides['away']}|{sides['home']}"] = {
                "home_spread": odds_["spread"], "text": odds_.get("details"), "total": odds_.get("overUnder"),
                "provider": odds_.get("provider", {}).get("name")}
    return out


# ---------------------------------------------------------------- outputs

def closing_lines(snap=None):
    """Last pre-tipoff snapshot for each game: DataFrame keyed by date/home/away."""
    snap = pd.read_csv(SNAPSHOTS) if snap is None and SNAPSHOTS.exists() else snap
    if snap is None or snap.empty:
        return pd.DataFrame(columns=["date", "home", "away", "p_home", "home_ml", "away_ml"])
    pre = snap[snap.fetched_at < snap.commence.str.replace(".000", "", regex=False)]
    return pre.sort_values("fetched_at").groupby(["date", "home", "away"]).tail(1)


def write_site(now, remaining, lines):
    snap = pd.read_csv(SNAPSHOTS) if SNAPSHOTS.exists() else pd.DataFrame(columns=["date", "home", "away", "fetched_at", "commence"])
    keep = snap[snap.date >= f"{(now.astimezone(ET) - dt.timedelta(days=1)):%Y-%m-%d}"]
    games = {}
    for (date, home, away), s in keep.sort_values("fetched_at").groupby(["date", "home", "away"]):
        pre = s[s.fetched_at < s.commence]
        last = (pre if len(pre) else s).iloc[-1]
        games[f"{date}|{away}|{home}"] = {
            "date": date, "home": home, "away": away, "commence": last.commence, "books": int(last.books),
            "p_home": float(last.p_home), "home_ml": int(last.home_ml), "away_ml": int(last.away_ml),
            "home_best": int(last.home_best), "away_best": int(last.away_best),
            "updated": last.fetched_at, "closed": bool(len(pre) < len(s)),
            "history": [[t, float(p)] for t, p in zip(pre.fetched_at, pre.p_home)],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"fetched_at": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "credits_remaining": remaining, "games": games, "spreads": lines}, separators=(",", ":")), encoding="utf-8")


def main(force=False):
    now = dt.datetime.now(dt.timezone.utc)
    state = load_state(now)
    try:
        lines = spreads(now)
    except Exception as e:
        print(f"Spreads unavailable: {e}")
        lines = {}
    key = os.environ.get("ODDS_API_KEY")
    ok, reserve = budget_ok(state, now)
    if not key:
        print("ODDS_API_KEY not set; spreads only")
    elif not force and not ok:
        print(f"Skipping moneylines: {state['remaining']} credits left, keeping {reserve} for the rest of the month")
    elif not force and not games_soon(now):
        print("Skipping moneylines: no games in the next 24 hours")
    else:
        rows, remaining = fetch(key, now)
        SNAPSHOTS.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(SNAPSHOTS, mode="a", header=not SNAPSHOTS.exists(), index=False)
        state.update(remaining=remaining, calls=state["calls"] + 1, last_call=f"{now:%Y-%m-%dT%H:%MZ}")
        STATE.write_text(json.dumps(state))
        print(f"Moneylines for {len(rows)} games; {remaining} credits left this month")
    write_site(now, state["remaining"], lines)
    print(f"Spreads for {len(lines)} games")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="ignore the budget and schedule checks")
    sys.exit(main(ap.parse_args().force))
