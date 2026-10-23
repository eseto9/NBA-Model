"""Live NBA betting lines from ESPN's public scoreboard (DraftKings): moneyline, spread, total.

Free and keyless, so it can poll as often as the schedule in .github/workflows/odds.yml runs.
Each run snapshots every not-yet-started game for today and tomorrow (US Eastern) into
log/odds.csv, which gives line movement and the closing line (last snapshot before tipoff).
site/data/odds.json is rebuilt from the snapshots. Spreads and totals are display only.

  python odds.py
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import espn

ROOT = Path(__file__).parent
ET = ZoneInfo("America/New_York")
SNAPSHOTS = ROOT / "log" / "odds.csv"
OUT = ROOT / "site" / "data" / "odds.json"


def implied(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def no_vig(home_ml, away_ml):
    """Home win probability with the bookmaker's margin removed."""
    h, a = implied(home_ml), implied(away_ml)
    return h / (h + a)


def american(text):
    """ESPN odds strings ("-265", "+215", "EVEN") as numbers; None when off the board."""
    text = str(text or "").strip().upper()
    if text == "EVEN":
        return 100
    try:
        return int(float(text))
    except ValueError:
        return None


def fetch(now):
    """(moneyline snapshot rows, spread/total lines) for games today and tomorrow."""
    rows, lines = [], {}
    today = now.astimezone(ET).date()
    for day in (today, today + dt.timedelta(days=1)):
        for e in espn.scoreboard(day):
            comp = e["competitions"][0]
            book = (comp.get("odds") or [None])[0]
            if not book or not espn.game_id(e):
                continue
            sides = {c["homeAway"]: espn.nba_abbr(c["team"]["abbreviation"]) for c in comp["competitors"]}
            date = f"{espn.et_day(e['date']):%Y-%m-%d}"
            key = f"{date}|{sides['away']}|{sides['home']}"
            if book.get("spread") is not None:
                lines[key] = {"home_spread": book["spread"], "text": book.get("details"), "total": book.get("overUnder"),
                              "provider": book.get("provider", {}).get("name")}
            mlb = book.get("moneyline") or {}
            h = american(mlb.get("home", {}).get("close", {}).get("odds"))
            a = american(mlb.get("away", {}).get("close", {}).get("odds"))
            if h is None or a is None or e["status"]["type"]["name"] != "STATUS_SCHEDULED":
                continue                    # no line, or already under way: closing line is already recorded
            rows.append({
                "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "event_id": e["id"], "commence": e["date"].replace("Z", ":00Z")
                if len(e["date"]) == 17 else e["date"], "date": date, "home": sides["home"], "away": sides["away"],
                "books": 1, "p_home": round(no_vig(h, a), 4), "home_ml": h, "away_ml": a, "home_best": h, "away_best": a,
                "home_open": american(mlb.get("home", {}).get("open", {}).get("odds")),
                "away_open": american(mlb.get("away", {}).get("open", {}).get("odds")),
                "provider": book.get("provider", {}).get("name"),
            })
    return rows, lines


# ---------------------------------------------------------------- outputs

def closing_lines(snap=None):
    """Last pre-tipoff snapshot for each game: DataFrame keyed by date/home/away."""
    snap = pd.read_csv(SNAPSHOTS) if snap is None and SNAPSHOTS.exists() else snap
    if snap is None or snap.empty:
        return pd.DataFrame(columns=["date", "home", "away", "p_home", "home_ml", "away_ml"])
    pre = snap[snap.fetched_at < snap.commence.str.replace(".000", "", regex=False)]
    return pre.sort_values("fetched_at").groupby(["date", "home", "away"]).tail(1)


def write_site(now, lines):
    snap = pd.read_csv(SNAPSHOTS) if SNAPSHOTS.exists() else pd.DataFrame(columns=["date", "home", "away", "fetched_at", "commence"])
    keep = snap[snap.date >= f"{(now.astimezone(ET) - dt.timedelta(days=1)):%Y-%m-%d}"]
    games = {}
    for (date, home, away), s in keep.sort_values("fetched_at").groupby(["date", "home", "away"]):
        pre = s[s.fetched_at < s.commence]
        last = (pre if len(pre) else s).iloc[-1]
        opened = s.dropna(subset=["home_open"]) if "home_open" in s else s.iloc[:0]
        opener = opened.iloc[0] if len(opened) else s.iloc[0]
        games[f"{date}|{away}|{home}"] = {
            "date": date, "home": home, "away": away, "commence": last.commence, "books": int(last.books),
            "provider": last.get("provider") if isinstance(last.get("provider"), str) else "DraftKings",
            "p_home": float(last.p_home), "home_ml": int(last.home_ml), "away_ml": int(last.away_ml),
            "home_best": int(last.home_best), "away_best": int(last.away_best),
            "home_open": None if pd.isna(opener.get("home_open")) else int(opener.home_open),
            "away_open": None if pd.isna(opener.get("away_open")) else int(opener.away_open),
            "updated": last.fetched_at, "closed": bool(len(pre) < len(s)),
            "history": [[t, float(p)] for t, p in zip(pre.fetched_at, pre.p_home)],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "ESPN (DraftKings)",
                               "games": games, "spreads": lines}, separators=(",", ":")), encoding="utf-8")


def main():
    now = dt.datetime.now(dt.timezone.utc)
    rows, lines = fetch(now)
    if rows:
        SNAPSHOTS.parent.mkdir(exist_ok=True)
        old = pd.read_csv(SNAPSHOTS) if SNAPSHOTS.exists() else None
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(SNAPSHOTS, index=False)
    write_site(now, lines)
    print(f"Moneylines for {len(rows)} upcoming games, spreads for {len(lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
