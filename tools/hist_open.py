"""Opening and closing moneylines for past games from ESPN's core odds API, which keeps the
open line next to the closing ("current") one from 2023-24 on. One sportsbook per game: the
first one that has both. Writes data/nba/odds_open.csv.gz, used for the closing-line value
backtest on the Record page. Resumable; the daily run adds the new season's finished games.

  python tools/hist_open.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nba" / "odds_open.csv.gz"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
FIRST = "2023-08-01"                     # ESPN has no open lines before 2023-24


def american(o):
    """American odds from one of the API's price objects, or None."""
    v = (o or {}).get("american")
    try:
        v = int(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None
    return v if abs(v) >= 100 else None


def lines(game):
    eid, date, home, away = game
    for _ in range(3):
        try:
            items = requests.get(CORE.format(eid), timeout=30).json().get("items", [])
            break
        except Exception:
            items = []
    for i in items:
        name = i.get("provider", {}).get("name", "")
        h, a = i.get("homeTeamOdds", {}), i.get("awayTeamOdds", {})
        if "live" in name.lower():
            continue
        ml = lambda side, k: american((side.get(k) or {}).get("moneyLine"))
        row = [ml(h, "open"), ml(a, "open"), ml(h, "close") or ml(h, "current"), ml(a, "close") or ml(a, "current")]
        if all(row):
            return [eid, date, home, away, name, *row]
    return [eid, date, home, away, None, None, None, None, None]


def season_games():
    """This season's finished games from the ESPN box score files (GAME_ID carries ESPN's event id)."""
    frames = [pd.read_csv(f, usecols=["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP"], dtype=str)
              for f in sorted((ROOT / "data" / "espn").glob("*/*.games.csv")) if f.stat().st_size > 200]
    if not frames:
        return pd.DataFrame(columns=["espn_id", "date", "home", "away"])
    g = pd.concat(frames)
    g = g.dropna(subset=["GAME_ID", "MATCHUP"])
    g = g[g.GAME_ID.astype(str).str.contains("E")]
    home = g[g.MATCHUP.str.contains("vs.")]
    return pd.DataFrame({"espn_id": home.GAME_ID.astype(str).str.split("E").str[1].astype(int), "date": home.GAME_DATE,
                         "home": home.TEAM_ABBREVIATION, "away": home.MATCHUP.str.split(" vs. ").str[1]})


def main():
    hist = pd.read_csv(ROOT / "data" / "nba" / "odds_hist.csv.gz")
    games = pd.concat([hist[hist.date >= FIRST][["espn_id", "date", "home", "away"]], season_games()]).drop_duplicates("espn_id")
    have = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["espn_id"])
    todo = [tuple(g) for g in games.itertuples(index=False) if g.espn_id not in set(have.espn_id)]
    cols = ["espn_id", "date", "home", "away", "provider", "home_open", "away_open", "home_close", "away_close"]
    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(todo), 400):
            got = pd.DataFrame(list(ex.map(lines, todo[k:k + 400])), columns=cols)
            have = pd.concat([have, got], ignore_index=True)
            have.to_csv(OUT, index=False)
            print(f"  {len(have)} of {len(games)} games, {have.home_open.notna().sum()} with open lines", flush=True)


if __name__ == "__main__":
    main()
