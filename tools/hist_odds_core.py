"""Fill seasons that ESPN's pickcenter left empty (2023-24, 2024-25) from ESPN's core odds API,
which keeps pre-game moneylines from several sportsbooks. Live (in-game) odds are excluded.
Overwrites data/nba/odds_hist/<season>.csv for the given seasons, then rebuilds odds_hist.csv.gz.

  python tools/hist_odds_core.py 2023 2024
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import hist_odds  # noqa: E402

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"


def core_lines(game):
    eid, date, home, away = game
    for attempt in range(3):
        try:
            items = requests.get(CORE.format(eid), timeout=30).json().get("items", [])
            break
        except Exception:
            items = []
    rows = []
    for i in items:
        name = i.get("provider", {}).get("name", "")
        h, a = i.get("homeTeamOdds", {}).get("moneyLine"), i.get("awayTeamOdds", {}).get("moneyLine")
        if "live" in name.lower() or h is None or a is None:
            continue
        rows.append([eid, date, home, away, name, h, a, i.get("spread")])
    return rows


def main(seasons):
    import datetime as dt
    with ThreadPoolExecutor(12) as ex:
        for y in seasons:
            days = [dt.date(y, 10, 1) + dt.timedelta(days=i) for i in range((dt.date(y + 1, 6, 30) - dt.date(y, 10, 1)).days + 1)]
            games = [g for gs in ex.map(hist_odds.day_games, days) for g in gs]
            rows = [r for rs in ex.map(core_lines, games) for r in rs]
            df = pd.DataFrame(rows, columns=hist_odds.COLS)
            df.to_csv(hist_odds.PARTS / f"{y}.csv", index=False)
            print(f"{y}-{str(y + 1)[-2:]}: {len(games)} games, {df.espn_id.nunique()} with lines, {df.provider.nunique()} books", flush=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(hist_odds.PARTS.glob("*.csv"))], ignore_index=True)
    df.to_csv(hist_odds.OUT, index=False, compression="gzip")
    print(f"wrote {len(df)} rows for {df.espn_id.nunique()} games")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [2023, 2024])
