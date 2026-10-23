"""One-time download of lineup stints for past seasons from ESPN play-by-play.

Writes data/nba/stints/<season>.csv.gz (one row per stint) and <season>_athletes.csv.gz
(ESPN athlete id, name and team for each game, used to map onto NBA player ids). Seasons
already on disk are skipped, so a rerun resumes.

  python tools/pbp.py [first_season] [last_season]
"""
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import espn  # noqa: E402
import hist_odds  # noqa: E402
import stints  # noqa: E402

OUT = ROOT / "data" / "nba" / "stints"


def game(g):
    eid, date, home, away = g
    try:
        rows, ath = stints.parse(espn.get("summary", event=eid), eid)
    except Exception as e:
        print(f"  {eid}: {e}", flush=True)
        return [], []
    for a in ath:
        a["date"] = date
    return rows, ath


def main(first=2010, last=2025):
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(12) as ex:
        for y in range(first, last + 1):
            path = OUT / f"{y}.csv.gz"
            if path.exists():
                continue
            days = [dt.date(y, 10, 1) + dt.timedelta(days=i) for i in range((dt.date(y + 1, 6, 30) - dt.date(y, 10, 1)).days + 1)]
            games = [g for gs in ex.map(hist_odds.day_games, days) for g in gs]
            rows, ath = [], []
            for r, a in ex.map(game, games):
                rows += r
                ath += a
            st = stints.frame(rows)
            pd.DataFrame(ath).to_csv(OUT / f"{y}_athletes.csv.gz", index=False, compression="gzip")
            st.to_csv(path, index=False, compression="gzip")
            print(f"{y}-{str(y + 1)[-2:]}: {len(games)} games, {st.espn_id.nunique()} parsed, {len(st)} stints", flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
