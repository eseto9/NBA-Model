"""One-time backfill of historical NBA moneylines from ESPN's pickcenter.

For every game in past seasons, records each provider's home/away moneyline so the
backtest can compare the model with the betting market. Saves each season as it finishes
(data/nba/odds_hist/<season>.csv, so a rerun resumes), then combines them into
data/nba/odds_hist.csv.gz.

  python tools/hist_odds.py [first_season] [last_season]
"""
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import espn  # noqa: E402

PARTS = ROOT / "data" / "nba" / "odds_hist"
OUT = ROOT / "data" / "nba" / "odds_hist.csv.gz"
COLS = ["espn_id", "date", "home", "away", "provider", "home_ml", "away_ml", "spread"]


def day_games(day):
    """(espn id, date, home, away) for each final game on `day`."""
    try:
        events = espn.scoreboard(day)
    except Exception as e:
        print(f"  {day}: {e}", flush=True)
        return []
    out = []
    for e in events:
        if espn.game_id(e) and e["status"]["type"]["name"] == "STATUS_FINAL":
            sides = {c["homeAway"]: espn.nba_abbr(c["team"]["abbreviation"]) for c in e["competitions"][0]["competitors"]}
            out.append((e["id"], f"{espn.et_day(e['date']):%Y-%m-%d}", sides["home"], sides["away"]))
    return out


def game_lines(game):
    eid, date, home, away = game
    try:
        pc = espn.get("summary", event=eid).get("pickcenter") or []
    except Exception:
        return []
    rows = []
    for p in pc:
        h, a = p.get("homeTeamOdds", {}).get("moneyLine"), p.get("awayTeamOdds", {}).get("moneyLine")
        if h is not None and a is not None:
            rows.append([eid, date, home, away, p.get("provider", {}).get("name"), h, a, p.get("spread")])
    return rows


def season(y, ex):
    days = [dt.date(y, 10, 1) + dt.timedelta(days=i) for i in range((dt.date(y + 1, 6, 30) - dt.date(y, 10, 1)).days + 1)]
    games = [g for gs in ex.map(day_games, days) for g in gs]
    # sample first: early seasons have no lines on ESPN at all, so don't fetch every game
    sample = [r for rs in ex.map(game_lines, games[::max(1, len(games) // 40)]) for r in rs]
    rows = [r for rs in ex.map(game_lines, games) for r in rs] if sample else []
    df = pd.DataFrame(rows, columns=COLS)
    print(f"{y}-{str(y + 1)[-2:]}: {len(games)} games, {df.espn_id.nunique()} with lines", flush=True)
    return df


def main(first=2010, last=2025):
    PARTS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        for y in range(first, last + 1):
            path = PARTS / f"{y}.csv"
            if not path.exists():
                season(y, ex).to_csv(path, index=False)
    df = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("*.csv"))], ignore_index=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {len(df)} rows for {df.espn_id.nunique()} games")
    print(df.groupby("provider").espn_id.nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
