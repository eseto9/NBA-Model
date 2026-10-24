"""Do players play better on their birthday? Compares each birthday game with the same player's
other games that season (Game Score, points, plus-minus, minutes), 2010-11 onward.

  python tools/birthdays.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nba  # noqa: E402
import players  # noqa: E402

CACHE = ROOT / "data" / "nba" / "birthdays.csv"


def fetch_dob(pid):
    from nba_api.stats.endpoints import commonplayerinfo
    for attempt in range(3):
        try:
            d = commonplayerinfo.CommonPlayerInfo(player_id=pid, timeout=30).get_data_frames()[0]
            return pid, str(d.BIRTHDATE.iloc[0])[:10]
        except Exception:
            time.sleep(5 * (attempt + 1))
    return pid, ""


def birthdays():
    """Birth dates from stats.nba.com player profiles (one request per player, keyed by NBA id),
    for everyone with a game since 2010. Resumable: progress is saved every 100 players."""
    from concurrent.futures import ThreadPoolExecutor
    pg = pd.concat([pd.read_csv(f, usecols=["PLAYER_ID"]) for f in nba.data_files("players")]).drop_duplicates()
    ids = sorted(int(x) for x in pg.PLAYER_ID if x > 0)
    have = pd.read_csv(CACHE) if CACHE.exists() and "PLAYER_ID" in pd.read_csv(CACHE, nrows=1) else pd.DataFrame(columns=["PLAYER_ID", "dob"])
    todo = [i for i in ids if i not in set(have.PLAYER_ID)]
    with ThreadPoolExecutor(3) as ex:
        for k in range(0, len(todo), 100):
            got = pd.DataFrame(list(ex.map(fetch_dob, todo[k:k + 100])), columns=["PLAYER_ID", "dob"])
            have = pd.concat([have, got], ignore_index=True)
            have.to_csv(CACHE, index=False)
            print(f"  birthdays: {len(have)} of {len(ids)}", flush=True)
    return have[have.dob.astype(str).str.len() == 10]


def main():
    b = birthdays()
    b["PLAYER_ID"] = b.PLAYER_ID.astype(str)
    print(f"birthdays for {len(b)} players")
    pg = players.load_player_games(nba.data_files("players"), nba.GAME_TYPES)
    pg = pg[pg.MIN >= 10].copy()
    pg["PLAYER_ID"] = pg.PLAYER_ID.astype(str)
    pg = pg.merge(b[["PLAYER_ID", "dob"]].drop_duplicates("PLAYER_ID"), on="PLAYER_ID")
    pg["gs"] = pg.gmsc * pg.MIN / 36                                   # Game Score (raw)
    pg["bday"] = pg.GAME_DATE.dt.strftime("%m-%d") == pg.dob.str[5:]
    pg["season"] = pg.GAME_DATE.dt.year - (pg.GAME_DATE.dt.month < 8)
    # each game relative to the same player's average in that season (excluding the game itself)
    for c in ("gs", "PTS", "PLUS_MINUS", "MIN"):
        grp = pg.groupby(["PLAYER_ID", "season"])[c]
        tot, cnt = grp.transform("sum"), grp.transform("count")
        pg[f"d_{c}"] = pg[c] - (tot - pg[c]) / (cnt - 1).replace(0, np.nan)
    bd = pg[pg.bday].dropna(subset=["d_gs"])
    print(f"{len(bd)} birthday games ({bd.PLAYER_ID.nunique()} players), vs {len(pg) - len(bd):,} other games\n")
    print(f"{'stat':12s} {'birthday vs own season avg':>28s} {'95% CI':>18s}")
    for c, label in (("d_gs", "Game Score"), ("d_PTS", "Points"), ("d_PLUS_MINUS", "Plus-minus"), ("d_MIN", "Minutes")):
        x = bd[c].dropna()
        se = x.std() / np.sqrt(len(x))
        print(f"{label:12s} {x.mean():>+28.2f} {f'[{x.mean() - 1.96 * se:+.2f}, {x.mean() + 1.96 * se:+.2f}]':>18s}")
    # placebo: the same comparison on the day before and after the birthday
    for shift, label in ((-1, "day before"), (1, "day after")):
        d = (pd.to_datetime("2000-" + pg.dob.str[5:], errors="coerce") + pd.Timedelta(days=shift)).dt.strftime("%m-%d")
        x = pg[pg.GAME_DATE.dt.strftime("%m-%d") == d].d_gs.dropna()
        print(f"  placebo, {label}: Game Score {x.mean():+.2f} (n={len(x)})")
    # win rate for the birthday player's team
    tg = nba.load_team_games()
    wins = tg.set_index(["GAME_ID", "TEAM_ID"]).WL.eq("W")
    bd_w = wins.reindex(pd.MultiIndex.from_arrays([bd.GAME_ID, bd.TEAM_ID])).mean()
    print(f"  team won {bd_w:.1%} of birthday games (all games: 50.0%)\n")

    lb = pg[pg.PLAYER_NAME.str.contains("LeBron")].copy()
    lbd = lb[lb.bday].sort_values("GAME_DATE")
    print(f"LeBron James (Dec 30): {len(lbd)} birthday games since 2010")
    for r in lbd.itertuples():
        avg = lb[(lb.season == r.season) & (lb.GAME_ID != r.GAME_ID)]
        print(f"  {r.GAME_DATE:%Y-%m-%d}  {r.PTS:>3.0f} pts  Game Score {r.gs:5.1f} (season avg {avg.gs.mean():4.1f})  +/- {r.PLUS_MINUS:+.0f}")
    print(f"  average: Game Score {lbd.d_gs.mean():+.2f} vs his season average, points {lbd.d_PTS.mean():+.1f}")


if __name__ == "__main__":
    main()
