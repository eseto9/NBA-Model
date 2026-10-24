"""How well do expected minutes predict actual minutes? Variants of the projection, scored by
mean absolute error on every player-game from 2012-13 on (pre-game projections only).

  python tools/minutes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nba  # noqa: E402
import players  # noqa: E402


def load():
    pg = nba.add_draft_bucket(players.load_player_games(nba.data_files("players"), nba.GAME_TYPES))
    tg = nba.load_team_games()
    margin = tg.merge(tg[["GAME_ID", "TEAM_ID", "PTS"]], on="GAME_ID", suffixes=("", "_o"))
    margin = margin[margin.TEAM_ID != margin.TEAM_ID_o].set_index(["GAME_ID", "TEAM_ID"])
    pg["margin"] = pd.Series((margin.PTS - margin.PTS_o).values, index=margin.index).reindex(
        pd.MultiIndex.from_arrays([pg.GAME_ID, pg.TEAM_ID])).values
    # team games missed since the player's previous appearance (same team)
    team_games = tg[["GAME_ID", "TEAM_ID", "GAME_DATE"]].sort_values("GAME_DATE")
    team_games["n"] = team_games.groupby("TEAM_ID").cumcount()
    pg = pg.merge(team_games[["GAME_ID", "TEAM_ID", "n"]], on=["GAME_ID", "TEAM_ID"], how="left")
    pg = pg.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    prev_n = pg.groupby("PLAYER_ID").n.shift(1)
    same_team = pg.groupby("PLAYER_ID").TEAM_ID.shift(1) == pg.TEAM_ID
    pg["missed"] = np.where(same_team, pg.n - prev_n - 1, 0).clip(0)
    return pg


def projection(pg, halflife=6, blowout_weight=1.0, blowout_at=20):
    """Pre-game expected minutes: weighted EWMA of past minutes (blowouts down-weighted)."""
    alpha = 1 - 0.5 ** (1 / halflife)
    w = np.where(pg.margin.abs() >= blowout_at, blowout_weight, 1.0)
    by = pg.PLAYER_ID
    num = (pg.MIN * w).groupby(by).transform(lambda x: x.ewm(alpha=alpha).sum())
    den = pd.Series(w, index=pg.index).groupby(by).transform(lambda x: x.ewm(alpha=alpha).sum())
    return (num / den).groupby(by).shift(1)


def score(pg, proj, name):
    m = pg.GAME_DATE.ge("2012-10-01") & proj.notna()
    err = (proj[m] - pg.MIN[m]).abs()
    back = m & (pg.missed >= 3)
    print(f"{name:42s} MAE {err.mean():.3f}  (returning from 3+ missed games: {(proj[back] - pg.MIN[back]).abs().mean():.3f}, "
          f"bias {(proj[back] - pg.MIN[back]).mean():+.2f})", flush=True)


def main():
    pg = load()
    base = projection(pg)
    score(pg, base, "current (half-life 6)")
    for h in (3, 4, 8):
        score(pg, projection(pg, halflife=h), f"half-life {h}")
    for w in (0.5, 0.25):
        score(pg, projection(pg, blowout_weight=w), f"blowouts (20+) weighted {w}")
    # minutes restriction after an absence: learn the factor from history (pre-2018), apply to the rest
    back = (pg.missed >= 3) & pg.GAME_DATE.lt("2018-08-01") & base.notna()
    f1 = (pg.MIN[back] / base[back]).clip(0, 2).median()
    print(f"  learned first-game-back factor (2010-18 only): {f1:.3f}")
    adj = base.where(pg.missed < 3, base * f1)
    score(pg, adj, f"+ first game back x {f1:.2f}")
    best = projection(pg, halflife=4, blowout_weight=0.5)
    score(pg, best.where(pg.missed < 3, best * f1), f"half-life 4, blowouts 0.5, back x {f1:.2f}")


if __name__ == "__main__":
    main()
