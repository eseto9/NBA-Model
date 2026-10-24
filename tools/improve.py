"""Candidate improvements to the default model, each scored walk-forward against the market.

  python tools/improve.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import experiments  # noqa: E402
import nba  # noqa: E402
import rapm  # noqa: E402


def with_rapm(g, lu):
    g = g.drop(columns=[c for c in ("lu_rapm_h", "lu_rapm_a") if c in g])
    for side, sfx in (("home", "_h"), ("away", "_a")):
        g = g.merge(lu.rename(columns={"team_id": f"{side}_id", "lu_rapm": f"lu_rapm{sfx}"}), on=["game_id", f"{side}_id"], how="left")
    return g


def box_priors(pg, months, k):
    """Each player's credibility-weighted Tipoff Impact just before each month, centred and scaled by k."""
    pg = pg.sort_values("GAME_DATE")
    out = {}
    for m in months:
        last = pg[pg.GAME_DATE < m].groupby("PLAYER_ID").post_tpi.last()
        if len(last):
            out[m] = (k * (last - last.mean())).to_dict()
    return out


def rapm_variant(st, pg, freq="MS", k=None):
    dates = pg.GAME_DATE
    months = pd.date_range(dates.min().to_period("M").to_timestamp(), dates.max() + pd.offsets.MonthBegin(1), freq=freq)
    priors = box_priors(pg, months, k) if k else {}
    fits, prev = {}, None
    for m in months:
        r = rapm.fit(st, m, prior=priors.get(m))
        if r:
            fits[m] = prev = r
        elif prev:
            fits[m] = prev
    rows = pg.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id", "GAME_DATE": "date"}).assign(avail=1.0)
    return rapm.lineup_rapm(rows, fits)


def main():
    g0 = experiments.load()                      # default features incl. RAPM (monthly, no prior)
    base = list(nba.features(g0.head(1)).columns)
    rows = [experiments.report("current default (with RAPM)", experiments.walk_forward(g0, base))]

    st = rapm.stints_ready()
    pg = nba.load_players()[0]
    for name, kw in {"RAPM, box prior k=0.5": {"k": 0.5}, "RAPM, box prior k=1.0": {"k": 1.0},
                     "RAPM, weekly refits": {"freq": "W-MON"}}.items():
        g = with_rapm(g0, rapm_variant(st, pg, **kw))
        rows.append(experiments.report(name, experiments.walk_forward(g, base)))

    for C in (0.03, 0.3, 1.0):
        orig = nba.make_models
        nba.make_models = lambda C=C: {"logistic": __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(
            __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
            __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(C=C, max_iter=2000))}
        rows.append(experiments.report(f"logistic C={C}", experiments.walk_forward(g0, base)))
        nba.make_models = orig

    for h in (7, 15):
        nba.HALFLIFE = h
        g = experiments.load()
        rows.append(experiments.report(f"team form half-life {h} games", experiments.walk_forward(g, base)))
    nba.HALFLIFE = 10
    pd.DataFrame(rows).to_csv(ROOT / "log" / "improve.csv", index=False)


if __name__ == "__main__":
    main()
