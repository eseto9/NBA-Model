"""Can a new metric or input beat the betting market? Walk-forward experiments.

Every variant is fit season by season on earlier seasons only (2012-13 to 2025-26) and
scored on the games that have a historical market line, next to the market itself.
Differences in log loss come with a paired standard error, because with ~15,000 games a
0.002 gap can be noise.

  python tools/experiments.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import export  # noqa: E402
import nba  # noqa: E402
import players  # noqa: E402

# ---------------------------------------------------------------- Tipoff Impact (TPI)

TPI_STATS = ["PTS", "FGM", "FGA", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
TPI_FIT_BEFORE = "2012-08-01"       # fit on 2010-11 and 2011-12 only: never on a tested season


def fit_tpi(pg, ridge=50.0):
    """Box-score regression onto on-court plus-minus per 100 possessions (a homemade BPM).
    Weighted by minutes, ridge-penalized, fit only on seasons before the backtest."""
    team = pg.groupby(["GAME_ID", "TEAM_ID"])[["FGA", "OREB", "TOV", "FTA"]].transform("sum")
    poss = (team.FGA - team.OREB + team.TOV + 0.44 * team.FTA).groupby(pg.GAME_ID).transform("mean")
    on_poss = poss * pg.MIN / 48
    rates = pd.DataFrame({s: 100 * pg[s] / on_poss for s in TPI_STATS})
    fit = (pg.GAME_DATE < TPI_FIT_BEFORE) & (pg.MIN >= 10) & rates.notna().all(axis=1)
    X, y, w = rates[fit].values, pg.net[fit].values, pg.MIN[fit].values
    mu = np.average(X, axis=0, weights=w)
    Xc, yc = X - mu, y - np.average(y, weights=w)
    beta = np.linalg.solve(Xc.T @ (Xc * w[:, None]) + ridge * np.eye(len(TPI_STATS)) * w.mean(), Xc.T @ (yc * w))
    intercept = np.average(y, weights=w) - mu @ beta
    pg["tpi"] = (intercept + rates.fillna(0).values @ beta).clip(-60, 60)
    return dict(zip(TPI_STATS, beta.round(3))), round(float(intercept), 2)


from situational import add_massey, add_schedule  # noqa: E402


# ---------------------------------------------------------------- harness

def walk_forward(g, cols, market_input=False):
    """Season-by-season predictions on games with a market line. Returns DataFrame(p, p_market, y, season)."""
    played = nba.training_rows(g)
    seasons = [s for s in sorted(played.season.unique()) if s < nba.current_season()][1:]
    out = []
    for s in seasons:
        tr, te = played[played.season < s], played[played.season == s]
        if market_input:          # the market can only be an input where it exists
            tr, te = tr[tr.p_market.notna()], te[te.p_market.notna()]
            if len(tr) < 1000 or not len(te):
                continue
        X_tr, X_te = design(tr, cols, market_input), design(te, cols, market_input)
        m = nba.make_models()["logistic"].fit(X_tr, (tr.home_pts > tr.away_pts).astype(int))
        out.append(pd.DataFrame({"season": s, "p": m.predict_proba(X_te)[:, 1], "p_market": te.p_market.values,
                                 "y": (te.home_pts > te.away_pts).astype(int).values}))
    return pd.concat(out)


SCHED = ["x_travel", "x_tz", "x_dense", "x_load7", "x_altitude"]   # round2 adds x_massey


def design(d, cols, market_input):
    X = nba.lab_features(d).assign(d_lu_rapm=d.d_lu_rapm.fillna(0).values if "d_lu_rapm" in d else 0)[cols]
    if market_input:
        X = X.assign(mkt_logit=np.log(d.p_market / (1 - d.p_market)))
    return X


def score(r):
    r = r[r.p_market.notna()]
    eps = 1e-9
    ll = lambda p: -(r.y * np.log(p + eps) + (1 - r.y) * np.log(1 - p + eps))
    diff = ll(r.p) - ll(r.p_market)
    return {"n": len(r), "acc": ((r.p >= .5) == (r.y == 1)).mean(), "mkt_acc": ((r.p_market >= .5) == (r.y == 1)).mean(),
            "ll": ll(r.p).mean(), "mkt_ll": ll(r.p_market).mean(),
            "ll_diff": diff.mean(), "se": diff.std() / math.sqrt(len(r))}


def load():
    """All games with default, Lab and situational features, plus the historical market line."""
    g = nba.build()
    mk = export.historical_market()
    return g.assign(day=g.date.dt.strftime("%Y-%m-%d")).merge(mk.rename(columns={"date": "day"}), on=["day", "home", "away"], how="left")


def report(name, r):
    s = score(r)
    print(f"{name:28s} n={s['n']:5d} acc {s['acc']:.4f} (mkt {s['mkt_acc']:.4f})  logloss {s['ll']:.4f} "
          f"(mkt {s['mkt_ll']:.4f})  diff {s['ll_diff']:+.4f} ± {s['se']:.4f}", flush=True)
    return {"variant": name, **s}


def add_rapm(g):
    """Walk-forward lineup RAPM (d_lu_rapm) for every game."""
    import rapm
    st, ath = rapm.load_stints()
    st = rapm.prepare(st, rapm.athlete_ids(ath))
    pg = nba.load_players()[0]
    lu, _ = rapm.history(pg, st)
    for side, sfx in (("home", "_h"), ("away", "_a")):
        g = g.merge(lu.rename(columns={"team_id": f"{side}_id", "lu_rapm": f"lu_rapm{sfx}"}), on=["game_id", f"{side}_id"], how="left")
    return g.assign(d_lu_rapm=(g.lu_rapm_h - g.lu_rapm_a).fillna(0))


def main():
    coefs, b0 = fit_tpi(players.load_player_games(nba.data_files("players"), nba.GAME_TYPES))
    print("TPI coefficients (per 100 poss):", coefs, "intercept", b0)
    g = add_rapm(load())
    base = list(nba.features(g.head(1)).columns)
    variants = {
        "+ RAPM (lineup data)": base + ["d_lu_rapm"],
        "+ RAPM + TPI": base + ["d_lu_rapm", "d_lu_tpi"],
        "default model": base,
        "+ TPI": base + ["d_lu_tpi"],
        "+ schedule/travel": base + SCHED,
        "+ Massey": base + ["x_massey"],
        "+ TPI + Massey + schedule": base + ["d_lu_tpi", "x_massey"] + SCHED,
        "market + default features": base,
    }
    rows = [report(name, walk_forward(g, cols, market_input=name.startswith("market"))) for name, cols in variants.items()]
    rows.append(report("stack: market + model", stacked(walk_forward(g, base), g)))
    rows.append(report("stack: market + model w/ RAPM", stacked(walk_forward(g, base + ["d_lu_rapm"]), g)))
    pd.DataFrame(rows).to_csv(ROOT / "log" / "experiments.csv", index=False)


# ---------------------------------------------------------------- round 2: Massey ratings, stacking

def stacked(model_r, g_played):
    """Two-input blend fit walk-forward: logit p = a + b*logit(market) + c*logit(model)."""
    from sklearn.linear_model import LogisticRegression
    r = model_r[model_r.p_market.notna()].copy()
    lg = lambda p: np.log(p / (1 - p))
    out = []
    for s in sorted(r.season.unique())[1:]:
        tr, te = r[r.season < s], r[r.season == s]
        m = LogisticRegression(C=1.0).fit(np.c_[lg(tr.p_market), lg(tr.p)], tr.y)
        out.append(te.assign(p=m.predict_proba(np.c_[lg(te.p_market), lg(te.p)])[:, 1]))
        last = m
    print("  blend weights (market, model):", last.coef_.round(3), "intercept", last.intercept_.round(3))
    return pd.concat(out)


if __name__ == "__main__":
    main()
