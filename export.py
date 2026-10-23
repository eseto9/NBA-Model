"""Builds the website's data (site/data/*.json) from the model.

Run once a day after the night's last game:
  python export.py                 # refresh data, retrain, predict the next slate
  python export.py --date 2026-10-25 --no-refresh

Predictions are frozen the first time they are made (log/predictions.csv), so the
public record only counts picks published before tipoff.
"""
import argparse
import datetime as dt
import json
import math
import sys
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import nba
import players

ET = ZoneInfo("America/New_York")
SITE = nba.ROOT / "site" / "data"
LOG = nba.ROOT / "log" / "predictions.csv"
RANK_LOG = nba.ROOT / "log" / "rankings.csv"

TEAM_COLORS = {
    "ATL": "#C8102E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160", "CHI": "#CE1141",
    "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240", "DET": "#C8102E", "GSW": "#1D428A",
    "HOU": "#CE1141", "IND": "#002D62", "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9",
    "MIA": "#98002E", "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160", "POR": "#E03A3E",
    "SAC": "#5A2D81", "SAS": "#8A8D8F", "TOR": "#CE1141", "UTA": "#002B5C", "WAS": "#002B5C",
}
GROUPS = {
    "Elo rating": ["elo_diff"],
    "Recent form": [f"d_{s}" for s in nba.STATS],
    "Who's playing": nba.LINEUP_FEATURES,
    "Rest": ["rest_h", "rest_a", "b2b_h", "b2b_a", "gp_min"],
}


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def logistic(z):
    return 1 / (1 + np.exp(-z))


def write(name, obj):
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / name).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- model explanations

def factor_steps(models, X):
    """Per game: how each feature group moves the home win probability, starting from 50%.
    Exact for the logistic model: logit = intercept + sum of standardized feature contributions."""
    pipe = models["logistic"]
    scaler, lr = pipe[0], pipe[-1]
    contrib = pd.DataFrame((X.values - scaler.mean_) / scaler.scale_ * lr.coef_[0], columns=X.columns, index=X.index)
    grouped = {k: contrib[v].sum(axis=1) for k, v in GROUPS.items()}
    used = sum(GROUPS.values(), [])
    grouped = {"Home court": lr.intercept_[0] + contrib.drop(columns=used).sum(axis=1), **grouped}
    out = []
    for i in X.index:
        z, steps = 0.0, []
        for label, s in grouped.items():
            before = logistic(z)
            z += s[i]
            steps.append({"label": label, "from": r(before), "to": r(logistic(z))})
        out.append(steps)
    return out


# ---------------------------------------------------------------- pieces

def rotation(roster, team_id, n=11):
    t = roster[roster.team_id == team_id].sort_values("exp_min", ascending=False).head(n)
    return [{"name": p.PLAYER_NAME, "status": p.status, "min": r(p.exp_min, 1), "gmsc": r(p.r_gmsc, 1),
             "pie": r(p.r_pie), "net": r(p.r_net, 1)} for p in t.itertuples()]


def game_json(row, roster, steps):
    def side(s):
        sfx = "_h" if s == "home" else "_a"
        abbr = row[s]
        return {
            "id": int(row[f"{s}_id"]), "abbr": abbr, "name": row.get(f"{s}_name") or abbr,
            "color": TEAM_COLORS.get(abbr, "#555"), "elo": r(row[f"elo{sfx}"], 0),
            "rest": int(row[f"rest{sfx}"]), "form_net": r(row[f"f_net{sfx}"], 1),
            "form_ortg": r(row[f"f_ortg{sfx}"], 1), "form_drtg": r(row[f"f_drtg{sfx}"], 1),
            "lineup": {k: r(row[f"lu_{k}{sfx}"]) for k in ("pie", "net", "gmsc", "star", "min")},
            "rotation": rotation(roster, row[f"{s}_id"]),
        }
    p = float(row.home_win_prob)
    return {
        "id": row.game_id, "date": f"{row.date:%Y-%m-%d}", "tip": row.get("tip"),
        "neutral": bool(row.neutral), "p_home": r(p), "p_elo": r(row.elo_prob),
        "pick": row.home if p >= 0.5 else row.away, "steps": steps,
        "home": side("home"), "away": side("away"),
    }


def power_rankings(g, start, models, records):
    """Each team's chance to beat a league-average team on a neutral floor, using its
    current Elo, form and healthy roster (taken from its next scheduled game)."""
    upcoming = g[(g.date >= start) & g.home_pts.isna()].sort_values("date")
    cols = ["elo", "gp"] + [f"f_{s}" for s in nba.STATS] + [f"lu_{k}" for k in ("pie", "net", "gmsc", "star", "min")]
    sides = []
    for s, sfx in (("home", "_h"), ("away", "_a")):
        t = upcoming[[f"{s}_id", s, "date"] + [c + sfx for c in cols]].copy()
        t.columns = ["team_id", "abbr", "next"] + cols
        sides.append(t)
    teams = pd.concat(sides).sort_values("next").drop_duplicates("team_id").set_index("team_id")
    avg = teams[cols].mean()
    syn = pd.DataFrame(index=teams.index)
    syn["elo_diff"] = teams.elo - avg.elo
    for s in nba.STATS:
        syn[f"d_{s}"] = teams[f"f_{s}"] - avg[f"f_{s}"]
    syn["rest_h"], syn["rest_a"], syn["b2b_h"], syn["b2b_a"] = 2, 2, 0, 0
    syn["gp_min"], syn["playoffs"], syn["neutral"] = teams.gp, 0, 1
    for f in nba.LINEUP_FEATURES:
        syn[f] = teams[f[2:]] - avg[f[2:]]
    teams["power"] = models["logistic"].predict_proba(syn[nba.features(upcoming.head(1)).columns])[:, 1]
    teams = teams.sort_values("power", ascending=False)
    teams["rank"] = np.arange(1, len(teams) + 1)
    teams = teams.join(records, how="left")
    return teams


def elo_history(g, season):
    """Post-game Elo for every team, this season and last."""
    played = g[g.home_pts.notna() & (g.season >= season - 1)]
    rows = pd.concat([
        played[["date", "season", "home", "elo_h_post"]].set_axis(["date", "season", "abbr", "elo"], axis=1),
        played[["date", "season", "away", "elo_a_post"]].set_axis(["date", "season", "abbr", "elo"], axis=1),
    ]).sort_values("date")
    return {a: [[f"{d:%Y-%m-%d}", int(round(e))] for d, e in zip(t.date, t.elo)] for a, t in rows.groupby("abbr")}


def season_records(g, season):
    reg = g[(g.season == season) & g.home_pts.notna() & g.game_id.str.startswith("002")]
    hw = reg.home_pts > reg.away_pts
    w = pd.concat([reg.home_id[hw], reg.away_id[~hw]]).value_counts()
    l = pd.concat([reg.home_id[~hw], reg.away_id[hw]]).value_counts()
    return pd.DataFrame({"w": w, "l": l}).fillna(0).astype(int)


def player_table(roster, pg, min_minutes=10):
    recent = pg.assign(gs=pg.gmsc * pg.MIN / 36).groupby("PLAYER_ID").tail(15)
    spark = recent.groupby("PLAYER_ID").gs.apply(lambda s: [r(x, 1) for x in s])
    p = roster[roster.exp_min >= min_minutes].copy()
    abbr = {int(k): v for k, v in nba.load_team_games().drop_duplicates("TEAM_ID").set_index("TEAM_ID").TEAM_ABBREVIATION.items()}
    return [{"name": x.PLAYER_NAME, "team": abbr.get(int(x.team_id), ""), "status": x.status,
             "min": r(x.exp_min, 1), "gmsc": r(x.r_gmsc, 1), "pie": r(x.r_pie), "net": r(x.r_net, 1),
             "spark": spark.get(x.PLAYER_ID, [])}
            for x in p.sort_values("r_gmsc", ascending=False).itertuples()]


# ---------------------------------------------------------------- record

def update_log(slate, now):
    LOG.parent.mkdir(exist_ok=True)
    log = pd.read_csv(LOG, dtype={"game_id": str}) if LOG.exists() else pd.DataFrame(
        columns=["game_id", "date", "away", "home", "p_home", "logged_at"])
    new = slate[~slate.game_id.isin(log.game_id)]
    if len(new):
        add = pd.DataFrame({"game_id": new.game_id, "date": new.date.dt.strftime("%Y-%m-%d"), "away": new.away,
                            "home": new.home, "p_home": new.home_win_prob.round(4), "logged_at": now})
        log = pd.concat([log, add], ignore_index=True)
        log.to_csv(LOG, index=False)
    return log


def calibration(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    conf = np.maximum(p, 1 - p)
    hit = np.where(p >= 0.5, y, 1 - y)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0001]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            out.append({"lo": lo, "hi": min(hi, 1), "n": int(m.sum()), "pred": r(conf[m].mean()), "actual": r(hit[m].mean())})
    return out


def summary(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return {"n": 0}
    hit = (p >= 0.5) == (y == 1)
    eps = 1e-9
    return {"n": int(len(p)), "correct": int(hit.sum()), "accuracy": r(hit.mean()),
            "log_loss": r(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
            "calibration": calibration(p, y)}


def backtest_season(g, season):
    """Walk-forward predictions for a finished season (model trained only on earlier seasons)."""
    played = nba.training_rows(g)
    train, test = played[played.season < season], played[played.season == season]
    models = nba.fit_final(train)
    p = nba.predict_proba(models, test)
    y = (test.home_pts > test.away_pts).astype(int).values
    days = pd.DataFrame({"date": test.date.dt.strftime("%Y-%m-%d"), "hit": (p >= 0.5) == (y == 1)})
    daily = days.groupby("date").hit.agg(["sum", "count"]).reset_index().values.tolist()
    return {"season": nba.season_str(season), **summary(p, y), "daily": [[d, int(s), int(c)] for d, s, c in daily]}


# ---------------------------------------------------------------- main

def main(date=None, refresh=True):
    now = dt.datetime.now(ET)
    start = pd.Timestamp(date or now.date())
    if refresh:
        nba.fetch()
    rated = nba.load_players()
    hist = nba.build(rated=rated)
    models = nba.fit_final(nba.training_rows(hist))
    nba.MODELS.mkdir(exist_ok=True)
    joblib.dump(models, nba.MODELS / "model.joblib")

    # a week ahead so every team has a next game for the power rankings
    g, window, roster, unmatched = nba.prepare(start, start + pd.Timedelta(days=8), rated=rated, models=models)
    season = nba.current_season(start.date())
    unplayed = window[window.home_pts.isna()]
    slate_day = unplayed.date.min() if len(unplayed) else None
    slate = unplayed[unplayed.date == slate_day].sort_values("tip") if slate_day is not None else unplayed

    steps = factor_steps(models, nba.features(slate)) if len(slate) else []
    games = [game_json(row, roster, s) for (_, row), s in zip(slate.iterrows(), steps)]
    write("slate.json", {"date": f"{slate_day:%Y-%m-%d}" if slate_day is not None else None, "games": games,
                         "unmatched_injuries": sorted(unmatched)})

    # record: frozen live picks plus last season's walk-forward backtest
    log = update_log(slate, now.isoformat(timespec="minutes"))
    res = g[g.home_pts.notna()][["game_id", "home_pts", "away_pts"]]
    graded = log.merge(res, on="game_id", how="left")
    done = graded[graded.home_pts.notna()]
    y = (done.home_pts > done.away_pts).astype(int)
    picks = [{"id": x.game_id, "date": x.date, "away": x.away, "home": x.home, "p_home": r(x.p_home),
              "score": None if math.isnan(x.home_pts) else [int(x.away_pts), int(x.home_pts)]}
             for x in graded.sort_values(["date", "game_id"], ascending=False).itertuples()]
    bt_path = SITE / "backtest.json"
    bt = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else {}
    if bt.get("season") != nba.season_str(season - 1):
        bt = backtest_season(hist, season - 1)
        write("backtest.json", bt)
    write("record.json", {"live": summary(done.p_home, y), "picks": picks})

    # teams
    records = season_records(g, season)
    teams = power_rankings(g, start, models, records)
    ranks_prev = None
    if RANK_LOG.exists():
        rl = pd.read_csv(RANK_LOG)
        prior = rl[rl.date <= f"{start - pd.Timedelta(days=7):%Y-%m-%d}"]
        prior = prior if len(prior) else rl[rl.date < f"{start:%Y-%m-%d}"]
        if len(prior):
            ranks_prev = prior[prior.date == prior.date.max()].set_index("abbr")["rank"]
    snap = pd.DataFrame({"date": f"{start:%Y-%m-%d}", "abbr": teams.abbr, "rank": teams["rank"], "power": teams.power.round(4)})
    rl = pd.read_csv(RANK_LOG) if RANK_LOG.exists() else snap.iloc[:0]
    rl = pd.concat([rl[rl.date != snap.date.iloc[0]], snap])
    rl.to_csv(RANK_LOG, index=False)
    history = {a: [[d, int(k), r(p)] for d, k, p in zip(t.date, t["rank"], t.power)] for a, t in rl.groupby("abbr")}

    elo_hist = elo_history(g, season)
    tg = nba.load_team_games().sort_values("GAME_DATE")
    names = tg.drop_duplicates("TEAM_ID", keep="last").set_index("TEAM_ID").TEAM_NAME
    out = []
    for tid, t in teams.iterrows():
        hurt = roster[(roster.team_id == tid) & (roster.avail < 1) & (roster.exp_min >= 12)]
        out.append({
            "abbr": t.abbr, "id": int(tid), "name": names.get(tid, t.abbr), "color": TEAM_COLORS.get(t.abbr, "#555"), "rank": int(t["rank"]),
            "prev_rank": None if ranks_prev is None or t.abbr not in ranks_prev else int(ranks_prev[t.abbr]),
            "power": r(t.power), "elo": r(t.elo, 0), "w": int(t.w) if not math.isnan(t.w) else 0,
            "l": int(t.l) if not math.isnan(t.l) else 0, "form_net": r(t.f_net, 1),
            "lineup_gmsc": r(t.lu_gmsc, 2), "star": r(t.lu_star, 1),
            "missing": [{"name": h.PLAYER_NAME, "status": h.status, "min": r(h.exp_min, 1)} for h in hurt.itertuples()],
            "rotation": rotation(roster, tid, 10), "elo_history": elo_hist.get(t.abbr, []),
            "rank_history": history.get(t.abbr, []),
        })
    write("teams.json", {"date": f"{start:%Y-%m-%d}", "season": nba.season_str(season), "teams": out})
    write("players.json", player_table(roster, rated[0]))
    played = nba.training_rows(hist)
    write("meta.json", {
        "generated_at": now.isoformat(timespec="minutes"), "season": nba.season_str(season),
        "slate_date": f"{slate_day:%Y-%m-%d}" if slate_day is not None else None,
        "trained_games": int(len(played)), "trained_through": f"{played.date.max():%Y-%m-%d}",
        "last_results_date": f"{played.date.max():%Y-%m-%d}",
    })
    print(f"Wrote site data: {len(games)} games on {slate_day:%Y-%m-%d}" if games else "Wrote site data: no upcoming games")
    # games that should be final by now but have no box score yet (stats feed lags, or postponed)
    missing = g[(g.season == season) & (g.date < start) & g.home_pts.isna()]
    return missing.game_id.tolist()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="first date to consider for the slate (default: today, US Eastern)")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    missing = main(a.date, refresh=not a.no_refresh)
    if missing:
        print(f"{len(missing)} past games have no box score yet: {', '.join(missing)}")
        sys.exit(3)
