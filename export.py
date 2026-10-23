"""Builds the website's data (site/data/*.json) from the model.

Run once a day after the night's last game (the daily GitHub Actions job does this):
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
import odds
import players

ET = ZoneInfo("America/New_York")
SITE = nba.ROOT / "site" / "data"
LOG = nba.ROOT / "log" / "predictions.csv"
RANK_LOG = nba.ROOT / "log" / "rankings.csv"
OVERRIDES = nba.ROOT / "data" / "overrides.csv"

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
    return [{"name": p.PLAYER_NAME, "status": p.status, "manual": bool(p.manual), "min": r(p.exp_min, 1),
             "gmsc": r(p.r_gmsc, 1), "pie": r(p.r_pie), "net": r(p.r_net, 1)} for p in t.itertuples()]


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
            "rotation": rotation(roster, row[f"{s}_id"], 13),
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
    """Post-game Elo for every team, this season and last: [date, elo, 1 if postseason]."""
    played = g[g.home_pts.notna() & (g.season >= season - 1)]
    played = played.assign(post=~played.game_id.str.startswith("002"))     # playoffs, play-in
    rows = pd.concat([
        played[["date", "season", "home", "elo_h_post", "post"]].set_axis(["date", "season", "abbr", "elo", "post"], axis=1),
        played[["date", "season", "away", "elo_a_post", "post"]].set_axis(["date", "season", "abbr", "elo", "post"], axis=1),
    ]).sort_values("date")
    return {a: [[f"{d:%Y-%m-%d}", int(round(e)), int(q)] for d, e, q in zip(t.date, t.elo, t.post)] for a, t in rows.groupby("abbr")}


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

def scratches(start):
    """Manual availability overrides (data/overrides.csv) for today and tomorrow: (out, in) name lists."""
    if not OVERRIDES.exists():
        return [], []
    o = pd.read_csv(OVERRIDES)
    o = o[(o.date >= f"{start:%Y-%m-%d}") & (o.date <= f"{start + pd.Timedelta(days=1):%Y-%m-%d}")]
    o = o.drop_duplicates("player", keep="last")         # the latest call on a player wins
    return o[o.status == "out"].player.tolist(), o[o.status == "in"].player.tolist()


def update_log(slate, now):
    """Record each slate pick. A pick can be revised (new injury news, a late scratch) until its
    game tips off; after that it is frozen and graded."""
    LOG.parent.mkdir(exist_ok=True)
    log = pd.read_csv(LOG, dtype={"game_id": str}) if LOG.exists() else pd.DataFrame(
        columns=["game_id", "date", "away", "home", "p_home", "logged_at"])
    key = lambda d: d.date.astype(str).str[:10] + d.home + d.away
    not_started = slate.tip.isna() | (pd.to_datetime(slate.tip) > pd.Timestamp(now.replace(tzinfo=None)))
    revisable = slate[not_started | ~key(slate).isin(key(log))]
    if len(revisable):
        add = pd.DataFrame({"game_id": revisable.game_id, "date": revisable.date.dt.strftime("%Y-%m-%d"), "away": revisable.away,
                            "home": revisable.home, "p_home": revisable.home_win_prob.round(4), "logged_at": now.isoformat(timespec="minutes")})
        log = pd.concat([log[~key(log).isin(key(add))], add], ignore_index=True)
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


BACKTEST_VERSION = 5        # bump when the model changes, to recompute the backtest


def historical_market():
    """Closing-ish moneylines for past games (ESPN pickcenter), averaged across providers, margin removed."""
    files = [nba.HIST / "odds_hist.csv.gz", *sorted(nba.LIVE.glob("*/*.odds.csv"))]
    frames = [pd.read_csv(f) for f in files if f.exists() and f.stat().st_size > 60]
    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "p_market"])
    o = pd.concat(frames, ignore_index=True)
    o = o[(o.home_ml.abs() >= 100) & (o.away_ml.abs() >= 100)]
    o["p_market"] = [odds.no_vig(h, a) for h, a in zip(o.home_ml, o.away_ml)]
    return o.groupby(["date", "home", "away"], as_index=False).p_market.mean()


def versus_market(df):
    """Model vs market on games that have a line: df has p (model), p_market, y."""
    m = df.dropna(subset=["p_market"])
    if not len(m):
        return {"n": 0}
    s_model, s_market = summary(m.p, m.y), summary(m.p_market, m.y)
    agree = (m.p >= 0.5) == (m.p_market >= 0.5)
    dis = m[~agree]
    return {"n": int(len(m)), "model": {k: s_model[k] for k in ("accuracy", "log_loss", "correct")},
            "market": {k: s_market[k] for k in ("accuracy", "log_loss", "correct", "calibration")},
            "agree": r(agree.mean()), "disagree_n": int(len(dis)),
            "disagree_model_right": int(((dis.p >= 0.5) == (dis.y == 1)).sum())}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_walk_forward(df):
    """Tipoff x Market: logit p = a + b*logit(market) + c*logit(model), fit season by season on
    earlier seasons' games that had a line. Returns (blended predictions, coefficients fit on all)."""
    from sklearn.linear_model import LogisticRegression
    m = df[df.p_market.notna()]
    out = []
    for s in sorted(m.season.unique())[1:]:
        tr, te = m[m.season < s], m[m.season == s]
        fit = LogisticRegression(C=1.0).fit(np.c_[logit(tr.p_market), logit(tr.p)], tr.y)
        out.append(te.assign(p=fit.predict_proba(np.c_[logit(te.p_market), logit(te.p)])[:, 1]))
    fit = LogisticRegression(C=1.0).fit(np.c_[logit(m.p_market), logit(m.p)], m.y)
    coef = {"a": r(fit.intercept_[0], 4), "market": r(fit.coef_[0][0], 4), "model": r(fit.coef_[0][1], 4)}
    return pd.concat(out), coef


def backtest_all(g):
    """Walk-forward backtest of every season: each season is predicted by a model trained
    only on the seasons before it. Returns per-season and overall results, with the market."""
    played = nba.training_rows(g)
    seasons = [s for s in sorted(played.season.unique()) if s < nba.current_season()][1:]   # need a season to train on
    games = []
    for s in seasons:
        train, test = played[played.season < s], played[played.season == s]
        p = nba.predict_proba(nba.fit_final(train), test)
        games.append(pd.DataFrame({"season": s, "date": test.date.dt.strftime("%Y-%m-%d"), "home": test.home.values,
                                   "away": test.away.values, "p": p, "y": (test.home_pts > test.away_pts).astype(int).values}))
        print(f"  backtest {nba.season_str(s)}")
    df = pd.concat(games).merge(historical_market(), on=["date", "home", "away"], how="left")
    blend, blend_coef = blend_walk_forward(df)
    per_season = []
    for s, d in df.groupby("season"):
        season_sum = summary(d.p, d.y)
        vs = versus_market(d)
        vs.get("market", {}).pop("calibration", None)
        season_sum.pop("calibration")
        per_season.append({"season": nba.season_str(s), **season_sum, "vs_market": vs})
    days = df.assign(hit=(df.p >= 0.5) == (df.y == 1)).groupby("date").hit.agg(["sum", "count"]).reset_index()
    return {"version": BACKTEST_VERSION, "through": nba.season_str(seasons[-1]), "first": nba.season_str(seasons[0]),
            "overall": summary(df.p, df.y), "vs_market": versus_market(df), "seasons": per_season,
            "blend": {"coef": blend_coef, "vs_market": versus_market(blend)},
            "daily": [[d, int(s), int(c)] for d, s, c in days.values.tolist()]}


# ---------------------------------------------------------------- model lab

LAB_GROUPS = [   # key, label, features, on by default, description
    ("elo", "Elo rating", ["elo_diff"], True, "Margin-adjusted team Elo difference, including home court."),
    ("efficiency", "Efficiency form", ["d_net", "d_ortg", "d_drtg", "d_win"], True, "Recent offensive, defensive and net rating per 100 possessions, and win rate (10-game half-life)."),
    ("four_factors", "Four factors", ["d_efg", "d_opp_efg", "d_tov", "d_orb", "d_ftr"], True, "Shooting (and shooting allowed), turnovers, offensive rebounding, free throws."),
    ("rest", "Rest & schedule", ["rest_h", "rest_a", "b2b_h", "b2b_a", "gp_min"], True, "Days of rest, back-to-backs, games played this season."),
    ("context", "Game context", ["playoffs", "neutral"], True, "Playoff and neutral-site flags."),
    ("pie", "Players: PIE", ["d_lu_pie"], True, "Available rotation's minutes-weighted share of box-score events."),
    ("onnet", "Players: on-court net rating", ["d_lu_net"], True, "Available rotation's plus-minus per 100 possessions on the floor."),
    ("gmsc", "Players: Game Score/36", ["d_lu_gmsc"], True, "Available rotation's box-score production per 36 minutes."),
    ("star", "Players: best available", ["d_lu_star"], True, "Highest-rated available player expected to play 20+ minutes."),
    ("depth", "Players: rotation minutes", ["d_lu_min"], True, "Expected minutes of the top 10 available players (missing-player load)."),
    ("pace", "Pace (tempo)", ["d_pace"], False, "Difference in recent possessions per game. Off by default: tempo says little about who wins."),
    ("tpi", "Players: Tipoff Impact (custom)", ["d_lu_tpi"], False, "Our own BPM-style metric: box-score rates per 100 possessions regressed onto on-court plus-minus, fit on 2010-12 only."),
    ("massey", "Massey ratings", ["x_massey"], False, "Opponent-adjusted team ratings: recency-weighted ridge regression of every margin on team indicators."),
    ("travel", "Travel & schedule density", ["x_travel", "x_tz", "x_dense", "x_load7", "x_altitude"], False, "Distance since the last game, time-zone shift, 3-in-4 nights, games in 7 days, Denver/Utah altitude."),
]
LAB_UNAVAILABLE = [
    ("LEBRON", "BBall Index's tracking-based plus-minus. Proprietary and paywalled, no free daily feed."),
    ("VORP", "Volume stat (BPM x minutes share). Minutes are already modeled through expected minutes and availability."),
    ("Win Shares", "Cumulative, and its defensive half is mostly team defense, which Elo and form already carry."),
]
LAB_VERSION = 2


def lab_columns(rows, X, market):
    """Columnar arrays for the Lab: features rounded to 4 significant figures."""
    m = rows[["date", "home", "away"]].assign(date=rows.date.dt.strftime("%Y-%m-%d")).merge(market, on=["date", "home", "away"], how="left")
    sig = lambda v: float(f"{v:.4g}")
    return {"season": rows.season.astype(int).tolist(), "date": rows.date.dt.strftime("%Y-%m-%d").tolist(),
            "home": rows.home.tolist(), "away": rows.away.tolist(),
            "y": (rows.home_pts > rows.away_pts).astype(int).tolist() if rows.home_pts.notna().all() else None,
            "elo_prob": [sig(v) for v in rows.elo_prob],
            "p_market": [None if pd.isna(v) else round(float(v), 4) for v in m.p_market],
            "X": {c: [sig(v) for v in X[c]] for c in X.columns}}


def write_lab(hist, slate, season):
    """lab_hist.json: completed seasons (rewritten only when a season ends or the Lab changes);
    lab_live.json: this season so far plus today's slate (small, daily)."""
    rows = nba.training_rows(hist)
    market = historical_market()
    meta = {"version": LAB_VERSION, "season": season,
            "groups": [{"key": k, "label": l, "features": f, "default": d, "desc": t} for k, l, f, d, t in LAB_GROUPS],
            "unavailable": [{"label": l, "why": w} for l, w in LAB_UNAVAILABLE],
            "default": {"model": "logistic", "C": 0.1, "groups": [k for k, _, _, d, _ in LAB_GROUPS if d]}}
    path = SITE / "lab_hist.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if old.get("version") != LAB_VERSION or old.get("season") != season:
        past = rows[rows.season < season]
        write("lab_hist.json", {**meta, **lab_columns(past, nba.lab_features(past), market)})
    cur = rows[rows.season == season]
    closing = odds.closing_lines()[["date", "home", "away", "p_home"]].rename(columns={"p_home": "p_close"})
    live_market = market.merge(closing, on=["date", "home", "away"], how="outer")
    live_market["p_market"] = live_market.p_close.fillna(live_market.p_market)
    payload = {**meta, **lab_columns(cur, nba.lab_features(cur), live_market[["date", "home", "away", "p_market"]])}
    if len(slate):
        sl = lab_columns(slate, nba.lab_features(slate), market.iloc[:0])
        payload["slate"] = {"id": slate.game_id.tolist(), "home": sl["home"], "away": sl["away"], "elo_prob": sl["elo_prob"], "X": sl["X"]}
    write("lab_live.json", payload)


# ---------------------------------------------------------------- main

def main(date=None, refresh=True):
    now = dt.datetime.now(ET)
    start = pd.Timestamp(date or now.date())
    pending = nba.update() if refresh else []
    rated = nba.load_players()
    hist = nba.build(rated=rated)
    models = nba.fit_final(nba.training_rows(hist))
    nba.MODELS.mkdir(exist_ok=True)
    joblib.dump(models, nba.MODELS / "model.joblib")

    # a week ahead so every team has a next game for the power rankings
    force_out, force_in = scratches(start)
    g, window, roster, unmatched = nba.prepare(start, start + pd.Timedelta(days=8), force_out, force_in, rated=rated, models=models)
    season = nba.current_season(start.date())
    unplayed = window[window.home_pts.isna()]
    slate_day = unplayed.date.min() if len(unplayed) else None
    slate = unplayed[unplayed.date == slate_day].sort_values("tip") if slate_day is not None else unplayed

    steps = factor_steps(models, nba.features(slate)) if len(slate) else []
    games = [game_json(row, roster, s) for (_, row), s in zip(slate.iterrows(), steps)]
    write("slate.json", {"date": f"{slate_day:%Y-%m-%d}" if slate_day is not None else None, "games": games,
                         "unmatched_injuries": sorted(unmatched)})

    # record: every game this season, graded, next to the closing betting line. Live picks were
    # published before tipoff; games from before the site launched are backfilled by a model trained
    # only on earlier seasons, using the same pre-game features (marked "backfill").
    log = update_log(slate, now)
    played = hist[(hist.season == season) & hist.home_pts.notna()]
    back = pd.DataFrame(columns=["date", "home", "away", "p_back", "home_pts", "away_pts"])
    if len(played):
        pre_season_model = nba.fit_final(nba.training_rows(hist[hist.season < season]))
        back = pd.DataFrame({"date": played.date.dt.strftime("%Y-%m-%d"), "home": played.home, "away": played.away,
                             "p_back": nba.predict_proba(pre_season_model, played),
                             "home_pts": played.home_pts, "away_pts": played.away_pts})
    keys = ["date", "home", "away"]
    graded = back.merge(log[keys + ["game_id", "p_home"]], on=keys, how="outer")
    graded["source"] = np.where(graded.p_home.notna(), "live", "backfill")
    graded["p_home"] = graded.p_home.fillna(graded.p_back)
    # closing line: last pre-tip snapshot from the live feed, else the line ESPN stored with the box score
    closing = odds.closing_lines()[keys + ["p_home", "home_ml", "away_ml"]].rename(columns={"p_home": "p_close"})
    espn_lines = historical_market()
    graded = graded.merge(closing, on=keys, how="left").merge(espn_lines, on=keys, how="left")
    graded["p_market"] = graded.p_close.fillna(graded.p_market)
    graded = graded.sort_values(["date", "home"], ascending=False)
    done = graded[graded.home_pts.notna()]
    y = (done.home_pts > done.away_pts).astype(int)
    live = done.source == "live"
    picks = [{"date": x.date, "away": x.away, "home": x.home, "p_home": r(x.p_home), "source": x.source,
              "p_market": r(x.p_market), "ml": None if pd.isna(x.home_ml) else [int(x.away_ml), int(x.home_ml)],
              "score": None if pd.isna(x.home_pts) else [int(x.away_pts), int(x.home_pts)]}
             for x in graded.itertuples()]
    vs = lambda d, yy: versus_market(pd.DataFrame({"p": d.p_home.values, "p_market": d.p_market.values, "y": yy.values}))
    write("record.json", {"season": summary(done.p_home, y), "live": summary(done.p_home[live], y[live]),
                          "vs_market": vs(done, y), "vs_market_live": vs(done[live], y[live]),
                          "first_live": log.date.min() if len(log) else None, "picks": picks})

    write_lab(hist, slate, season)

    # full walk-forward backtest; recomputed when a season finishes or the model changes
    bt_path = SITE / "backtest.json"
    bt = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else {}
    if bt.get("version") != BACKTEST_VERSION or bt.get("through") != nba.season_str(season - 1):
        write("backtest.json", backtest_all(hist))

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
    trained = nba.training_rows(hist)
    pipe = models["logistic"]
    scaler, lr = pipe[0], pipe[-1]
    write("meta.json", {
        "generated_at": now.isoformat(timespec="minutes"), "season": nba.season_str(season),
        "slate_date": f"{slate_day:%Y-%m-%d}" if slate_day is not None else None,
        "trained_games": int(len(trained)), "trained_through": f"{trained.date.max():%Y-%m-%d}",
        "last_results_date": f"{trained.date.max():%Y-%m-%d}",
        # standardized logistic coefficients: log-odds change per one-SD change in each feature
        "model": {"intercept": r(lr.intercept_[0], 4), "C": lr.C, "features": [
            {"name": f, "coef": r(c, 4), "mean": r(mu, 4), "sd": r(sd, 4)}
            for f, c, mu, sd in zip(nba.features(trained.head(1)).columns, lr.coef_[0], scaler.mean_, scaler.scale_)]},
        "priors": {b: {k: r(v, 3) for k, v in p.items()} for b, p in rated[2].items()},
    })
    print(f"Wrote site data: {len(games)} games on {slate_day:%Y-%m-%d}" if games else "Wrote site data: no upcoming games")
    return [str(d) for d in pending]      # days whose games weren't final yet


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="first date to consider for the slate (default: today, US Eastern)")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    missing = main(a.date, refresh=not a.no_refresh)
    if missing:
        print(f"Box scores not final yet for: {', '.join(missing)}")
        sys.exit(3)
