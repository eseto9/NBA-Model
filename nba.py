"""NBA game winner model.

Pulls team and player game logs from stats.nba.com, builds pre-game features
(Elo rating, exponentially weighted efficiency / four-factor form, rest, and
the rated strength of the players available), and predicts the probability
that the home team wins.

Usage:
  python nba.py fetch                      # download / refresh game logs
  python nba.py backtest                   # walk-forward evaluation by season
  python nba.py train                      # fit final model on all completed games
  python nba.py predict [--date YYYY-MM-DD] [--days N] [--out "Name,Name"] [--in "Name"]
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import espn
import players

ROOT = Path(__file__).parent
DATA, MODELS, PREDS = ROOT / "data", ROOT / "models", ROOT / "predictions"
HIST, LIVE = DATA / "nba", DATA / "espn"     # past seasons (stats.nba.com) / season in progress (ESPN)
FIRST_SEASON = 2010          # 2010-11; first season is used only to warm up Elo/form
GAME_TYPES = ("002", "004", "005", "006")  # regular season, playoffs, play-in, cup final
SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")

ELO_K, ELO_HCA, ELO_MEAN, ELO_CARRY = 20, 100, 1505, 0.75
HALFLIFE = 10                # games, for exponentially weighted team form
STATS = ["net", "ortg", "drtg", "efg", "opp_efg", "tov", "orb", "ftr", "win"]
LAB_STATS = ["pace"]         # tracked for the Model Lab, not used by the default model


def season_str(y):
    return f"{y}-{str(y + 1)[-2:]}"


def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


# ---------------------------------------------------------------- data

def fetch_history():
    """Past seasons from stats.nba.com, stored once in data/nba. Needs a home connection:
    stats.nba.com blocks cloud servers. The season in progress comes from ESPN (update)."""
    from nba_api.stats.endpoints import leaguegamefinder

    HIST.mkdir(parents=True, exist_ok=True)
    for y in range(FIRST_SEASON, current_season()):
        for kind in ("games", "players"):
            path = HIST / f"{kind}_{y}.csv.gz"
            if path.exists():
                continue
            if kind == "players":
                out = players.fetch_season(season_str(y), SEASON_TYPES)
            else:
                frames = []
                for st in SEASON_TYPES:
                    frames.append(players.call(lambda: leaguegamefinder.LeagueGameFinder(
                        season_nullable=season_str(y), league_id_nullable="00",
                        season_type_nullable=st, timeout=60,
                    ).get_data_frames()[0], f"{season_str(y)} {st}"))
                    time.sleep(0.7)
                out = pd.concat(frames, ignore_index=True)
            save_history(out, kind, path)
            print(f"{season_str(y)}: {out.GAME_ID.nunique()} games ({kind})")


def save_history(df, kind, path):
    cols = espn.TEAM_COLS if kind == "games" else espn.PLAYER_COLS + ["FG3M"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["GAME_DATE"] = df.GAME_DATE.astype(str).str[:10]
    if "MIN" in df:
        df["MIN"] = df.MIN.round(2)
    df.to_csv(path, index=False, compression="gzip")


def data_files(kind):
    return sorted(HIST.glob(f"{kind}_*.csv.gz")) + sorted(LIVE.glob(f"*/*.{kind}.csv"))


def known_players():
    """Each NBA player's latest appearance, for mapping ESPN athletes onto NBA ids.
    Recent draftees count as appearing on draft day, so rookies get their NBA ids."""
    cols = ["PLAYER_ID", "PLAYER_NAME", "GAME_DATE"]
    p = pd.concat([pd.read_csv(f, usecols=cols) for f in data_files("players")], ignore_index=True)
    d = load_draft()
    d = d[(d.SEASON >= current_season() - 1) & d.PLAYER_ID.notna()]
    d = pd.DataFrame({"PLAYER_ID": d.PLAYER_ID.astype(int), "PLAYER_NAME": d.PLAYER_NAME, "GAME_DATE": d.SEASON.astype(str) + "-06-25"})
    p = pd.concat([d, p], ignore_index=True)
    p["GAME_DATE"] = pd.to_datetime(p.GAME_DATE)
    return p.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")


def load_draft():
    """Draft picks: stats.nba.com history plus newer classes from ESPN (no NBA ids)."""
    d = pd.read_csv(HIST / "draft.csv.gz", usecols=["PERSON_ID", "PLAYER_NAME", "SEASON", "OVERALL_PICK"])
    d.columns = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "PICK"]
    return pd.concat([d, *[pd.read_csv(f) for f in sorted(LIVE.glob("draft_*.csv"))]], ignore_index=True)


def add_draft_bucket(df, draft=None):
    """Draft range of each row's player, by NBA id, or by name for players known only to ESPN."""
    draft = (load_draft() if draft is None else draft).sort_values("SEASON")
    by_id = dict(zip(draft.PLAYER_ID.dropna().astype(int), draft.PICK[draft.PLAYER_ID.notna()]))
    recent = draft[draft.SEASON >= 2015]
    by_name = dict(zip(recent.PLAYER_NAME.map(players.norm_name), recent.PICK))
    picks = df.PLAYER_ID.map(by_id)
    espn_only = df.PLAYER_ID < 0
    picks[espn_only] = df.PLAYER_NAME[espn_only].map(players.norm_name).map(by_name)
    df["draft_bucket"] = picks.map(players.draft_bucket)
    return df


def team_maps():
    tg = load_team_games().sort_values("GAME_DATE").drop_duplicates("TEAM_ID", keep="last")
    return dict(zip(tg.TEAM_ABBREVIATION, tg.TEAM_ID)), dict(zip(tg.TEAM_ABBREVIATION, tg.TEAM_NAME))


def update():
    """Add the season in progress, day by day, from ESPN."""
    season = current_season()
    if load_draft().SEASON.max() < season and dt.date.today() > dt.date(season, 7, 1):
        espn.draft(season).to_csv(LIVE / f"draft_{season}.csv", index=False)   # this year's class
    ids, names = team_maps()
    return espn.update(LIVE, current_season(), ids, names, known_players())


def load_team_games():
    files = data_files("games")
    if not files:
        raise SystemExit("No data yet - run: python nba.py fetch-history, then python nba.py update")
    tg = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in files if f.stat().st_size > 200], ignore_index=True)
    tg = tg[tg.WL.notna() & tg.GAME_ID.str[:3].isin(GAME_TYPES)]
    tg = tg.drop_duplicates(["GAME_ID", "TEAM_ID"])
    tg["GAME_DATE"] = pd.to_datetime(tg.GAME_DATE)
    return tg


def to_games(tg):
    """One row per game: home team vs away team."""
    is_home = tg.MATCHUP.str.contains(" vs. ")
    g = tg[is_home].merge(tg[~is_home], on="GAME_ID", suffixes=("_h", "_a"))
    g = g.drop_duplicates("GAME_ID", keep=False)
    return pd.DataFrame({
        "game_id": g.GAME_ID, "date": g.GAME_DATE_h,
        "season": g.SEASON_ID_h.astype(str).str[-4:].astype(int),
        "home_id": g.TEAM_ID_h, "away_id": g.TEAM_ID_a,
        "home": g.TEAM_ABBREVIATION_h, "away": g.TEAM_ABBREVIATION_a,
        "home_pts": g.PTS_h.astype(float), "away_pts": g.PTS_a.astype(float),
        "neutral": False,
    })


def team_box(tg):
    """Per team-game efficiency and four factors."""
    cols = ["GAME_ID", "TEAM_ID", "PTS", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV"]
    m = tg[cols].merge(tg[cols], on="GAME_ID", suffixes=("", "_o"))
    m = m[m.TEAM_ID != m.TEAM_ID_o]
    poss = 0.5 * ((m.FGA - m.OREB + m.TOV + 0.44 * m.FTA)
                  + (m.FGA_o - m.OREB_o + m.TOV_o + 0.44 * m.FTA_o))
    box = pd.DataFrame({
        "game_id": m.GAME_ID, "team_id": m.TEAM_ID,
        "ortg": 100 * m.PTS / poss, "drtg": 100 * m.PTS_o / poss,
        "efg": (m.FGM + 0.5 * m.FG3M) / m.FGA,
        "opp_efg": (m.FGM_o + 0.5 * m.FG3M_o) / m.FGA_o,
        "tov": m.TOV / poss, "orb": m.OREB / (m.OREB + m.DREB_o), "ftr": m.FTM / m.FGA,
        "win": (m.PTS > m.PTS_o).astype(float),
    })
    box["net"] = box.ortg - box.drtg
    box["pace"] = poss.values
    return box


# ---------------------------------------------------------------- features

def add_elo(games):
    """FiveThirtyEight-style Elo with margin-of-victory multiplier.
    Unplayed games (NaN score) get pre-game ratings but no update."""
    elo, last_season, pre_h, pre_a, post_h, post_a = {}, {}, [], [], [], []
    for r in games.itertuples():
        for t in (r.home_id, r.away_id):
            e = elo.get(t, ELO_MEAN)
            if last_season.get(t, r.season) != r.season:
                e = ELO_CARRY * e + (1 - ELO_CARRY) * ELO_MEAN
            elo[t], last_season[t] = e, r.season
        eh, ea = elo[r.home_id], elo[r.away_id]
        pre_h.append(eh)
        pre_a.append(ea)
        if np.isnan(r.home_pts):
            post_h.append(eh)
            post_a.append(ea)
            continue
        hca = 0 if r.neutral else ELO_HCA
        p = 1 / (1 + 10 ** (-(eh + hca - ea) / 400))
        mov = r.home_pts - r.away_pts
        winner_edge = (eh + hca - ea) * np.sign(mov)
        mult = (abs(mov) + 3) ** 0.8 / (7.5 + 0.006 * winner_edge)
        delta = ELO_K * mult * ((mov > 0) - p)
        elo[r.home_id] += delta
        elo[r.away_id] -= delta
        post_h.append(elo[r.home_id])
        post_a.append(elo[r.away_id])
    games["elo_h"], games["elo_a"] = pre_h, pre_a
    games["elo_h_post"], games["elo_a_post"] = post_h, post_a
    games["elo_diff"] = games.elo_h - games.elo_a + np.where(games.neutral, 0, ELO_HCA)
    games["elo_prob"] = 1 / (1 + 10 ** (-games.elo_diff / 400))
    return games


def add_form(games, box):
    """Pre-game exponentially weighted team form and rest, for home and away sides."""
    sides = []
    for side in ("home", "away"):
        s = games[["game_id", "date", "season", f"{side}_id"]].rename(columns={f"{side}_id": "team_id"})
        sides.append(s)
    long = pd.concat(sides).merge(box, on=["game_id", "team_id"], how="left")
    long = long.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)
    grp = long.groupby("team_id")
    for s in STATS + LAB_STATS:
        # ewm carries the last value through unplayed (NaN) rows; shift makes it pre-game
        long[f"f_{s}"] = grp[s].transform(lambda x: x.ewm(halflife=HALFLIFE, ignore_na=True).mean().shift(1))
    long["rest"] = grp.date.diff().dt.days.clip(upper=5).fillna(5)
    long["gp"] = long.groupby(["team_id", "season"]).cumcount()
    keep = ["game_id", "team_id", "rest", "gp"] + [f"f_{s}" for s in STATS + LAB_STATS]
    for side, sfx in (("home", "_h"), ("away", "_a")):
        f = long[keep].rename(columns={c: c + sfx for c in keep if c not in ("game_id", "team_id")})
        games = games.merge(f.rename(columns={"team_id": f"{side}_id"}), on=["game_id", f"{side}_id"], how="left")
    return games


def load_players():
    """Rated player-games plus each player's latest state and newcomer priors."""
    pg = players.load_player_games(data_files("players"), GAME_TYPES)
    return players.rate_players(add_draft_bucket(pg))


def add_lineups(games, lineups):
    for side, sfx in (("home", "_h"), ("away", "_a")):
        lu = lineups.rename(columns={"team_id": f"{side}_id"})
        lu = lu.rename(columns={c: c + sfx for c in lu.columns if c.startswith("lu_")})
        games = games.merge(lu, on=["game_id", f"{side}_id"], how="left")
    return games


def build(extra_games=None, extra_lineups=None, rated=None):
    tg = load_team_games()
    games = to_games(tg)
    if extra_games is not None:
        games = pd.concat([games, extra_games[~extra_games.game_id.isin(games.game_id)]], ignore_index=True)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games = add_elo(games)
    games = add_form(games, team_box(tg))
    pg = (rated or load_players())[0]
    lineups = players.history_lineups(pg)
    if extra_lineups is not None:
        lineups = pd.concat([lineups, extra_lineups[~extra_lineups.game_id.isin(lineups.game_id)]])
    return add_lineups(games, lineups)


LINEUP_FEATURES = [f"d_lu_{m}" for m in players.METRICS] + ["d_lu_star", "d_lu_min"]


def lab_features(g):
    """Default features plus the optional ones the site's Model Lab can switch on."""
    X = features(g)
    for s in LAB_STATS:
        X[f"d_{s}"] = (g[f"f_{s}_h"] - g[f"f_{s}_a"]).fillna(0)
    return X


def features(g):
    X = pd.DataFrame(index=g.index)
    X["elo_diff"] = g.elo_diff
    for s in STATS:
        X[f"d_{s}"] = g[f"f_{s}_h"] - g[f"f_{s}_a"]
    X["rest_h"], X["rest_a"] = g.rest_h, g.rest_a
    X["b2b_h"], X["b2b_a"] = (g.rest_h == 1).astype(int), (g.rest_a == 1).astype(int)
    X["gp_min"] = np.minimum(g.gp_h, g.gp_a)
    X["playoffs"] = g.game_id.str[:3].eq("004").astype(int)
    X["neutral"] = g.neutral.astype(int)
    for f in LINEUP_FEATURES:
        X[f] = g[f[2:] + "_h"] - g[f[2:] + "_a"]
    return X.fillna(0)


def make_models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                              min_samples_leaf=100, l2_regularization=1.0),
    }


def fit_final(train):
    """Logistic alone: best log loss in backtest, and the gbm blend added nothing."""
    X, y = features(train), (train.home_pts > train.away_pts).astype(int)
    models = {"logistic": make_models()["logistic"]}
    for m in models.values():
        m.fit(X, y)
    return models


def predict_proba(models, g):
    X = features(g)
    return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)


def training_rows(g):
    return g[g.home_pts.notna() & (g.season > FIRST_SEASON)]


# ---------------------------------------------------------------- commands

def backtest(n_test=6):
    played = training_rows(build())
    seasons = [s for s in sorted(played.season.unique()) if s < current_season()]
    rows = []
    for s in seasons[-n_test:]:
        train, test = played[played.season < s], played[played.season == s]
        y = (test.home_pts > test.away_pts).astype(int).values
        X_tr, X_te = features(train), features(test)
        y_tr = (train.home_pts > train.away_pts).astype(int)
        probs = {"home team": np.full(len(test), y_tr.mean()), "elo": test.elo_prob.values}
        for name, m in make_models().items():
            probs[name] = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        team_only = [c for c in X_tr.columns if c not in LINEUP_FEATURES]
        probs["logistic team-only"] = make_models()["logistic"].fit(
            X_tr[team_only], y_tr).predict_proba(X_te[team_only])[:, 1]
        for name, p in probs.items():
            rows.append({"season": season_str(s), "model": name, "n": len(y),
                         "accuracy": accuracy_score(y, p > 0.5), "log_loss": log_loss(y, p),
                         "brier": brier_score_loss(y, p)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(res.pivot(index="season", columns="model", values="accuracy").round(3).to_string())
    print("\nOverall (all test seasons):")
    overall = res.assign(w=res.n).groupby("model").apply(
        lambda d: pd.Series({c: np.average(d[c], weights=d.w) for c in ("accuracy", "log_loss", "brier")}),
        include_groups=False)
    print(overall.sort_values("log_loss").round(4).to_string())
    return res


def train():
    played = training_rows(build())
    models = fit_final(played)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(models, MODELS / "model.joblib")
    print(f"Trained on {len(played)} games through {played.date.max():%Y-%m-%d}")
    return models


def prepare(start, end, force_out=(), force_in=(), rated=None, models=None):
    """All games through `end` with features; unplayed ones use current rosters and injuries.
    Returns (games, window of games in [start, end) with home_win_prob, roster, unmatched injury names)."""
    season = current_season(start.date())
    team_ids, team_names = team_maps()
    sched = espn.schedule(start.date(), end.date(), season, team_ids, team_names)

    # unplayed games: current roster minus the injury report
    rated = rated or load_players()
    _, latest, priors = rated
    unplayed = sched[~sched.game_id.isin(load_team_games().GAME_ID)]
    roster = add_draft_bucket(espn.rosters(LIVE, team_ids, known_players()))
    injuries = players.fetch_injuries()
    roster, unmatched = players.availability(roster, injuries, force_out, force_in)
    lineups, roster = players.upcoming_lineups(unplayed, roster, latest, priors)

    g = build(sched, lineups, rated)
    window = g[(g.date >= start) & (g.date < end)].copy()
    if models is None:
        path = MODELS / "model.joblib"
        models = joblib.load(path) if path.exists() else train()
    if not window.empty:
        window["home_win_prob"] = predict_proba(models, window)
    return g, window, roster, unmatched


def predict(date=None, days=1, refresh=True, force_out=(), force_in=()):
    if refresh:
        update()
    start = pd.Timestamp(date or dt.date.today())
    end = start + pd.Timedelta(days=days)
    _, window, roster, unmatched = prepare(start, end, force_out, force_in)
    if window.empty:
        print(f"No games scheduled {start:%Y-%m-%d} to {end - pd.Timedelta(days=1):%Y-%m-%d}")
        return window

    # rotation players missing or doubtful for unplayed games
    hurt = roster[(roster.avail < 1) & (roster.exp_min >= 12)]
    surname = hurt.PLAYER_NAME.str.replace(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", regex=True).str.split().str[-1]
    tag = surname + np.where(hurt.avail > 0, "(dtd)", "")
    missing = tag.groupby(hurt.team_id).agg(", ".join)
    played = window.home_pts.notna()
    window["missing"] = np.where(played, "", (
        window.away + ": " + window.away_id.map(missing).fillna("-") + " | "
        + window.home + ": " + window.home_id.map(missing).fillna("-")))

    out = window.assign(
        pick=np.where(window.home_win_prob >= 0.5, window.home, window.away),
        confidence=np.maximum(window.home_win_prob, 1 - window.home_win_prob),
        result=np.where(window.home_pts.isna(), "",
                        window.away + " " + window.away_pts.fillna(0).astype(int).astype(str) + " - "
                        + window.home_pts.fillna(0).astype(int).astype(str) + " " + window.home),
    )
    cols = ["date", "away", "home", "home_win_prob", "elo_prob", "pick", "confidence", "result", "missing"]
    out = out[cols].sort_values(["date", "confidence"], ascending=[True, False])
    print(out.to_string(index=False, formatters={
        "date": "{:%Y-%m-%d}".format, "home_win_prob": "{:.1%}".format,
        "elo_prob": "{:.1%}".format, "confidence": "{:.1%}".format}, justify="left"))
    if unmatched:
        print(f"\nInjury report names not on any roster (ignored): {', '.join(sorted(unmatched))}")
    PREDS.mkdir(exist_ok=True)
    out.to_csv(PREDS / f"predictions_{start:%Y-%m-%d}.csv", index=False)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # accented player names on Windows consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch-history", "update", "backtest", "train", "predict"])
    ap.add_argument("--date", help="first date to predict (default: today)")
    ap.add_argument("--days", type=int, default=1, help="number of days to predict")
    ap.add_argument("--no-refresh", action="store_true", help="predict without re-fetching data")
    ap.add_argument("--out", default="", help="comma-separated players to treat as out")
    ap.add_argument("--in", dest="force_in", default="", help="comma-separated players to treat as playing")
    a = ap.parse_args()
    names = lambda s: [n.strip() for n in s.split(",") if n.strip()]
    if a.command == "fetch-history":
        fetch_history()
    elif a.command == "update":
        update()
    elif a.command == "backtest":
        backtest()
    elif a.command == "train":
        train()
    else:
        predict(a.date, a.days, refresh=not a.no_refresh, force_out=names(a.out), force_in=names(a.force_in))
