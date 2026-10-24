"""Regularized adjusted plus-minus (RAPM) from lineup stints.

Each stint is one observation: the five home players get +1, the five away players -1, and
the response is the home margin per 100 possessions, weighted by possessions. Ridge
regression (with a home-court intercept) gives each player's on-court impact holding the
other nine players fixed:

    min_b  sum_s w_s (y_s - h - x_s . b)^2 + lambda * ||b - prior||^2

Ratings are refit on the first of every month from stints strictly before that date (this
season so far plus the previous two, older ones down-weighted), so a game is always priced
with ratings fit on earlier data only.
"""
import datetime as dt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

import nba
import players

LAMBDA = 1000.0          # ridge strength, in possessions (tools/tune_rapm.py)
HALFLIFE_DAYS = 730      # weight of older stints halves every two years
WINDOW_SEASONS = 3       # this season plus the two before it
PRIOR_K = 1.0            # shrink toward k x box-score impact (Tipoff Impact) instead of zero (tools/improve.py)
HOME = "__home__"


def load_stints():
    """All stints with dates, ESPN athlete ids mapped to NBA player ids where possible."""
    parts, ath = [], []
    for f in sorted((nba.HIST / "stints").glob("*.csv.gz")):
        if f.name.endswith("_athletes.csv.gz"):
            ath.append(pd.read_csv(f, dtype={"athlete": str, "espn_id": str}))
        else:
            parts.append(pd.read_csv(f, dtype=str))
    for f in sorted(nba.LIVE.glob("*/*.stints.csv")):
        parts.append(pd.read_csv(f, dtype=str))
    for f in sorted(nba.LIVE.glob("*/*.athletes.csv")):
        ath.append(pd.read_csv(f, dtype={"athlete": str, "espn_id": str}))
    st, ath = pd.concat(parts, ignore_index=True), pd.concat(ath, ignore_index=True)
    for c in ("secs", "home_pts", "away_pts"):
        st[c] = st[c].astype(float)
    dates = ath.drop_duplicates("espn_id").set_index("espn_id").date
    st["date"] = pd.to_datetime(st.espn_id.map(dates))
    return st[st.date.notna() & (st.secs > 0)], ath


def athlete_ids(ath):
    """ESPN athlete id -> NBA player id, by normalized name on the same game date."""
    pg = pd.concat([pd.read_csv(f, usecols=["PLAYER_ID", "PLAYER_NAME", "GAME_DATE"]) for f in nba.data_files("players")])
    pg["key"] = pg.GAME_DATE.astype(str).str[:10] + "|" + pg.PLAYER_NAME.map(players.norm_name)
    by_key = dict(zip(pg.key, pg.PLAYER_ID))
    a = ath.drop_duplicates(["athlete", "date"])
    a = a.assign(nba=(a.date.astype(str).str[:10] + "|" + a.name.map(players.norm_name)).map(by_key))
    known = a.dropna(subset=["nba"]).groupby("athlete").nba.agg(lambda s: s.mode().iloc[0])
    espn_ids = pd.read_csv(nba.LIVE / "ids.csv", dtype={"espn_id": str}) if (nba.LIVE / "ids.csv").exists() else None
    m = {k: int(v) for k, v in known.items()}
    if espn_ids is not None:
        for e, p in zip(espn_ids.espn_id, espn_ids.player_id):
            m.setdefault(str(e), int(p))
    return m


def season_pace():
    """Average possessions per 48 minutes, by season (for converting stint seconds to possessions)."""
    tg = nba.load_team_games()
    poss = tg.FGA - tg.OREB + tg.TOV + 0.44 * tg.FTA
    return poss.groupby(tg.SEASON_ID.astype(str).str[-4:].astype(int)).mean().to_dict()


def prepare(st, idmap):
    lineup_cols = [f"h{i}" for i in range(5)] + [f"a{i}" for i in range(5)]
    for c in lineup_cols:
        st[c] = st[c].map(lambda a: idmap.get(str(a), f"e{a}"))       # unmapped: keep the ESPN id
    pace = season_pace()
    season = st.date.dt.year - (st.date.dt.month < 8)
    poss = st.secs / 2880 * season.map(pace).fillna(98)
    st = st.assign(poss=poss, y=100 * (st.home_pts - st.away_pts) / poss.clip(lower=1))
    return st


def fit(st, asof, prior=None):
    """Ratings from stints before `asof` (a date). Returns {player id: rating per 100 possessions}."""
    d = st[(st.date < asof) & (st.date >= asof - pd.Timedelta(days=365 * WINDOW_SEASONS))]
    if len(d) < 1000:
        return {}
    cols = [f"h{i}" for i in range(5)] + [f"a{i}" for i in range(5)]
    ids = pd.unique(d[cols].values.ravel())
    ix = {p: i for i, p in enumerate(ids)}
    n, k = len(d), len(ids)
    rows = np.repeat(np.arange(n), 10)
    colix = np.array([ix[p] for p in d[cols].values.ravel()])
    vals = np.tile(np.r_[np.ones(5), -np.ones(5)], n)
    X = sparse.csr_matrix((vals, (rows, colix)), shape=(n, k))
    w = d.poss.values * 0.5 ** ((asof - d.date).dt.days.values / HALFLIFE_DAYS)
    y = d.y.values.clip(-200, 200)
    mu = np.average(y, weights=w)                                  # home-court intercept
    p0 = np.array([prior.get(p, 0.0) for p in ids]) if prior else np.zeros(k)
    resid = y - mu - X @ p0
    A = (X.T @ sparse.diags(w) @ X + LAMBDA * sparse.identity(k)).tocsc()
    b = spsolve(A, X.T @ (w * resid))
    return dict(zip(ids, p0 + b))


def monthly_ratings(st, dates):
    """{refit date: ratings} for the first of each month spanning `dates`."""
    months = pd.date_range(dates.min().to_period("M").to_timestamp(), dates.max() + pd.offsets.MonthBegin(1), freq="MS")
    out, prev = {}, None
    for m in months:
        r = fit(st, m)
        if r:
            out[m] = prev = r
        elif prev:
            out[m] = prev
    return out


REPLACEMENT = -2.0        # rating for players with no stints yet (rookies, call-ups)
_cache = {}


def _pid(x):
    x = str(x)
    return int(x) if x.lstrip("-").isdigit() else x


def stints_ready():
    """Stints with NBA player ids and possessions, loaded once per process."""
    if "st" not in _cache:
        st, ath = load_stints()
        _cache["st"] = prepare(st, athlete_ids(ath))
    return _cache["st"]


def box_prior(pg, asof):
    """Each player's credibility-weighted Tipoff Impact before `asof`, centred, times PRIOR_K:
    what RAPM shrinks toward, so a player with few stints sits near his box-score value."""
    last = pg[pg.GAME_DATE < asof].sort_values("GAME_DATE").groupby("PLAYER_ID").post_tpi.last()
    return (PRIOR_K * (last - last.mean())).to_dict() if len(last) else {}


def monthly_cached(st, dates, keep_before, pg=None):
    """Monthly walk-forward ratings. Months before `keep_before` (the current season) never change,
    so they are cached in data/nba/rapm_monthly.csv.gz and only this season's months are refit."""
    path, version = nba.HIST / "rapm_monthly.csv.gz", f"{LAMBDA}|{HALFLIFE_DAYS}|{WINDOW_SEASONS}|prior{PRIOR_K if pg is not None else 0}"
    cache = {}
    if path.exists():
        c = pd.read_csv(path, dtype={"player": str})
        if len(c) and (c.version == version).all():
            cache = {pd.Timestamp(m): dict(zip(d.player.map(_pid), d.rating)) for m, d in c.groupby("month")}
    months = pd.date_range(dates.min().to_period("M").to_timestamp(), dates.max() + pd.offsets.MonthBegin(1), freq="MS")
    out, prev, fresh = {}, None, False
    for m in months:
        r = cache.get(m)
        if r is None:
            r = fit(st, m, prior=box_prior(pg, m) if pg is not None else None)
            fresh = fresh or (bool(r) and m < keep_before)
        if r:
            out[m] = prev = r
        elif prev:
            out[m] = prev
    if fresh:
        rows = [(f"{m:%Y-%m-%d}", str(p), round(v, 3), version) for m, r in out.items() if m < keep_before for p, v in r.items()]
        pd.DataFrame(rows, columns=["month", "player", "rating", "version"]).to_csv(path, index=False, compression="gzip")
    return out


def lineup_rapm(rows, monthly):
    """rows: one per (game_id, team_id, player) with PLAYER_ID, date, exp_min, avail; monthly: {month start: ratings}.
    Minutes-weighted RAPM of the top 10 available players, each rated as of the start of the game's month."""
    rows = rows.assign(w=rows.exp_min * rows.avail).sort_values("w", ascending=False)
    rows = rows[rows.groupby(["game_id", "team_id"]).cumcount() < players.LINEUP_SIZE].copy()
    month = pd.to_datetime(rows.date).dt.to_period("M").dt.to_timestamp()
    keys = np.array(sorted(monthly), dtype="datetime64[ns]")
    use = keys[np.clip(np.searchsorted(keys, month.values, side="right") - 1, 0, None)] if len(keys) else month.values
    rows["r"] = REPLACEMENT
    for m in np.unique(use):
        sel = use == m
        rows.loc[sel, "r"] = rows.PLAYER_ID[sel].map(monthly.get(pd.Timestamp(m), {})).fillna(REPLACEMENT).values
    k = [rows.game_id, rows.team_id]
    out = (rows.w * rows.r).groupby(k).sum() / rows.w.groupby(k).sum().replace(0, np.nan)
    return out.fillna(REPLACEMENT).rename("lu_rapm").reset_index()


def history(pg, st, keep_before=None):
    """Lineup RAPM for every past game, from the players who played and monthly walk-forward fits."""
    monthly = monthly_cached(st, pg.GAME_DATE, keep_before, pg) if keep_before is not None else monthly_ratings(st, pg.GAME_DATE)
    rows = pg.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id", "GAME_DATE": "date"}).assign(avail=1.0)
    return lineup_rapm(rows, monthly), monthly


def latest(st, pg=None):
    """Ratings from every stint so far, for pricing upcoming games."""
    asof = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    return fit(st, asof, prior=box_prior(pg, asof) if pg is not None else None)
