"""Situational inputs for the Model Lab (not used by the default model):
travel and time zones, schedule density, altitude, and Massey (opponent-adjusted) ratings.

All are strictly pre-game. Columns are home-vs-away differences, named x_*.
"""
import math

import numpy as np
import pandas as pd

COLUMNS = ["x_travel", "x_tz", "x_dense", "x_load7", "x_altitude", "x_massey"]

ARENAS = {  # lat, lon, UTC offset (standard time), elevation (m); old abbreviations for early seasons
    "ATL": (33.757, -84.396, -5, 320), "BOS": (42.366, -71.062, -5, 6), "BKN": (40.683, -73.975, -5, 10),
    "CHA": (35.225, -80.839, -5, 230), "CHI": (41.881, -87.674, -6, 180), "CLE": (41.497, -81.688, -5, 200),
    "DAL": (32.790, -96.810, -6, 140), "DEN": (39.749, -105.008, -7, 1609), "DET": (42.341, -83.055, -5, 190),
    "GSW": (37.768, -122.388, -8, 5), "HOU": (29.751, -95.362, -6, 15), "IND": (39.764, -86.156, -5, 220),
    "LAC": (33.945, -118.343, -8, 30), "LAL": (34.043, -118.267, -8, 90), "MEM": (35.138, -90.051, -6, 80),
    "MIA": (25.781, -80.188, -5, 2), "MIL": (43.045, -87.917, -6, 180), "MIN": (44.979, -93.276, -6, 260),
    "NOP": (29.949, -90.082, -6, 2), "NYK": (40.751, -73.994, -5, 10), "OKC": (35.463, -97.515, -6, 370),
    "ORL": (28.539, -81.384, -5, 30), "PHI": (39.901, -75.172, -5, 10), "PHX": (33.446, -112.071, -7, 330),
    "POR": (45.532, -122.667, -8, 15), "SAC": (38.580, -121.500, -8, 9), "SAS": (29.427, -98.438, -6, 200),
    "TOR": (43.643, -79.379, -5, 76), "UTA": (40.768, -111.901, -7, 1288), "WAS": (38.898, -77.021, -5, 20),
    "NJN": (40.734, -74.171, -5, 10), "NOH": (29.949, -90.082, -6, 2),
}


def km(a, b):
    (la1, lo1), (la2, lo2) = a[:2], b[:2]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    return 6371 * math.acos(min(1, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))


def add_schedule(g):
    """Travel since each team's previous game, time-zone shift, 3-in-4 nights, games in 7 days, altitude.
    Games are located at the home team's arena (neutral sites are rare and ignored)."""
    long = pd.concat([g[["game_id", "date", "home"]].assign(team=g.home_id, side="h"),
                      g[["game_id", "date", "home"]].assign(team=g.away_id, side="a")]).sort_values(["team", "date", "game_id"])
    loc = long.home.map(lambda h: ARENAS.get(h, ARENAS["NYK"]))
    prev = loc.groupby(long.team).shift(1)
    long["travel"] = [0.0 if not isinstance(p, tuple) else km(p, c) for p, c in zip(prev, loc)]
    long["tz"] = [0 if not isinstance(p, tuple) else c[2] - p[2] for p, c in zip(prev, loc)]
    d = long.date.values.astype("datetime64[D]").astype(int)
    team = long.team.values
    for days in (4, 7):
        long[f"g{days}"] = [int(np.sum((d[max(0, i - 6):i] > d[i] - days) & (team[max(0, i - 6):i] == team[i]))) for i in range(len(long))]
    h, a = long[long.side == "h"].set_index("game_id"), long[long.side == "a"].set_index("game_id")
    g = g.copy()
    g["x_travel"] = (g.game_id.map(a.travel) - g.game_id.map(h.travel)) / 1000         # thousand km, away minus home
    g["x_tz"] = g.game_id.map(a.tz).abs() - g.game_id.map(h.tz).abs()
    g["x_dense"] = (g.game_id.map(a.g4) >= 2).astype(int) - (g.game_id.map(h.g4) >= 2).astype(int)
    g["x_load7"] = g.game_id.map(a.g7) - g.game_id.map(h.g7)
    g["x_altitude"] = g.home.map(lambda t: ARENAS.get(t, (0, 0, 0, 0))[3] > 1000).astype(int)
    return g


def add_massey(g, halflife_days=45, window_days=400, ridge=2.0):
    """Opponent-adjusted ratings: for each game day, a recency-weighted ridge regression of every prior
    game's margin (capped at 30) on team indicators plus home court. x_massey = home minus away rating."""
    played = g[g.home_pts.notna()]
    teams = sorted(set(g.home_id) | set(g.away_id))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = played.home_id.map(ix).values, played.away_id.map(ix).values
    margin = (played.home_pts - played.away_pts).clip(-30, 30).values
    days = played.date.values.astype("datetime64[D]").astype(int)
    P = np.eye(n + 1) * ridge
    P[n, n] = 0
    rating = {}
    for day in np.unique(g.date.values.astype("datetime64[D]").astype(int)):
        past = (days < day) & (days >= day - window_days)
        if past.sum() < 50:
            continue
        w = 0.5 ** ((day - days[past]) / halflife_days)
        A = np.zeros((past.sum(), n + 1))
        r = np.arange(past.sum())
        A[r, hi[past]], A[r, ai[past]], A[:, n] = 1, -1, 1
        rating[day] = np.linalg.solve(A.T @ (A * w[:, None]) + P, A.T @ (margin[past] * w))
    gd = g.date.values.astype("datetime64[D]").astype(int)
    g = g.copy()
    g["x_massey"] = [rating[d][ix[h]] - rating[d][ix[a]] if d in rating else 0.0 for d, h, a in zip(gd, g.home_id, g.away_id)]
    return g
