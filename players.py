"""Player ratings, lineup strength, rosters and injuries.

Each player carries a minutes-weighted, exponentially decayed rating for a few
advanced metrics, shrunk toward newcomer level until they have played enough
minutes. A team's strength for a game is the expected-minutes-weighted average
rating of the players available for it: for past games, the players who
actually played; for upcoming games, the current roster minus the injury report.
"""
import datetime as dt
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

PLAYER_HALFLIFE = 25     # player games
MIN_HALFLIFE = 6         # player games, for expected minutes
PRIOR_MIN = 250          # pseudo-minutes of newcomer-level play blended into each rating
NEW_PLAYER_MIN = 8.0     # expected minutes for a player with no NBA history
LINEUP_SIZE = 10         # top players by expected minutes that make up the lineup
DTD_WEIGHT = 0.5         # day-to-day players count for half their expected minutes
METRICS = ["pie", "net", "gmsc"]

BASE_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS", "FGM", "FGA",
             "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]
ADV_COLS = ["PLAYER_ID", "GAME_ID", "NET_RATING", "PIE", "USG_PCT", "TS_PCT"]


def call(make, label, tries=4):
    for attempt in range(tries):
        try:
            return make()
        except Exception as e:
            print(f"  retry {label}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {label}")


# ---------------------------------------------------------------- data

def fetch_season(season, season_types):
    """Player box scores (base + advanced) for one season."""
    from nba_api.stats.endpoints import playergamelogs

    frames = []
    for st in season_types:
        parts = {}
        for mt in ("Base", "Advanced"):
            parts[mt] = call(lambda: playergamelogs.PlayerGameLogs(
                season_nullable=season, season_type_nullable=st,
                measure_type_player_game_logs_nullable=mt, timeout=120,
            ).get_data_frames()[0], f"players {season} {st} {mt}")
            time.sleep(0.7)
        if parts["Base"].empty:
            continue
        frames.append(parts["Base"][BASE_COLS].merge(parts["Advanced"][ADV_COLS], on=["PLAYER_ID", "GAME_ID"], how="left"))
    return pd.concat(frames, ignore_index=True)


def load_player_games(data_dir, game_types):
    files = sorted(data_dir.glob("players_*.csv"))
    if not files:
        raise SystemExit("No player data yet - run: python nba.py fetch")
    pg = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in files], ignore_index=True)
    pg = pg[(pg.MIN > 0) & pg.GAME_ID.str[:3].isin(game_types)].drop_duplicates(["PLAYER_ID", "GAME_ID"])
    pg["GAME_DATE"] = pd.to_datetime(pg.GAME_DATE.str[:10])
    gmsc = (pg.PTS + 0.4 * pg.FGM - 0.7 * pg.FGA - 0.4 * (pg.FTA - pg.FTM) + 0.7 * pg.OREB
            + 0.3 * pg.DREB + pg.STL + 0.7 * pg.AST + 0.7 * pg.BLK - 0.4 * pg.PF - pg.TOV)
    pg["pie"] = pg.PIE
    pg["net"] = pg.NET_RATING.clip(-100, 100)
    pg["gmsc"] = 36 * gmsc / pg.MIN      # Hollinger game score per 36 minutes
    return pg.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------- ratings

def rate_players(pg):
    """Adds pre-game ratings r_* and expected minutes exp_min to each player-game.
    Returns (pg, latest, priors): latest is each player's state after his last game."""
    alpha = 1 - 0.5 ** (1 / PLAYER_HALFLIFE)
    by = pg.PLAYER_ID
    den = pg.MIN.groupby(by).transform(lambda x: x.ewm(alpha=alpha).sum())
    first = pg.groupby("PLAYER_ID").cumcount() < 20
    priors = {}
    for m in METRICS:
        ok = first & pg[m].notna()
        # newcomer level: minute-weighted average over players' first 20 career games
        priors[m] = np.average(pg.loc[ok, m], weights=pg.loc[ok, "MIN"])
        num = (pg.MIN * pg[m].fillna(priors[m])).groupby(by).transform(lambda x: x.ewm(alpha=alpha).sum())
        pg[f"post_{m}"] = (num + PRIOR_MIN * priors[m]) / (den + PRIOR_MIN)
        pg[f"r_{m}"] = pg[f"post_{m}"].groupby(by).shift(1).fillna(priors[m])
    pg["post_min"] = pg.MIN.groupby(by).transform(lambda x: x.ewm(halflife=MIN_HALFLIFE).mean())
    pg["exp_min"] = pg.post_min.groupby(by).shift(1).fillna(NEW_PLAYER_MIN)

    post = ["post_min"] + [f"post_{m}" for m in METRICS]
    latest = pg.groupby("PLAYER_ID").tail(1)[["PLAYER_ID", "PLAYER_NAME", "GAME_DATE"] + post]
    latest = latest.rename(columns={c: c.replace("post_", "r_") for c in post}).rename(columns={"r_min": "exp_min"})
    return pg, latest, priors


def lineup_strength(rows):
    """rows: one per (game_id, team_id, player) with exp_min, avail and r_* columns.
    Returns one row per (game_id, team_id) with lineup features."""
    rows = rows.assign(w=rows.exp_min * rows.avail).sort_values("w", ascending=False)
    rows = rows[rows.groupby(["game_id", "team_id"]).cumcount() < LINEUP_SIZE]
    keys = [rows.game_id, rows.team_id]
    out = pd.DataFrame({"lu_min": rows.w.groupby(keys).sum()})
    for m in METRICS:
        out[f"lu_{m}"] = (rows.w * rows[f"r_{m}"]).groupby(keys).sum() / out.lu_min
    # best available player, among those expected to play real minutes
    out["lu_star"] = rows[rows.w >= 20].r_gmsc.groupby(keys).max()
    out["lu_star"] = out.lu_star.fillna(out.lu_gmsc)
    return out.reset_index()


def history_lineups(pg):
    """Lineup strength for completed games, from the players who actually played."""
    rows = pg.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id"}).assign(avail=1.0)
    return lineup_strength(rows)


# ---------------------------------------------------------------- upcoming games

def norm_name(name):
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z ]", "", name.replace("-", " "))
    return " ".join(w for w in name.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def fetch_rosters(season, team_ids, cache_dir):
    from nba_api.stats.endpoints import commonteamroster

    path = cache_dir / f"rosters_{dt.date.today():%Y-%m-%d}.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = []
    for t in team_ids:
        frames.append(call(lambda: commonteamroster.CommonTeamRoster(
            team_id=t, season=season, timeout=60).get_data_frames()[0], f"roster {t}"))
        time.sleep(0.6)
    r = pd.concat(frames, ignore_index=True)[["TeamID", "PLAYER", "PLAYER_ID"]]
    r.columns = ["team_id", "PLAYER_NAME", "PLAYER_ID"]
    r.to_csv(path, index=False)
    return r


def fetch_injuries():
    """ESPN injury report -> DataFrame(name, status, detail)."""
    j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries", timeout=30).json()
    rows = []
    for team in j.get("injuries", []):
        for i in team.get("injuries", []):
            d = i.get("details", {})
            rows.append({"name": i["athlete"]["displayName"], "status": i["status"],
                         "detail": " ".join(str(d.get(k, "")) for k in ("type", "detail")).strip(),
                         "return": d.get("returnDate", "")})
    return pd.DataFrame(rows, columns=["name", "status", "detail", "return"])


def availability(roster, injuries, force_out=(), force_in=()):
    """Weight per roster player: 0 out, DTD_WEIGHT day-to-day, 1 healthy."""
    status = dict(zip(injuries.name.map(norm_name), injuries.status))
    keys = roster.PLAYER_NAME.map(norm_name)
    unmatched = set(status) - set(keys)
    st = keys.map(status).fillna("Healthy")
    st[keys.isin({norm_name(n) for n in force_out})] = "Out"
    st[keys.isin({norm_name(n) for n in force_in})] = "Healthy"
    avail = st.map({"Out": 0.0, "Day-To-Day": DTD_WEIGHT}).fillna(1.0)
    return roster.assign(status=st.values, avail=avail.values), unmatched


def upcoming_lineups(games, roster, latest, priors):
    """Lineup strength for unplayed games from roster availability and latest ratings."""
    r = roster.merge(latest.drop(columns=["PLAYER_NAME"]), on="PLAYER_ID", how="left")
    r["exp_min"] = r.exp_min.fillna(NEW_PLAYER_MIN)
    for m in METRICS:
        r[f"r_{m}"] = r[f"r_{m}"].fillna(priors[m])
    sides = pd.concat([games[["game_id", f"{s}_id"]].rename(columns={f"{s}_id": "team_id"}) for s in ("home", "away")])
    rows = sides.merge(r, on="team_id")
    return lineup_strength(rows), r
