"""ESPN's public NBA feed: box scores, schedule, rosters and injuries.

stats.nba.com blocks cloud servers, so the season in progress comes from here while
past seasons come from stats.nba.com (fetched once, stored in data/nba). ESPN rows are
written in the same column layout as the NBA logs, with ESPN teams and players mapped
onto NBA ids so ratings carry over:

  data/espn/<season>/<date>.games.csv     one row per team per game
  data/espn/<season>/<date>.players.csv   one row per player per game
  data/espn/<season>/<date>.odds.csv      betting lines ESPN shows for each game (pickcenter)
  data/espn/ids.csv                       ESPN athlete id -> NBA player id
"""
import datetime as dt
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")
ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
TYPE_PREFIX = {2: "002", 3: "004", 5: "005"}       # regular season, playoffs, play-in
FINAL_STATES = {"STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED"}
TEAM_COLS = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
             "PTS", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV"]
PLAYER_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS", "FGM", "FGA",
               "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]


def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry espn {path}: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not fetch espn {path}")


def nba_abbr(a):
    return ABBR.get(a, a)


def et_day(iso_utc):
    return dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(ET)


def game_id(ev):
    prefix = TYPE_PREFIX.get(ev["season"]["type"])
    return f"{prefix}E{ev['id']}" if prefix else None


def scoreboard(day):
    return get("scoreboard", dates=f"{day:%Y%m%d}", limit=100).get("events", [])


# ---------------------------------------------------------------- player ids

class PlayerIds:
    """Maps ESPN athletes onto NBA player ids by name, remembering every decision."""

    def __init__(self, live_dir, known):
        """known: DataFrame(PLAYER_ID, PLAYER_NAME, GAME_DATE) of players' latest NBA appearance."""
        from players import norm_name
        self.norm = norm_name
        self.path = live_dir / "ids.csv"
        self.map = {}
        if self.path.exists():
            ids = pd.read_csv(self.path)
            self.map = dict(zip(ids.espn_id.astype(str), ids.player_id))
        # only match players active in the last four years, so a rookie never inherits a retiree's history
        recent = known[known.GAME_DATE >= known.GAME_DATE.max() - pd.Timedelta(days=1500)]
        recent = recent.sort_values("GAME_DATE").drop_duplicates("PLAYER_NAME", keep="last")
        self.by_name = dict(zip(recent.PLAYER_NAME.map(norm_name), recent.PLAYER_ID))
        self.new = []

    def __call__(self, espn_id, name):
        espn_id = str(espn_id)
        if espn_id not in self.map:
            pid = self.by_name.get(self.norm(name), -int(espn_id))
            self.map[espn_id] = pid
            self.new.append({"espn_id": espn_id, "player_id": pid, "name": name})
        return self.map[espn_id]

    def save(self):
        if self.new:
            old = pd.read_csv(self.path) if self.path.exists() else None
            pd.concat([old, pd.DataFrame(self.new)]).to_csv(self.path, index=False)
            self.new = []


# ---------------------------------------------------------------- box scores

def stat(stats, labels, key):
    return stats[labels.index(key)] if key in labels else "0"


def split(v):
    m, a = (v or "0-0").split("-")
    return int(m), int(a)


def box_score(ev, team_ids, team_names, ids):
    """(team rows, player rows, betting lines) for one final game, in NBA log layout."""
    gid, day = game_id(ev), et_day(ev["date"])
    season_start = ev["season"]["year"] - 1
    comp = ev["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    summary = get("summary", event=ev["id"])
    lines = [{"date": f"{day:%Y-%m-%d}", "home": nba_abbr(sides["home"]["team"]["abbreviation"]),
                        "away": nba_abbr(sides["away"]["team"]["abbreviation"]), "provider": p.get("provider", {}).get("name"),
                        "home_ml": p["homeTeamOdds"]["moneyLine"], "away_ml": p["awayTeamOdds"]["moneyLine"]}
                       for p in summary.get("pickcenter") or []
                       if p.get("homeTeamOdds", {}).get("moneyLine") is not None and p.get("awayTeamOdds", {}).get("moneyLine") is not None]
    prows, teams = [], {}
    for block in summary["boxscore"]["players"]:
        abbr = nba_abbr(block["team"]["abbreviation"])
        tid = team_ids[abbr]
        st = block["statistics"][0]
        labels = st["labels"]
        for a in st["athletes"]:
            s = a.get("stats") or []
            if not s or a.get("didNotPlay"):
                continue
            mins = float(stat(s, labels, "MIN") or 0)
            if mins <= 0:
                continue
            fgm, fga = split(stat(s, labels, "FG"))
            fg3m, _ = split(stat(s, labels, "3PT"))
            ftm, fta = split(stat(s, labels, "FT"))
            ath = a["athlete"]
            prows.append({
                "PLAYER_ID": ids(ath["id"], ath["displayName"]), "PLAYER_NAME": ath["displayName"], "TEAM_ID": tid,
                "GAME_ID": gid, "GAME_DATE": f"{day:%Y-%m-%d}", "MIN": mins, "PTS": int(stat(s, labels, "PTS")),
                "FGM": fgm, "FGA": fga, "FG3M": fg3m, "FTM": ftm, "FTA": fta,
                "OREB": int(stat(s, labels, "OREB")), "DREB": int(stat(s, labels, "DREB")),
                "AST": int(stat(s, labels, "AST")), "STL": int(stat(s, labels, "STL")),
                "BLK": int(stat(s, labels, "BLK")), "TOV": int(stat(s, labels, "TO")),
                "PF": int(stat(s, labels, "PF")), "PLUS_MINUS": int(str(stat(s, labels, "+/-")).replace("+", "") or 0),
            })
        # team turnovers (including team-only turnovers) from the team box
        tstats = next((t["statistics"] for t in summary["boxscore"]["teams"]
                       if nba_abbr(t["team"]["abbreviation"]) == abbr), [])
        tov = next((s.get("displayValue") for s in tstats if s.get("name") == "totalTurnovers"), None)
        teams[abbr] = {"tid": tid, "tov": int(tov) if tov not in (None, "") else None}
    p = pd.DataFrame(prows)
    trows = []
    for side, opp in (("home", "away"), ("away", "home")):
        c, o = sides[side], sides[opp]
        abbr, oabbr = nba_abbr(c["team"]["abbreviation"]), nba_abbr(o["team"]["abbreviation"])
        mine = p[p.TEAM_ID == team_ids[abbr]]
        pts, opts = int(c["score"]), int(o["score"])
        trows.append({
            "SEASON_ID": f"2{season_start}", "TEAM_ID": team_ids[abbr], "TEAM_ABBREVIATION": abbr,
            "TEAM_NAME": team_names.get(abbr, c["team"]["displayName"]), "GAME_ID": gid, "GAME_DATE": f"{day:%Y-%m-%d}",
            "MATCHUP": f"{abbr} vs. {oabbr}" if side == "home" else f"{abbr} @ {oabbr}",
            "WL": "W" if pts > opts else "L", "PTS": pts,
            **{k: int(mine[k].sum()) for k in ("FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB")},
            "TOV": teams.get(abbr, {}).get("tov") or int(mine.TOV.sum()),
        })
    return pd.DataFrame(trows, columns=TEAM_COLS), p.reindex(columns=PLAYER_COLS + ["FG3M"]), lines


def update(live_dir, season, team_ids, team_names, known_players, through=None):
    """Store box scores for every finished day of `season` not stored yet.
    Returns the days that could not be stored because games were not final yet."""
    through = through or (dt.datetime.now(ET).date() - dt.timedelta(days=1))
    ids = PlayerIds(live_dir, known_players)
    sdir = live_dir / str(season)
    sdir.mkdir(parents=True, exist_ok=True)
    day, pending = dt.date(season, 10, 1), []
    end = min(through, dt.date(season + 1, 7, 15))
    while day <= end:
        gpath, ppath = sdir / f"{day}.games.csv", sdir / f"{day}.players.csv"
        if not gpath.exists():
            events = [e for e in scoreboard(day) if game_id(e)]
            if all(e["status"]["type"]["name"] in FINAL_STATES for e in events):
                tframes, pframes = [pd.DataFrame(columns=TEAM_COLS)], [pd.DataFrame(columns=PLAYER_COLS + ["FG3M"])]
                lines = []
                for e in events:
                    if e["status"]["type"]["name"] == "STATUS_FINAL":
                        t, p, game_lines = box_score(e, team_ids, team_names, ids)
                        tframes.append(t)
                        pframes.append(p)
                        lines += game_lines
                pd.DataFrame(lines, columns=["date", "home", "away", "provider", "home_ml", "away_ml"]).to_csv(sdir / f"{day}.odds.csv", index=False)
                pd.concat(pframes).to_csv(ppath, index=False)
                pd.concat(tframes).to_csv(gpath, index=False)   # written last: marks the day complete
                ids.save()
                if len(events):
                    print(f"{day}: {len(events)} games (espn)")
            else:
                pending.append(day)
                print(f"{day}: games not final yet, will retry")
        day += dt.timedelta(days=1)
    return pending


def draft(year):
    """A draft class from ESPN: DataFrame(PLAYER_ID (unknown), PLAYER_NAME, SEASON, PICK)."""
    core = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
    rows = []
    for rnd in requests.get(f"{core}/seasons/{year}/draft/rounds", timeout=30).json().get("items", []):
        for p in rnd.get("picks", []):
            ref = p.get("athlete", {}).get("$ref")
            if not ref:
                continue
            name = requests.get(ref.replace("http://", "https://"), timeout=30).json().get("displayName")
            rows.append({"PLAYER_ID": None, "PLAYER_NAME": name, "SEASON": year, "PICK": p["overall"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- upcoming games

def schedule(start, end, season, team_ids, team_names):
    """Games from `start` up to (not including) `end`, in the model's game-table layout."""
    rows, day = [], start
    while day < end:
        for e in scoreboard(day):
            gid = game_id(e)
            if not gid:
                continue
            comp = e["competitions"][0]
            sides = {c["homeAway"]: nba_abbr(c["team"]["abbreviation"]) for c in comp["competitors"]}
            tip = et_day(e["date"])
            rows.append({
                "game_id": gid, "date": pd.Timestamp(tip.date()), "season": season,
                "home_id": team_ids[sides["home"]], "away_id": team_ids[sides["away"]],
                "home": sides["home"], "away": sides["away"],
                "home_name": team_names.get(sides["home"], sides["home"]), "away_name": team_names.get(sides["away"], sides["away"]),
                "tip": None if e.get("timeValid") is False else f"{tip:%Y-%m-%dT%H:%M}",
                "home_pts": float("nan"), "away_pts": float("nan"), "neutral": bool(comp.get("neutralSite")),
            })
        day += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def rosters(live_dir, team_ids, known_players):
    """Current roster of every team: DataFrame(team_id, PLAYER_NAME, PLAYER_ID)."""
    ids = PlayerIds(live_dir, known_players)
    rows = []
    for t in get("teams")["sports"][0]["leagues"][0]["teams"]:
        team = t["team"]
        abbr = nba_abbr(team["abbreviation"])
        if abbr not in team_ids:
            continue
        for a in get(f"teams/{team['id']}/roster").get("athletes", []):
            rows.append({"team_id": team_ids[abbr], "PLAYER_NAME": a["displayName"], "PLAYER_ID": ids(a["id"], a["displayName"])})
    ids.save()
    return pd.DataFrame(rows)
