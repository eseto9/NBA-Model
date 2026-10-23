"""Lineup stints from ESPN play-by-play: who was on the floor, for how long, and the score.

A stint is a stretch of a game with the same ten players on the court. ESPN's play-by-play
logs every substitution with athlete ids ("X enters the game for Y": participants [in, out]),
and the box score flags starters. Lineup changes between periods are often not logged, so
each period's starting five is inferred: a player who appears in a play (or is subbed out)
before being subbed in that period was on the floor when it began.

Output: one row per stint with seconds, points for each side and the ten ESPN athlete ids.
"""
import pandas as pd

STINT_COLS = ["espn_id", "period", "secs", "home_pts", "away_pts"] + [f"h{i}" for i in range(5)] + [f"a{i}" for i in range(5)]


def clock_secs(text):
    m, _, s = str(text).partition(":")
    try:
        return int(m) * 60 + float(s or 0) if _ else float(m)
    except ValueError:
        return 0.0


def period_len(p):
    return 720 if p <= 4 else 300


def parse(summary, eid):
    """(stint rows, athlete rows) for one game, or ([], []) if the play-by-play is unusable."""
    comp = summary["header"]["competitions"][0]
    side = {c["id"]: c["homeAway"] for c in comp["competitors"]}
    team_of, starters, athletes = {}, {"home": [], "away": []}, []
    for blk in summary["boxscore"]["players"]:
        h = side.get(blk["team"]["id"])
        for a in blk["statistics"][0]["athletes"]:
            aid = a["athlete"]["id"]
            team_of[aid] = h
            athletes.append({"espn_id": eid, "athlete": aid, "name": a["athlete"]["displayName"], "side": h,
                             "team": blk["team"]["abbreviation"]})
            if a.get("starter"):
                starters[h].append(aid)
    if len(starters["home"]) != 5 or len(starters["away"]) != 5:
        return [], athletes
    plays = summary.get("plays") or []
    periods = sorted({p["period"]["number"] for p in plays})
    on = {k: list(v) for k, v in starters.items()}
    rows = []
    for per in periods:
        pp = [p for p in plays if p["period"]["number"] == per]
        if per > 1:
            on = {s: infer_starters(pp, team_of, s, on[s]) for s in ("home", "away")}
        t0, hs0, as0 = 0.0, pp[0].get("homeScore", 0), pp[0].get("awayScore", 0)
        hs0, as0 = score_before(plays, pp[0])
        hs, as_ = hs0, as0
        for p in pp:
            t = period_len(per) - clock_secs(p.get("clock", {}).get("displayValue", "0:00"))
            if p.get("type", {}).get("text") == "Substitution" and len(p.get("participants", [])) >= 2:
                s = team_of.get(p["participants"][0]["athlete"]["id"]) or team_of.get(p["participants"][1]["athlete"]["id"])
                pin, pout = p["participants"][0]["athlete"]["id"], p["participants"][1]["athlete"]["id"]
                if s and pout in on[s] and pin not in on[s]:
                    if t > t0 or (hs, as_) != (hs0, as0):
                        rows.append(stint(eid, per, t - t0, hs - hs0, as_ - as0, on))
                        t0, hs0, as0 = t, hs, as_
                    on[s] = [pin if x == pout else x for x in on[s]]
            hs, as_ = p.get("homeScore", hs), p.get("awayScore", as_)
        rows.append(stint(eid, per, period_len(per) - t0, hs - hs0, as_ - as0, on))
    return rows, athletes


def score_before(plays, first):
    i = plays.index(first)
    return (plays[i - 1].get("homeScore", 0), plays[i - 1].get("awayScore", 0)) if i else (0, 0)


def infer_starters(pp, team_of, s, previous):
    seen, subbed_in = [], set()
    for p in pp:
        ids = [x["athlete"]["id"] for x in p.get("participants", [])]
        if p.get("type", {}).get("text") == "Substitution" and len(ids) >= 2:
            pin, pout = ids[0], ids[1]
            if team_of.get(pout) == s and pout not in subbed_in and pout not in seen:
                seen.append(pout)
            if team_of.get(pin) == s:
                subbed_in.add(pin)
            continue
        for a in ids:
            if team_of.get(a) == s and a not in subbed_in and a not in seen:
                seen.append(a)
    seen = seen[:5]
    for a in previous:                              # quiet players carried over from the last period
        if len(seen) == 5:
            break
        if a not in seen and a not in subbed_in:
            seen.append(a)
    return seen if len(seen) == 5 else list(previous)


def stint(eid, per, secs, hp, ap, on):
    return [eid, per, round(secs, 1), hp, ap] + list(on["home"]) + list(on["away"])


def frame(rows):
    return pd.DataFrame(rows, columns=STINT_COLS)
