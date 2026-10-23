"""Record a late scratch (or a return) so the next export treats the player accordingly.

Overrides go in data/overrides.csv and apply to today's and tomorrow's games; the most recent
entry for a player wins. Run from GitHub: Actions -> "Late scratch" -> Run workflow, which also
republishes the slate. Locally:

  python scratch.py --out "Jalen Brunson, Mitchell Robinson" [--in "Josh Hart"] [--date 2026-10-24]
"""
import argparse
import datetime as dt
import os
import sys
from zoneinfo import ZoneInfo

import pandas as pd

import espn
import nba
import players

OVERRIDES = nba.ROOT / "data" / "overrides.csv"


def main(out, back, date):
    ids, _ = nba.team_maps()
    roster = espn.rosters(nba.LIVE, ids, nba.known_players())
    known = dict(zip(roster.PLAYER_NAME.map(players.norm_name), roster.PLAYER_NAME))
    rows, report = [], []
    for status, names in (("out", out), ("in", back)):
        for n in names:
            match = known.get(players.norm_name(n))
            report.append(f"- {n}: {'**' + status + '**' if match else 'not found on any roster (check spelling)'}")
            if match:
                rows.append({"date": date, "player": match, "status": status,
                             "added_at": dt.datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes")})
    if rows:
        old = pd.read_csv(OVERRIDES) if OVERRIDES.exists() else None
        pd.concat([old, pd.DataFrame(rows)]).to_csv(OVERRIDES, index=False)
    text = f"### Late scratches for {date}\n" + "\n".join(report)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if rows else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="comma-separated players who won't play")
    ap.add_argument("--in", dest="back", default="", help="comma-separated players who will play after all")
    ap.add_argument("--date", default=f"{dt.datetime.now(ZoneInfo('America/New_York')).date()}")
    a = ap.parse_args()
    split = lambda s: [n.strip() for n in s.split(",") if n.strip()]
    sys.exit(main(split(a.out), split(a.back), a.date))
