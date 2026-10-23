"""Grid over RAPM's ridge strength and recency half-life, scored walk-forward against the market.

  python tools/tune_rapm.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import experiments  # noqa: E402
import nba  # noqa: E402
import rapm  # noqa: E402


def main():
    base_g = experiments.load()
    st, ath = rapm.load_stints()
    st = rapm.prepare(st, rapm.athlete_ids(ath))
    pg = nba.load_players()[0]
    base = list(nba.features(base_g.head(1)).columns)
    for lam, half in [(500, 365), (1000, 365), (4000, 365), (1000, 180), (1000, 730)]:
        rapm.LAMBDA, rapm.HALFLIFE_DAYS = lam, half
        lu, _ = rapm.history(pg, st)
        g = base_g
        for side, sfx in (("home", "_h"), ("away", "_a")):
            g = g.merge(lu.rename(columns={"team_id": f"{side}_id", "lu_rapm": f"lu_rapm{sfx}"}), on=["game_id", f"{side}_id"], how="left")
        g = g.assign(d_lu_rapm=(g.lu_rapm_h - g.lu_rapm_a).fillna(0))
        experiments.report(f"lambda {lam}, half-life {half}d", experiments.walk_forward(g, base + ["d_lu_rapm"]))


if __name__ == "__main__":
    main()
