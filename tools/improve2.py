"""Combine the two winners from tools/improve.py: box-prior RAPM and a stronger L2 penalty.

  python tools/improve2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import experiments  # noqa: E402
import improve  # noqa: E402
import nba  # noqa: E402
import rapm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    g0 = experiments.load()
    base = list(nba.features(g0.head(1)).columns)
    st, pg = rapm.stints_ready(), nba.load_players()[0]
    orig = nba.make_models
    for k in (1.0, 1.5):
        g = improve.with_rapm(g0, improve.rapm_variant(st, pg, k=k))
        for C in (0.1, 0.03):
            nba.make_models = lambda C=C: {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))}
            experiments.report(f"box prior k={k}, C={C}", experiments.walk_forward(g, base))
    nba.make_models = orig


if __name__ == "__main__":
    main()
