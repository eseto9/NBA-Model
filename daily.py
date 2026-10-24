"""Daily job: pull last night's box scores from ESPN, publish the next slate, commit and push.

Runs on GitHub Actions every morning (.github/workflows/daily.yml), which then redeploys
the site. It retries until every game from last night is final.

  python daily.py              # run now
  python daily.py --no-push    # build the data but don't commit
"""
import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RETRIES, WAIT_MIN = 6, 20          # up to ~2 hours for games that run very late


def log(msg):
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main(push=True):
    for attempt in range(1, RETRIES + 1):
        r = subprocess.run([sys.executable, "export.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        out = (r.stdout + r.stderr).strip().splitlines()
        print("\n".join(out), flush=True)
        log(f"export attempt {attempt}: exit {r.returncode}")
        if r.returncode == 0:
            break
        if r.returncode != 3:          # a real failure, not just late games
            return r.returncode
        if attempt < RETRIES:
            time.sleep(WAIT_MIN * 60)
    else:
        log("some games still not final; publishing anyway")
    # morning odds snapshot for the slate just published
    o = subprocess.run([sys.executable, "odds.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    log(f"odds: {(o.stdout + o.stderr).strip().splitlines()[-1:] or ''}")
    if not push:
        return 0
    subprocess.run([sys.executable, "injuries.py"], cwd=ROOT, capture_output=True)
    git("add", "site/data", "data/espn", "log/predictions.csv", "log/rankings.csv", "log/odds.csv")
    if git("diff", "--cached", "--quiet").returncode == 0:
        log("no changes to publish")
        return 0
    c = git("commit", "-m", f"Daily update {dt.date.today():%Y-%m-%d}")
    git("pull", "--rebase", "origin", "main")
    p = git("push", "origin", "HEAD:main")
    log(f"commit: {c.returncode}, push: {p.returncode} {p.stderr.strip()}")
    return p.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-push", action="store_true")
    sys.exit(main(push=not ap.parse_args().no_push))
