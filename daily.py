"""Daily job: refresh data, publish the next slate, push to GitHub (which redeploys the site).

stats.nba.com blocks cloud servers such as GitHub Actions, so this runs on a home machine
via Windows Task Scheduler (see README). It retries until last night's box scores are in.

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
RETRIES, WAIT_MIN = 8, 20          # up to ~2.5 hours for late box scores
LOG_FILE = ROOT / "log" / "daily.log"


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main(push=True):
    if push:
        r = git("pull", "--rebase", "--autostash")
        log(f"git pull: {r.stdout.strip() or r.stderr.strip()}")
    for attempt in range(1, RETRIES + 1):
        r = subprocess.run([sys.executable, "export.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        out = (r.stdout + r.stderr).strip().splitlines()
        log(f"export attempt {attempt}: exit {r.returncode} - {out[-1] if out else ''}")
        if r.returncode == 0:
            break
        if attempt < RETRIES:
            time.sleep(WAIT_MIN * 60)
    else:
        log("box scores still incomplete; publishing anyway")
        if r.returncode not in (0, 3):
            log("\n".join(out[-20:]))
            return 1
    if not push:
        return 0
    git("add", "site/data", "log/predictions.csv", "log/rankings.csv")
    if git("diff", "--cached", "--quiet").returncode == 0:
        log("no changes to publish")
        return 0
    c = git("commit", "-m", f"Daily update {dt.date.today():%Y-%m-%d}")
    p = git("push", "origin", "HEAD")
    log(f"commit: {c.returncode}, push: {p.returncode} {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}")
    return p.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-push", action="store_true")
    sys.exit(main(push=not ap.parse_args().no_push))
