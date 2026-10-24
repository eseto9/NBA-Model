"""Injury and availability news from two sources:

  - the official NBA injury report (a PDF the league publishes every 15 minutes on game days,
    with Out / Doubtful / Questionable / Probable / Available for every listed player), and
  - ESPN's injury feed (Out / Day-To-Day), which also covers players not on today's report.

The official report wins where both list a player. Players are matched by name key: letters of
the normalized name with suffixes dropped ("Michael Porter Jr." and "PorterJr.,Michael" agree).
"""
import datetime as dt
import io
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
REPORT = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{:%Y-%m-%d}_{}.pdf"
STATUSES = "Available|Out|Questionable|Doubtful|Probable"
ROW = re.compile(rf"([A-Za-z.'\-]+),([A-Za-z.'\-]+)\s+({STATUSES})\b\s*(.*)")
SUFFIX = re.compile(r"(jr|sr|ii|iii|iv)$")
OUT = Path(__file__).parent / "site" / "data" / "injuries.json"


def name_key(name):
    from players import norm_name
    return re.sub(r"[^a-z]", "", norm_name(name))


def _key_last_first(last, first):
    last = SUFFIX.sub("", re.sub(r"[^a-z]", "", last.lower()))
    return re.sub(r"[^a-z]", "", first.lower()) + last


def official_report(now=None, max_back=48):
    """(report time, DataFrame(key, name, status, reason)) from today's latest official report."""
    import pdfplumber
    now = (now or dt.datetime.now(ET)).astimezone(ET)
    t = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
    for _ in range(max_back):
        url = REPORT.format(t, t.strftime("%I_%M%p"))
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException:          # unreachable (e.g. blocked from cloud servers): give up
            break
        if r is not None and r.status_code == 200 and r.content[:4] == b"%PDF":
            rows = []
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for page in pdf.pages:
                    for line in (page.extract_text() or "").splitlines():
                        m = ROW.search(line)
                        if m:
                            last, first, status, reason = m.groups()
                            rows.append({"key": _key_last_first(last, first), "name": f"{first} {last}",
                                         "status": status, "reason": reason.strip(" -")})
            return t, pd.DataFrame(rows, columns=["key", "name", "status", "reason"]).drop_duplicates("key", keep="last")
        t -= dt.timedelta(minutes=15)
        if t.date() < now.date():
            break
    return None, pd.DataFrame(columns=["key", "name", "status", "reason"])


def espn():
    j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries", timeout=30).json()
    rows = []
    for team in j.get("injuries", []):
        for i in team.get("injuries", []):
            d = i.get("details", {})
            rows.append({"key": name_key(i["athlete"]["displayName"]), "name": i["athlete"]["displayName"], "status": i["status"],
                         "reason": " ".join(str(d.get(k, "")) for k in ("type", "detail")).strip(), "updated": i.get("date", "")})
    return pd.DataFrame(rows, columns=["key", "name", "status", "reason", "updated"])


def combined(now=None):
    """Every listed player's status: the official report where it lists him, ESPN otherwise."""
    try:
        when, off = official_report(now)
    except Exception as e:                       # the report is a bonus; ESPN alone still works
        print(f"official injury report unavailable: {e}")
        when, off = None, pd.DataFrame(columns=["key", "name", "status", "reason"])
    es = espn()
    off = off.assign(source="NBA injury report")
    es = es[~es.key.isin(off.key)].assign(source="ESPN")
    return when, pd.concat([off, es], ignore_index=True)


def write_site(now=None):
    """site/data/injuries.json for the browser, which also polls ESPN itself every 5 minutes."""
    when, df = combined(now)
    OUT.write_text(json.dumps({"report_time": when.isoformat(timespec="minutes") if when else None,
                               "fetched_at": dt.datetime.now(ET).isoformat(timespec="minutes"),
                               "players": df[["key", "name", "status", "reason", "source"]].to_dict("records")},
                              separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return when, len(df)


if __name__ == "__main__":
    print(write_site())
