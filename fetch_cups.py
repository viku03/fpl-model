"""Download non-Premier League matches (Champions League, Europa, Conference, FA Cup, League Cup)
for English clubs from ESPN into data/cups.csv. Used for fixture congestion features.

Past days are cached in data/cups/ and never refetched; only the last week and the next
6 weeks are refreshed on each run.
"""
import datetime as dt
import json
import time
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

LEAGUES = ["uefa.champions", "uefa.europa", "uefa.europa.conf", "eng.fa", "eng.league_cup"]
START = dt.date(2022, 7, 1)
CACHE = "data/cups"
os.makedirs(CACHE, exist_ok=True)
today = dt.date.today()


def fetch(league, day):
    path = f"{CACHE}/{league}_{day:%Y%m%d}.json"
    fresh = day < today - dt.timedelta(days=7)
    if fresh and os.path.exists(path):
        return path
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={day:%Y%m%d}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=20))
            json.dump(data.get("events", []), open(path, "w"))
            return path
        except Exception:
            time.sleep(2 ** attempt)  # ESPN throttles bursts; back off and retry
    return None


days = [START + dt.timedelta(days=i) for i in range((today + dt.timedelta(days=42) - START).days)]
days = [d for d in days if d.month not in (6,)]  # no club football in June
jobs = [(lg, d) for lg in LEAGUES for d in days]
with ThreadPoolExecutor(6) as ex:
    paths = list(ex.map(lambda j: fetch(*j), jobs))

rows = []
for (league, _), path in zip(jobs, paths):
    if not path:
        continue
    for ev in json.load(open(path)):
        comp = ev["competitions"][0]
        for c in comp["competitors"]:
            rows.append(dict(date=ev["date"], league=league, team=c["team"]["displayName"],
                             status=ev["status"]["type"]["name"]))
df = pd.DataFrame(rows).drop_duplicates()
df.to_csv("data/cups.csv", index=False)
failed = paths.count(None)
print(f"Saved data/cups.csv: {len(df)} team-matches, {df.team.nunique()} teams, failed days {failed}")
if failed:
    print("  Some days could not be downloaded; run again later to fill them in (finished ones are cached)")
