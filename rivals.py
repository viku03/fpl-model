"""Your mini-league rivals: who owns and captains whom, how far you are behind, and from that how
much risk to take (automatic). Saves data/rivals.json for the planner.

Usage: python rivals.py   (league_id and entry_id come from config.json)

Why ownership matters: your league position only moves on players you own and rivals don't (or
the reverse). Owning a player is worth his expected points whoever else owns him, but players
rivals don't own add swing. When you're behind you want some swing to catch up; when you're ahead
you want less.
"""
import json
import urllib.request

API = "https://fantasy.premierleague.com/api"


def get(path):
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def risk_settings(gap, weeks_left):
    """How much swing to chase, from the gap to first place and the gameweeks left.
    max_differentials: cap on starters owned by at most one rival (1-6).
    risk: how much a differential's big-score chance counts (0 = ignore, 1 = a lot; negative
    when leading, which favours players your rivals own so they can't gain on you)."""
    weeks_left = max(weeks_left, 1)
    per_week = gap / weeks_left  # points a week you need to make up
    if gap <= 0:
        return {"max_differentials": 1, "risk": -0.3}
    return {"max_differentials": int(min(6, max(1, round(per_week + 0.5)))),
            "risk": round(min(1.0, per_week / 6), 2)}


def main():
    cfg = json.load(open("config.json"))
    boot = get("bootstrap-static/")
    last_gw = max(e["id"] for e in boot["events"] if e["finished"] or e["is_current"])
    standings = get(f"leagues-classic/{cfg['league_id']}/standings/")["standings"]["results"]
    me = next(r for r in standings if r["entry"] == cfg["entry_id"])
    rivals = [r for r in standings if r["entry"] != cfg["entry_id"]]
    own, cap = {}, {}
    for r in rivals:
        picks = get(f"entry/{r['entry']}/event/{last_gw}/picks/")["picks"]
        for p in picks:
            own[p["element"]] = own.get(p["element"], 0) + 1
            if p["is_captain"]:
                cap[p["element"]] = cap.get(p["element"], 0) + 1
    n = len(rivals)
    leader = max(r["total"] for r in rivals) if rivals else me["total"]
    weeks_left = 38 - last_gw
    out = dict(league=cfg["league_id"], gameweek=last_gw, rivals=n, my_total=me["total"], leader_total=leader,
               gap=leader - me["total"], weeks_left=weeks_left,
               ownership={str(k): v / n for k, v in own.items()},
               captaincy={str(k): v / n for k, v in cap.items()},
               **risk_settings(leader - me["total"], weeks_left))
    json.dump(out, open("data/rivals.json", "w"), indent=1)
    print(f"Saved data/rivals.json: {n} rivals, {out['gap']} pts behind first with {weeks_left} weeks left "
          f"-> up to {out['max_differentials']} differentials, risk {out['risk']}")


if __name__ == "__main__":
    main()
