"""Build squad.json (your current squad, bank, free transfers, chips left) from your FPL team ID.

Usage: python state.py <fpl_team_id>

Your team ID is the number in the URL of your Points page:
fantasy.premierleague.com/entry/<ID>/event/<GW>

FPL only makes picks public after a deadline, so transfers you have made for the
upcoming gameweek don't show until it starts. In that case edit squad.json by hand.
"""
import json
import sys
import urllib.request

API = "https://fantasy.premierleague.com/api"


def get(path):
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req))


def free_transfers(weeks, chips_by_gw):
    """Free transfers for the next gameweek, replaying the season: 1 from GW2, +1 each week up
    to 5; a wildcard or free hit week keeps the banked ones but doesn't add the weekly +1
    (checked against the FPL app)."""
    ft = 1
    for n, row in enumerate(weeks):
        g = row["event"]
        if n == 0:  # your first gameweek (GW1, or later if you joined late): transfers were unlimited
            ft = 1
        elif chips_by_gw.get(g) not in ("wildcard", "freehit"):
            ft = min(5, max(0, ft - row["event_transfers"]) + 1)
    return ft


def real_squad_gw(last_gw, active_chip):
    """Which gameweek's picks are your real squad: a free hit squad only lasts one week,
    so after a free hit it's the week before."""
    return last_gw - 1 if active_chip == "freehit" else last_gw


def purchase_from_history(history, started_event, now):
    """Price paid for a player held since you joined: his price in your first gameweek."""
    start = [r["value"] for r in history if r["round"] >= started_event]
    return start[0] if start else now


def selling_price(cost, now):
    """FPL selling price in tenths of £m: purchase price plus half of any rise, rounded down.
    A fall is taken in full."""
    return now if now <= cost else cost + (now - cost) // 2


def main(entry_id):
    boot = json.load(open("data/boot.json"))
    els = {e["id"]: e for e in boot["elements"]}
    next_gw = next(e["id"] for e in boot["events"] if e["is_next"])
    last_gw = next_gw - 1

    hist = get(f"entry/{entry_id}/history/")
    entry = get(f"entry/{entry_id}/")
    picks = get(f"entry/{entry_id}/event/{last_gw}/picks/")
    squad_gw = real_squad_gw(last_gw, picks.get("active_chip"))
    if squad_gw != last_gw:
        picks = get(f"entry/{entry_id}/event/{squad_gw}/picks/")
    transfers = get(f"entry/{entry_id}/transfers/")

    chips_by_gw = {c["event"]: c["name"] for c in hist["chips"]}
    ft = free_transfers(hist["current"], chips_by_gw)

    # selling price = purchase price + half of any rise (rounded down)
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if chips_by_gw.get(t["event"]) != "freehit":  # free hit buys are undone after the week
            bought[t["element_in"]] = t["element_in_cost"]
    squad = []
    for p in picks["picks"]:
        i = p["element"]
        now = els[i]["now_cost"]
        cost = bought.get(i)
        if cost is None:  # in the squad since you joined; use their price in that gameweek
            h = json.load(open(f"data/es/{i}.json"))["history"]
            cost = purchase_from_history(h, entry["started_event"], now)
        sell = selling_price(cost, now)
        squad.append(dict(id=i, name=els[i]["web_name"], purchase=cost / 10, sell=sell / 10))

    used = [dict(name=c["name"], gw=c["event"]) for c in hist["chips"]]
    state = dict(entry_id=entry_id, next_gw=next_gw, bank=picks["entry_history"]["bank"] / 10,
                 free_transfers=ft, chips_used=used, squad=squad)
    json.dump(state, open("squad.json", "w"), indent=2, ensure_ascii=False)
    print(f"Saved squad.json: {len(squad)} players, bank £{state['bank']}m, {ft} free transfers, chips used {used}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
