"""Team news for the report: has each club's news come out this week, and each planned
player's injury status. Source: FPL's own injury flags and their update times (FPL updates
them after press conferences).
"""
from datetime import timedelta

import pandas as pd

def club_status(boot, fixtures, gw):
    """Per club: next kickoff, when FPL last updated any of its players' news, and whether that
    was after its previous match (i.e. this week's news is in)."""
    teams = {t["id"]: t for t in boot["teams"]}
    out = {}
    for tid, t in teams.items():
        mine = [f for f in fixtures if f["event"] == gw and tid in (f["team_h"], f["team_a"])]
        prev = [f for f in fixtures if f["event"] and f["event"] < gw and f["kickoff_time"]
                and tid in (f["team_h"], f["team_a"])]
        updates = [pd.Timestamp(e["news_added"]) for e in boot["elements"] if e["team"] == tid and e["news_added"]]
        last = max(updates) if updates else None
        prev_ko = max(pd.Timestamp(f["kickoff_time"]) for f in prev) if prev else None
        kickoff = min(pd.Timestamp(f["kickoff_time"]) for f in mine) if mine else None
        out[t["name"]] = dict(kickoff=kickoff, last_update=last, prev_kickoff=prev_ko,
                              updated=bool(last and prev_ko and last > prev_ko + timedelta(hours=36)),
                              flagged=sum(1 for e in boot["elements"] if e["team"] == tid and e["status"] != "a"))
    return out


def player_line(e, club, clubs):
    """One status line for a planned player."""
    name = e["web_name"]
    chance = e["chance_of_playing_next_round"]
    news = e["news"]
    c = clubs.get(club, {})
    if e["status"] in ("i", "s", "u", "n") or chance == 0:
        return f"❌ **{name}**: out ({news})"
    if e["status"] == "d" or (chance is not None and chance < 100):
        return f"⚠️ **{name}**: {chance}% ({news})"
    sym, txt = c.get("state", ("⏳", "waiting"))
    if sym == "✅":
        return f"✅ **{name}**: no injury flag ({club}: {txt})"
    return f"{sym} **{name}**: no injury flag yet ({club}: {txt})"


def expected_time(kickoff):
    """When a club's news usually reaches FPL: managers speak the day before a Friday or Saturday
    match and two days before a Sunday or Monday one, and FPL updates the flags by about 17:00."""
    if kickoff is None:
        return None
    k = kickoff.tz_convert("Europe/Dublin")
    days_before = 2 if k.dayofweek in (6, 0) else 1
    return (k - timedelta(days=days_before)).replace(hour=17, minute=0, second=0, microsecond=0)


def news_state(c, deadline, now):
    """(symbol, text) for a club: news in, no changes reported, or still waiting (and whether that
    lands before the deadline)."""
    if c["updated"]:
        return "✅", "news in"
    exp = expected_time(c["kickoff"])
    if exp is None:
        return "-", "no match"
    if now >= exp:
        return "✅", "no changes reported since its press conference"
    when = f"{exp:%a %H:%M}"
    if exp > deadline:
        return "⌛", f"usually ~{when}, after the deadline, so you'll decide without it"
    return "⏳", f"expected ~{when}"
