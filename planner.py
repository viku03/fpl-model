"""Plan transfers, captains and chips over the next few gameweeks for the squad in squad.json.

Usage: python planner.py

Reads predictions.csv (from model.py), squad.json (from state.py or edited by hand),
config.json (your rules) and data/boot.json (prices, chip windows), solves with solver.py
and writes reports/plan-GW<n>-<date>.md with the stats behind every decision.
Chips run per half of the season (GW1-19 and GW20-38); unused ones are lost at GW19.
"""
import json

import pandas as pd

import team_news
from solver import CHIPS, chip_windows, check_week, solve

cfg = json.load(open("config.json"))
state = json.load(open("squad.json"))
boot = json.load(open("data/boot.json"))
pred = pd.read_csv("predictions.csv", index_col=0)

team = {t["id"]: t["short_name"] for t in boot["teams"]}
els = {e["id"]: e for e in boot["elements"]}
GWS = [int(c[2:]) for c in pred.columns]
g0 = GWS[0]
owned = {p["id"]: p["sell"] for p in state["squad"]}
assert len(owned) == 15, "squad.json must list 15 players"

players = pd.DataFrame([dict(id=e["id"], pos=e["element_type"], club=team[e["team"]], price=e["now_cost"] / 10)
                        for e in boot["elements"] if e["status"] not in ("u", "n") or e["id"] in owned]).set_index("id")
windows = chip_windows(boot["chips"], state["chips_used"], g0)
active = {u["name"] for u in state["chips_used"] if u["gw"] == g0}

# config "keep": names of players never to sell (e.g. ["Raya"])
by_name = {els[i]["web_name"]: i for i in owned}
cfg["keep"] = [by_name[n] for n in cfg.get("keep", []) if n in by_name]
plan = solve(pred, players, owned, state["bank"], state["free_transfers"], windows, cfg, active)
print("Solver status:", plan.status)
for w in plan.weeks:
    problems = check_week(w, players, cfg)
    if problems:
        raise SystemExit(f"Plan for GW{w.gw} breaks the rules: {', '.join(problems)}")

POS_NAME = "GKP DEF MID FWD".split()
CHIP_NAME = {"wildcard": "Wildcard", "freehit": "Free hit", "bboost": "Bench boost", "3xc": "Triple captain"}
pos, club, buy = players["pos"], players["club"], players["price"]
pts = lambda i, g: float(pred.at[i, f"gw{g}"]) if i in pred.index else 0.0
horizon_pts = lambda i: sum(pts(i, g) for g in GWS)
name = lambda i: els[i]["web_name"]
try:
    detail = pd.read_csv("predictions_detail.csv").groupby(["element", "GW"]).first()
except FileNotFoundError:
    detail = None


def why(i, g):
    """Short stats line backing up a pick: fixture, odds, recent form."""
    label = f"{name(i)} ({club[i]}, £{buy[i]}m)"
    if detail is None or (i, g) not in detail.index:
        return f"{label}: {pts(i, g):.1f} pts predicted"
    r = detail.loc[(i, g)]
    opp = f"{r.opp} ({'H' if r.was_home else 'A'})"
    form = (f"last 6: {r.total_points_r6:.1f} pts/game, {r.minutes_r6:.0f} mins, "
            f"{r.starts_r6 * 6:.0f}/6 starts") if pd.notna(r.total_points_r6) else "no recent games"
    if pos[i] <= 2:
        role = f"clean sheet chance {r.odds_clean_sheet:.0%}, opponent expected goals {r.odds_opp_goals:.2f}"
    else:
        role = f"xGI {r.expected_goal_involvements_r6:.2f}/game, team expected goals {r.odds_team_goals:.2f}"
    euro = " · European game within 5 days" if (r.euro_prev5 or r.euro_next5) else ""
    return f"{label} vs {opp}: **{pts(i, g):.1f} pts** · {role} · {form}{euro}"


# deadlines in Dublin / UK time (handles GMT vs BST automatically)
dublin = {e["id"]: pd.Timestamp(e["deadline_time"]).tz_convert("Europe/Dublin") for e in boot["events"]}
lines = [f"# ⏰ GW{g0} DEADLINE: {dublin[g0]:%A %d %B, %H:%M} (Dublin / UK time)", "",
         "Make transfers and set your team before this. Later deadlines: "
         + " · ".join(f"GW{g} {dublin[g]:%a %d %b %H:%M}" for g in GWS[1:]), "",
         f"## FPL plan: GW{GWS[0]}-{GWS[-1]}", "",
         f"Made {pd.Timestamp.now(tz='Europe/Dublin'):%a %d %b %H:%M}. "
         f"Bank £{state['bank']}m, {state['free_transfers']} free transfers, "
         f"chips left this half: {', '.join(CHIP_NAME[k] for k in CHIPS if any(a <= g0 <= b for a, b in windows.get(k, []))) or 'none'}.",
         "", "Only act on the first gameweek. Run it again after each gameweek, since everything later will change.", ""]

# team news: is each club's news in yet, and the status of every player this week's plan uses
fixtures = json.load(open("data/fix.json"))
clubs = team_news.club_status(boot, fixtures, g0)
now = pd.Timestamp.now(tz="Europe/Dublin")
for c in clubs.values():
    c["state"] = team_news.news_state(c, dublin[g0], now)
first = plan.weeks[0]
full_name = {t["short_name"]: t["name"] for t in boot["teams"]}
lines += ["## 📰 Team news check", "",
          f"FPL updates injury flags after each club's press conference. Your GW{g0} players "
          "(starting XI, bench, and anyone coming in):", ""]
for i in first.xi + first.bench:
    tag = " (new signing)" if i in first.ins else ""
    lines.append("- " + team_news.player_line(els[i], full_name[club[i]], clubs) + tag)
waiting = sorted({full_name[club[i]] for i in first.xi + first.bench
                  if clubs[full_name[club[i]]]["state"][0] in ("⏳", "⌛")})
lines += ["", ("**Still waiting on:** " + "; ".join(f"{n} ({clubs[n]['state'][1]})" for n in waiting)
               + ". Run it again after the ones marked ⏳ if you can.") if waiting else "**All your clubs' news is in.**", "",
          "FPL only timestamps a club when a flag changes, so once the usual time has passed with no update, "
          "it's read as no changes.", "",
          "<details><summary>Every club</summary>", "",
          "| Club | Kicks off | FPL news last updated | This week's news in? | Flagged players |", "|---|---|---|---|---|"]
for club_name, c in sorted(clubs.items(), key=lambda kv: kv[1]["kickoff"] or pd.Timestamp.max.tz_localize("UTC")):
    ko = f"{c['kickoff'].tz_convert('Europe/Dublin'):%a %H:%M}" if c["kickoff"] is not None else "no match"
    upd = f"{c['last_update'].tz_convert('Europe/Dublin'):%a %d %b %H:%M}" if c["last_update"] is not None else "-"
    state_txt = " ".join(c["state"])
    lines.append(f"| {club_name} | {ko} | {upd} | {state_txt} | {c['flagged']} |")
lines += ["", "</details>", ""]

for w in plan.weeks:
    g = w.gw
    free_used = 0 if w.chip in ("wildcard", "freehit") else len(w.ins) - w.hits
    lines += [f"## GW{g}: {w.points:.1f} predicted points", "",
              f"Chip: **{CHIP_NAME[w.chip] if w.chip else 'none'}** · **{len(w.ins)} transfer(s)** · "
              f"free transfers: using {free_used} of {w.free_transfers}, saving {w.free_transfers - free_used} · "
              f"hits {w.hits} (-{cfg['hit_cost'] * w.hits}) · bank after £{w.bank:.1f}m", ""]
    if w.ins:
        lines += ["### Transfers", ""]
        for o, n in zip(sorted(w.outs, key=lambda i: pos[i]), sorted(w.ins, key=lambda i: pos[i])):
            lines += [f"- **OUT** {why(o, g)}", f"- **IN** {why(n, g)}",
                      f"  - Over GW{GWS[0]}-{GWS[-1]}: {horizon_pts(n):.1f} vs {horizon_pts(o):.1f} predicted pts "
                      f"({horizon_pts(n) - horizon_pts(o):+.1f})", ""]
    else:
        lines += ["No transfers: none of the options gain enough to beat saving the transfer.", ""]

    if w.chip == "bboost":
        lines += [f"**Why bench boost now:** the bench is predicted {sum(pts(i, g) for i in w.bench):.1f} pts this week.", ""]
    elif w.chip == "3xc":
        lines += [f"**Why triple captain now:** {name(w.captain)} is predicted {pts(w.captain, g):.1f} pts, "
                  f"so the chip adds about {pts(w.captain, g):.1f}.", ""]
    elif w.chip == "freehit":
        lines += ["**Why free hit now:** this week's best squad beats yours by enough to be worth the chip.", ""]
    elif w.chip == "wildcard":
        lines += ["**Why wildcard now:** rebuilding the squad beats using free transfers by enough to be worth the chip.", ""]

    lines += ["### Team", "", "| Pos | Player | Club | vs | Pts |", "|---|---|---|---|---|"]
    for i in sorted(w.xi, key=lambda i: (pos[i], -pts(i, g))):
        tag = " **(C)**" if i == w.captain else " (V)" if i == w.vice else ""
        opp = detail.loc[(i, g)].opp if detail is not None and (i, g) in detail.index else "-"
        lines.append(f"| {POS_NAME[pos[i] - 1]} | {name(i)}{tag} | {club[i]} | {opp} | {pts(i, g):.1f} |")
    lines += ["", "Bench (in order): " + ", ".join(f"{name(i)} {pts(i, g):.1f}" for i in w.bench), "",
              f"**Captain:** {why(w.captain, g)}", ""]
    if g == g0:
        top = sorted(w.xi, key=lambda i: -pts(i, g))[1:3]
        lines += ["Next best captains: " + ", ".join(f"{name(i)} {pts(i, g):.1f}" for i in top), "",
                  "### Why each starter", ""] + [f"- {why(i, g)}" for i in sorted(w.xi, key=lambda i: (pos[i], -pts(i, g)))] + [""]

total = sum(w.points for w in plan.weeks)
lines += [f"**Total predicted over GW{GWS[0]}-{GWS[-1]}: {total:.1f}**", "",
          "Chips not used in this plan are being saved: the planner only uses one early when it beats the "
          "`chip_reserve_points` in config.json, and that bar drops to 0 as GW19 gets close."]
report = "\n".join(lines)
out = f"reports/plan-GW{g0}-{pd.Timestamp.now():%Y-%m-%d}.md"
open(out, "w").write(report)
print(report)
print(f"\nSaved {out}")
