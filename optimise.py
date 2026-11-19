"""Pick the GW9 wildcard squad from predictions.csv (no Liverpool, at least 1 Man Utd, bench boost in GW10)."""
import json
import sys

import pandas as pd
import pulp

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
boot = json.load(open("data/boot.json"))
team = {t["id"]: t["short_name"] for t in boot["teams"]}
pr = pd.read_csv("predictions.csv", index_col=0)
P = [dict(id=e["id"], n=e["web_name"], t=team[e["team"]], pos=e["element_type"], c=e["now_cost"] / 10,
          x=[pr.at[e["id"], f"gw{g}"] if e["id"] in pr.index else 0 for g in (9, 10, 11)])
     for e in boot["elements"] if team[e["team"]] != "LIV"]

m = pulp.LpProblem("fpl", pulp.LpMaximize)
s, x9, c9, x11 = ({p["id"]: pulp.LpVariable(f"{k}{p['id']}", cat="Binary") for p in P} for k in "sxcz")
# GW9: XI + captain. GW10: bench boost so all 15 count, captain approximated by best-XI weighting.
# GW11: normal XI. Small bench weight in GW9 for autosubs.
m += pulp.lpSum(x9[p["id"]] * p["x"][0] + c9[p["id"]] * p["x"][0] + 0.1 * s[p["id"]] * p["x"][0]
                + 0.9 * s[p["id"]] * p["x"][1] + 0.8 * x11[p["id"]] * p["x"][2] for p in P)
m += pulp.lpSum(s.values()) == 15
m += pulp.lpSum(s[p["id"]] * p["c"] for p in P) <= BUDGET
for xi in (x9, x11):
    m += pulp.lpSum(xi.values()) == 11
    for k, lo, hi in ((1, 1, 1), (2, 3, 5), (3, 2, 5), (4, 1, 3)):
        m += pulp.lpSum(xi[p["id"]] for p in P if p["pos"] == k) >= lo
        m += pulp.lpSum(xi[p["id"]] for p in P if p["pos"] == k) <= hi
m += pulp.lpSum(c9.values()) == 1
for i in s:
    m += x9[i] <= s[i]
    m += x11[i] <= s[i]
    m += c9[i] <= x9[i]
for k, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
    m += pulp.lpSum(s[p["id"]] for p in P if p["pos"] == k) == n
for t in team.values():
    m += pulp.lpSum(s[p["id"]] for p in P if p["t"] == t) <= 3
m += pulp.lpSum(s[p["id"]] for p in P if p["t"] == "MUN") >= 1
m.solve(pulp.PULP_CBC_CMD(msg=0))

squad = sorted((p for p in P if s[p["id"]].value() > 0.5),
               key=lambda p: (-x9[p["id"]].value(), p["pos"], -p["x"][0]))
for p in squad:
    role = "XI " if x9[p["id"]].value() > 0.5 else "BEN"
    cap = "C" if c9[p["id"]].value() > 0.5 else " "
    print(role, cap, "GKP DEF MID FWD".split()[p["pos"] - 1], f"{p['n']:16s}", p["t"], p["c"],
          " ".join(f"{v:.1f}" for v in p["x"]))
xi = [p for p in squad if x9[p["id"]].value() > 0.5]
print(f"cost {sum(p['c'] for p in squad):.1f}  GW9 XI+C {sum(p['x'][0] for p in xi) + max(p['x'][0] for p in xi):.1f}"
      f"  GW10 all 15 {sum(p['x'][1] for p in squad):.1f}")
