"""The optimisation behind planner.py, as a function so the season replay (experiments/backtest.py)
can call exactly the same code.

solve() plans every gameweek in `pred` at once: transfers, -4 hits, free transfer banking,
starting XI, captain and chips, under the FPL rules and your own (config.json).
"""
from dataclasses import dataclass, field

import pandas as pd
import pulp

POS_N = {1: 2, 2: 5, 3: 5, 4: 3}
XI_RANGE = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}
CHIPS = ["wildcard", "freehit", "bboost", "3xc"]


@dataclass
class Week:
    gw: int
    chip: str | None
    squad: list          # real squad after this week's transfers
    playing: list        # squad that plays (the free hit squad when free hit is on)
    xi: list
    bench: list          # in autosub order, keeper last
    captain: int
    vice: int
    ins: list
    outs: list
    hits: int
    free_transfers: int  # available at the start of this week
    bank: float
    points: float        # predicted, after captain, chips and hits


@dataclass
class Plan:
    status: str
    weeks: list = field(default_factory=list)


def chip_windows(chip_rules, chips_used, g0):
    """Chips still available: FPL's windows (GW1-19, GW20-38) minus the ones already played."""
    windows = {}
    for c in chip_rules:
        used = any(u["name"] == c["name"] and c["start_event"] <= u["gw"] <= c["stop_event"] and u["gw"] < g0
                   for u in chips_used)
        if not used:
            windows.setdefault(c["name"], []).append((c["start_event"], c["stop_event"]))
    return windows


def solve(pred, players, owned, bank0, ft0, windows, cfg, active=(), time_limit=None, free_first=False):
    """pred: DataFrame (index player id, columns gw<N>) of predicted points.
    players: DataFrame (index player id) with pos (1-4), club, price.
    owned: {player id: selling price} for your 15. active: chips already played this week.
    free_first: unlimited free transfers in the first week (building a squad from scratch)."""
    gws = [int(c[2:]) for c in pred.columns]
    g0 = gws[0]
    pts = lambda i, g: float(pred.at[i, f"gw{g}"]) if i in pred.index else 0.0
    active = set(active)

    # candidate pool: your squad plus the best players per position over the horizon
    allowed = players[~players["club"].isin(cfg["banned_teams"])]
    total = pred.reindex(allowed.index).fillna(0).sum(axis=1)
    pool = set(owned)
    for k in POS_N:
        pool |= set(total[allowed["pos"] == k].nlargest(cfg["players_per_position"]).index)
    for tm in cfg["min_players_from_team"]:  # keep the best options for forced teams
        pool |= set(total[allowed["club"] == tm].nlargest(5).index)
    P = sorted(pool)
    pos = players["pos"].to_dict()
    club = players["club"].to_dict()
    buy = players["price"].to_dict()
    sell = {i: owned.get(i, buy[i]) for i in P}

    m = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    B = lambda name: pulp.LpVariable(name, cat="Binary")
    s = {(i, g): B(f"s_{i}_{g}") for i in P for g in gws}     # in squad
    x = {(i, g): B(f"x_{i}_{g}") for i in P for g in gws}     # starting XI
    c = {(i, g): B(f"c_{i}_{g}") for i in P for g in gws}     # captain
    t3 = {(i, g): B(f"t_{i}_{g}") for i in P for g in gws}    # triple captain bonus
    bb = {(i, g): B(f"b_{i}_{g}") for i in P for g in gws}    # bench player counted by bench boost
    f = {(i, g): B(f"f_{i}_{g}") for i in P for g in gws}     # free hit squad
    tin = {(i, g): B(f"in_{i}_{g}") for i in P for g in gws}
    tout = {(i, g): B(f"out_{i}_{g}") for i in P for g in gws}
    chip = {(k, g): B(f"chip_{k}_{g}") for k in CHIPS for g in gws}
    ft = {g: pulp.LpVariable(f"ft_{g}", 0, 5, cat="Integer") for g in gws + [gws[-1] + 1]}
    used_ft = {g: pulp.LpVariable(f"u_{g}", 0, 5, cat="Integer") for g in gws}
    hits = {g: pulp.LpVariable(f"h_{g}", 0, 15, cat="Integer") for g in gws}
    bank = {g: pulp.LpVariable(f"bank_{g}", 0) for g in gws}

    m += ft[g0] == ft0
    obj = []
    for n, g in enumerate(gws):
        w = cfg["discount"] ** n
        prev = (lambda i: 1 if i in owned else 0) if g == g0 else (lambda i, g=g: s[i, gws[n - 1]])
        prev_bank = bank0 if g == g0 else bank[gws[n - 1]]
        wc, fh, bbc, tc = (chip[k, g] for k in CHIPS)

        # squad changes and money
        for i in P:
            m += s[i, g] == prev(i) + tin[i, g] - tout[i, g]
        m += bank[g] == prev_bank + pulp.lpSum(tout[i, g] * sell[i] - tin[i, g] * buy[i] for i in P)
        n_in = pulp.lpSum(tin[i, g] for i in P)
        m += n_in <= 15 * (1 - fh)  # free hit leaves the real squad untouched
        m += n_in <= used_ft[g] + hits[g] + 15 * wc + (15 if free_first and g == g0 else 0)
        m += used_ft[g] <= ft[g]
        nxt = gws[n + 1] if n + 1 < len(gws) else gws[-1] + 1
        m += ft[nxt] <= ft[g] - used_ft[g] + 1 - wc - fh  # no weekly +1 in a wildcard / free hit week

        # squad rules (real squad and free hit squad)
        for sq, on in ((s, 1), (f, fh)):
            m += pulp.lpSum(sq[i, g] for i in P) == 15 * on
            for k, cnt in POS_N.items():
                m += pulp.lpSum(sq[i, g] for i in P if pos[i] == k) == cnt * on
            for tm in set(club[i] for i in P):
                m += pulp.lpSum(sq[i, g] for i in P if club[i] == tm) <= 3
            for tm, k in cfg["min_players_from_team"].items():
                m += pulp.lpSum(sq[i, g] for i in P if club[i] == tm) >= k * on
        m += pulp.lpSum(f[i, g] * buy[i] for i in P) <= prev_bank + pulp.lpSum(
            (prev(i) * sell[i]) for i in P) + 200 * (1 - fh)

        # lineup, captain and chip effects; the playing squad is the free hit squad when it's on
        for i in P:
            for v in (x, bb):
                m += v[i, g] <= s[i, g] + fh
                m += v[i, g] <= f[i, g] + 1 - fh
            m += x[i, g] + bb[i, g] <= 1
            m += c[i, g] <= x[i, g]
            m += t3[i, g] <= c[i, g]
        m += pulp.lpSum(x[i, g] for i in P) == 11
        for k, (lo, hi) in XI_RANGE.items():
            m += pulp.lpSum(x[i, g] for i in P if pos[i] == k) >= lo
            m += pulp.lpSum(x[i, g] for i in P if pos[i] == k) <= hi
        m += pulp.lpSum(c[i, g] for i in P) == 1
        m += pulp.lpSum(t3[i, g] for i in P) <= tc
        m += pulp.lpSum(bb[i, g] for i in P) <= 4 * bbc
        m += pulp.lpSum(chip[k, g] for k in CHIPS) <= 1

        # 0.01 on the XI keeps the best 11 starting even when bench boost makes the split not matter;
        # bench_weight values bench players a little (autosubs)
        week = pulp.lpSum((1.01 * x[i, g] + c[i, g] + t3[i, g] + bb[i, g] + cfg.get("bench_weight", 0.1) * s[i, g])
                          * pts(i, g) for i in P)
        # transfer_cost: a small charge per free transfer used, so the plan skips moves that only
        # gain a fraction of a point (predictions aren't that precise)
        obj.append(w * week - cfg["hit_cost"] * hits[g] - cfg.get("transfer_cost", 0) * used_ft[g])

    # chip availability and the cost of using one now instead of saving it
    for k in CHIPS:
        for g in gws:
            if not any(a <= g <= b for a, b in windows.get(k, [])):
                m += chip[k, g] == 0
        for a, b in windows.get(k, []):
            in_win = [g for g in gws if a <= g <= b]
            m += pulp.lpSum(chip[k, g] for g in in_win) <= 1
            if b > gws[-1]:  # the window outlasts the plan, so saving the chip still has value
                obj.append(-cfg["chip_reserve_points"][k] * pulp.lpSum(chip[k, g] for g in in_win))
        if active:
            m += chip[k, g0] == (1 if k in active else 0)
    for i in cfg.get("keep", []):  # players you want to hold no matter what
        if i in owned:
            for g in gws:
                m += s[i, g] == 1
    if active & {"wildcard", "freehit"}:  # chip already played and squad built for this week
        for i in P:
            m += tin[i, g0] == 0
    obj.append(cfg["value_of_saved_free_transfer"] * ft[gws[-1] + 1])
    m += pulp.lpSum(obj)
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit or cfg["solver_seconds"]))

    plan = Plan(pulp.LpStatus[m.status])
    v = lambda var: (var.value() or 0) > 0.5
    for g in gws:
        chips_on = [k for k in CHIPS if v(chip[k, g])]
        fh_on = "freehit" in chips_on
        squad = [i for i in P if v(s[i, g])]
        playing = [i for i in P if v(f[i, g] if fh_on else s[i, g])]
        xi = [i for i in playing if v(x[i, g])]
        bench = sorted((i for i in playing if not v(x[i, g])), key=lambda i: (pos[i] == 1, -pts(i, g)))
        cap = next(i for i in xi if v(c[i, g]))
        vice = max((i for i in xi if i != cap), key=lambda i: pts(i, g))
        n_hits = round(hits[g].value())
        points = sum(pts(i, g) for i in xi) + pts(cap, g) * (2 if "3xc" in chips_on else 1)
        if "bboost" in chips_on:
            points += sum(pts(i, g) for i in bench)
        points -= cfg["hit_cost"] * n_hits
        plan.weeks.append(Week(g, chips_on[0] if chips_on else None, squad, playing, xi, bench, cap, vice,
                               [i for i in P if v(tin[i, g])], [i for i in P if v(tout[i, g])], n_hits,
                               round(ft[g].value()), bank[g].value(), points))
    return plan


def check_week(week, players, cfg):
    """Problems with a planned week (empty if it follows every FPL rule and your rules)."""
    pos, club = players["pos"], players["club"]
    pl, xi = week.playing, week.xi
    problems = []
    if len(pl) != 15 or len(xi) != 11:
        problems.append(f"{len(pl)} in squad, {len(xi)} starting")
    for k, n in POS_N.items():
        if sum(pos[i] == k for i in pl) != n:
            problems.append(f"wrong number in position {k}")
        lo, hi = XI_RANGE[k]
        if not lo <= sum(pos[i] == k for i in xi) <= hi:
            problems.append(f"invalid formation (position {k})")
    counts = pd.Series([club[i] for i in pl]).value_counts()
    problems += [f"more than 3 from {tm}" for tm, n in counts.items() if n > 3]
    if any(club[i] in cfg["banned_teams"] for i in pl):
        problems.append("banned team in squad")
    problems += [f"fewer than {k} from {tm}" for tm, k in cfg["min_players_from_team"].items() if counts.get(tm, 0) < k]
    if week.bank < -1e-6:
        problems.append(f"overspent (bank {week.bank:.1f})")
    if week.captain not in xi:
        problems.append("captain not starting")
    return problems
