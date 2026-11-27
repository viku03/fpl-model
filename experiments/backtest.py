"""Season replay: play a past season week by week with the planner making every decision,
scored on what actually happened. Used to choose planner settings (config.json).

    PYTHONPATH=. .venv/bin/python -m experiments.backtest preds 2025-26      # build as-of predictions (once)
    PYTHONPATH=. .venv/bin/python -m experiments.backtest run 2025-26        # replay every config in CONFIGS

Predictions for week t only use what was known before t's deadline: form up to t-1, odds only
for matches up to t (later ones come from market ratings), a model trained on earlier seasons.
Simplifications: no injury flags (the model only sees past minutes), selling price uses the
match-time price, and a bench player only comes on if he played.
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1] + ["6", "--no-validate"]
import model as M  # noqa: E402
from solver import chip_windows, solve  # noqa: E402
from state import selling_price  # noqa: E402
from settings import load_config  # noqa: E402

SHORT = {"Liverpool": "LIV", "Man Utd": "MUN"}  # only the clubs your rules mention
BOOT = json.load(open("data/boot.json"))
BASE = load_config()


def preds_path(season):
    return f"data/backtest_{season}.pkl"


def build_preds(season, horizon=6):
    d, feats, _ = pickle.load(open("data/features.pkl", "rb"))
    feats = [c for c in d.columns if c in set(feats)]
    train = d[(d.season < season) & ~d["future"] & d["total_points"].notna()]
    model = M.Ensemble(train, feats)
    calib = pickle.load(open("data/calibration.pkl", "rb"))
    raw = M.load_past(season)
    odds = pd.read_csv("data/odds_goals.csv")
    home = raw[raw.was_home.astype(bool)]
    pair_gw = {(a, b): g for a, b, g in zip(M.key(home.team), M.key(home.opp), home.GW)}
    actual = raw.groupby(["element", "GW"]).agg(pts=("total_points", "sum"), mins=("minutes", "sum"))
    info = raw.sort_values("GW").groupby("element").agg(pos=("pos", "first"), team=("team", "last"))
    out = {"actual": actual, "info": info, "weeks": {}}
    for t in range(1, 39):
        r = raw.copy()
        fut = r.GW >= t
        r.loc[fut, M.STATS] = np.nan
        r["future"] = fut
        played = np.array([pair_gw.get((h, a), 99) <= t for h, a in zip(odds.home, odds.away)])
        known = odds[(odds.season != season).values | played]
        dt, _ = M.add_features(r, odds=known)
        rows = dt[dt["future"] & dt.GW.between(t, t + horizon - 1)].copy()
        rows["pred"] = M.calibrate(calib, model.predict(rows[feats]), rows["pos"])
        pred = rows.pivot_table(index="element", columns="GW", values="pred", aggfunc="sum")
        pred = pred.reindex(columns=range(t, min(t + horizon, 39))).fillna(0)
        price = raw[raw.GW <= t].sort_values("GW").groupby("element")["value"].last() / 10
        out["weeks"][t] = (pred, price, rows.groupby("element")["team"].last())
        print(f"{season} GW{t}: predicted {len(pred)} players", flush=True)
    pickle.dump(out, open(preds_path(season), "wb"))


def score(week, t, actual):
    """Actual points for a planned week: autosubs, captain/vice, chips, hits."""
    pts = lambda i: actual["pts"].get((i, t), 0)
    mins = lambda i: actual["mins"].get((i, t), 0)
    if week.chip == "bboost":
        players = week.playing
    else:
        xi, bench = list(week.xi), list(week.bench)
        pos = week._pos
        for i in [p for p in xi if mins(p) == 0]:
            for b in bench:
                if mins(b) == 0 or (pos[b] == 1) != (pos[i] == 1):
                    continue
                trial = [p for p in xi if p != i] + [b]
                n = {k: sum(pos[p] == k for p in trial) for k in (2, 3, 4)}
                if n[2] >= 3 and n[3] >= 2 and n[4] >= 1:
                    xi, bench = trial, [p for p in bench if p != b]
                    break
        players = xi
    cap = week.captain if mins(week.captain) > 0 else week.vice
    mult = 3 if week.chip == "3xc" else 2
    return sum(pts(i) for i in players) + (mult - 1) * pts(cap) * (cap in players) - 4 * week.hits


def simulate(args):
    season, name, overrides = args
    cfg = dict(BASE, **overrides)
    data = pickle.load(open(preds_path(season), "rb"))
    actual = {"pts": data["actual"]["pts"].to_dict(), "mins": data["actual"]["mins"].to_dict()}
    info = data["info"]
    horizon = cfg["horizon"]
    owned, bank, ft, chips_used, total, log = {}, 100.0, 1, [], 0.0, []
    for t in range(1, 39):
        pred, price, teams = data["weeks"][t]
        pred = pred.iloc[:, :horizon]
        pred.columns = [f"gw{g}" for g in pred.columns]
        ids = sorted(set(pred.index) | set(owned))
        players = pd.DataFrame({"pos": info["pos"].reindex(ids).values,
                                "club": [SHORT.get(teams.get(i, info["team"].get(i)), teams.get(i, info["team"].get(i)))
                                         for i in ids],
                                "price": price.reindex(ids).fillna(4.5).values}, index=ids)
        sell = {i: selling_price(round(p * 10), round(players.price[i] * 10)) / 10 for i, p in owned.items()}
        plan = solve(pred, players, sell, bank, ft, chip_windows(BOOT["chips"], chips_used, t), cfg,
                     time_limit=cfg.get("backtest_seconds", 20), free_first=(t == 1))
        w = plan.weeks[0]
        w._pos = players["pos"].to_dict()
        pts = score(w, t, actual)
        total += pts
        log.append(dict(gw=t, pts=pts, chip=w.chip, transfers=len(w.ins), hits=w.hits))
        if w.chip:
            chips_used.append({"name": w.chip, "gw": t})
        if w.chip != "freehit":
            for i in w.outs:
                owned.pop(i, None)
            for i in w.ins:
                owned[i] = players.price[i]
            if t == 1:
                owned = {i: players.price[i] for i in w.squad}
            bank = w.bank
        used = 0 if (t == 1 or w.chip in ("wildcard", "freehit")) else len(w.ins) - w.hits
        ft = 1 if t == 1 else (ft if w.chip in ("wildcard", "freehit") else min(5, max(0, ft - used) + 1))
    return season, name, total, pd.DataFrame(log)


CONFIGS = {
    "current": {},
    "transfer cost 0.5": {"transfer_cost": 0.5},
    "transfer cost 1.0": {"transfer_cost": 1.0},
    "transfer cost 2.0": {"transfer_cost": 2.0},
    "horizon 3": {"horizon": 3},
    "horizon 1": {"horizon": 1},
    "saved FT worth 3": {"value_of_saved_free_transfer": 3.0},
    "chips used freely": {"chip_reserve_points": {"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 0}},
    "chips held hard": {"chip_reserve_points": {"wildcard": 20, "freehit": 20, "bboost": 12, "3xc": 15}},
    "no discount": {"discount": 1.0},
}

if __name__ == "__main__":
    mode, seasons = ARGS[0], ARGS[1:]
    if mode == "preds":
        for s in seasons:
            build_preds(s)
    else:
        names = ARGS[2:] if len(ARGS) > 2 else list(CONFIGS)
        seasons = ARGS[1].split(",")
        jobs = [(s, n, CONFIGS[n]) for s in seasons for n in names]
        with Pool(min(7, len(jobs))) as pool:
            for season, name, total, log in pool.imap_unordered(simulate, jobs):
                chips = ", ".join(f"{r.chip} GW{r.gw}" for r in log.itertuples() if isinstance(r.chip, str))
                print(f"{season} {name:22s} total {total:6.0f}  avg {total / 38:5.1f}/GW  transfers {log.transfers.sum():3d}"
                      f"  hits {log.hits.sum():2d}  chips: {chips}", flush=True)
                log.to_csv(f"data/backtest_{season}_{name.replace(' ', '_')}.csv", index=False)
