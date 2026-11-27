"""Calibration by position and for regular starters (walk-forward predictions)."""
import pickle
import pandas as pd
from experiments.harness import load, M

d, feats, extra = load(rebuild=True)
hist = d[~d["future"] & d["total_points"].notna()]
calib = pickle.load(open("data/calibration.pkl", "rb"))
oof = []
for s in ("2023-24", "2024-25", "2025-26"):
    tr, te = hist[hist.season < s], hist[hist.season == s].copy()
    te["raw"] = M.Ensemble(tr, M_feats := [c for c in d.columns if c in set(feats) | {"venue_total_points_r5"}]).predict(te)
    oof.append(te)
o = pd.concat(oof)
o["cal"] = calib.predict(o.raw)
reg = o[o.minutes_r3 >= 60]
print("regular starters, by position:")
print(reg.groupby("pos").agg(n=("raw", "size"), raw=("raw", "mean"), calibrated=("cal", "mean"), actual=("total_points", "mean")).round(2))
top = reg[reg.cal >= 4]
print("starters predicted 4+, by position:")
print(top.groupby("pos").agg(n=("raw", "size"), calibrated=("cal", "mean"), actual=("total_points", "mean")).round(2))
o[["season", "GW", "element", "pos", "raw", "cal", "total_points", "minutes_r3", "n_games"]].to_pickle("data/oof.pkl")
