"""Are predictions the right size? Compare predicted vs actual by prediction bucket (walk-forward)."""
import numpy as np, pandas as pd
from experiments.harness import load, M

d, feats, extra = load()
feats = [c for c in d.columns if c in set(feats) | {"venue_total_points_r5"}]
hist = d[~d["future"] & d["total_points"].notna()]
oof = []
for s in ("2023-24", "2024-25", "2025-26"):
    tr, te = hist[hist.season < s], hist[(hist.season == s)].copy()
    te["pred"] = M.Ensemble(tr, feats).predict(te)
    oof.append(te[["season", "pred", "total_points", "n_games", "minutes_r3"]])
o = pd.concat(oof)
o["bucket"] = pd.cut(o.pred, [-1, 1, 2, 3, 4, 5, 6, 8, 20])
print(o.groupby("bucket", observed=True).agg(n=("pred", "size"), predicted=("pred", "mean"), actual=("total_points", "mean")).round(2))
o.to_pickle("data/oof.pkl")
