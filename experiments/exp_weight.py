"""How much should this season's games count? Replay 2025-26 from GW9 (like now): train on earlier
seasons + 2025-26 GW1-8, test on GW9-38."""
import numpy as np
from scipy.stats import spearmanr
from experiments.harness import load, best_xi_points, M

d, feats, extra = load(rebuild=True)
feats = [c for c in d.columns if c in set(feats) | {"venue_total_points_r5"}]
hist = d[~d["future"] & d["total_points"].notna()]
for s, cut in (("2025-26", 8), ("2024-25", 8)):
    tr = hist[(hist.season < s) | ((hist.season == s) & (hist.GW <= cut))]
    te = hist[(hist.season == s) & (hist.GW > cut) & (hist.n_games >= 3)].copy()
    for w in (1.0, 3.0, 6.0):
        m = M.Ensemble(tr, feats, weight=np.where(tr.season == s, w, 1.0))
        te["pred"] = m.predict(te)
        reg = te[te.minutes_r3 >= 60]
        print(f"{s} this-season weight {w}: MAE {np.abs(te.pred-te.total_points).mean():.3f}  sp {spearmanr(reg.pred, reg.total_points)[0]:.3f}  XI {best_xi_points(te):.1f}", flush=True)
