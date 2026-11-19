"""Each idea on its own against the baseline (all four past seasons)."""
import lightgbm as lgb
import numpy as np
from experiments.harness import load, evaluate, M, Model

if __name__ != "__main__":
    raise SystemExit  # results are in the README
d, feats, extra = load()


groups = {
    "+ 10-game window": [c for c in extra if c.endswith("_r10")],
    "+ per-90 rates": [c for c in extra if "_p90_" in c],
    "+ home/away form": [c for c in extra if c.startswith("venue_")],
    "+ player exp goals/CS from odds": ["exp_goals_match", "exp_assists_match", "exp_cs_points", "exp_attack_points"],
}
evaluate(d, feats, lambda tr, f: Model(tr, f), "baseline")
for name, cols in groups.items():
    evaluate(d, feats + cols, lambda tr, f: Model(tr, f), name)
evaluate(d, feats + extra, lambda tr, f: Model(tr, f), "+ all extra features")
evaluate(d, feats, lambda tr, f: Model(tr, f, by_pos=True), "per-position models")
evaluate(d, feats, lambda tr, f: Model(tr, f, seeds=3), "3 seeds averaged")
haul = lambda part: np.where(part.total_points >= 5, 2.0, 1.0)
evaluate(d, feats, lambda tr, f: Model(tr, f, weight=haul), "hauls (5+) weighted 2x")
recent = lambda part: part.season.map({s: 0.7 ** (len(M.PAST_SEASONS) - 1 - i) for i, s in enumerate(M.PAST_SEASONS)}).values
evaluate(d, feats, lambda tr, f: Model(tr, f, weight=recent), "older seasons faded (0.7/yr)")
