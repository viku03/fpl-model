"""Random search over LightGBM settings: pick on 2024-25, confirm on 2025-26."""
import random
import lightgbm as lgb
import numpy as np
from experiments.harness import load, evaluate, M, Model

d, feats, extra = load()
feats = feats + [c for c in extra if c.startswith("venue_")]
random.seed(0)
results = []
for i in range(14):
    p = dict(learning_rate=random.choice([0.01, 0.02, 0.03, 0.05]), num_leaves=random.choice([7, 15, 31, 63]),
             min_child_samples=random.choice([50, 100, 200, 400]), feature_fraction=random.choice([0.5, 0.7, 0.9]),
             lambda_l2=random.choice([1, 5, 20]))
    rounds = int(18 / p["learning_rate"])
    P1, P2 = dict(M.PARAMS, **p), dict(M.HUBER, **{k: v for k, v in p.items() if k != "num_leaves"}, num_leaves=max(7, p["num_leaves"] // 2))
    r = evaluate(d, feats, lambda tr, f: Model(tr, f, params=(P1, P2), rounds=(rounds, rounds)), f"cfg{i} {p}", seasons=("2024-25",), verbose=False)
    results.append((r.spearman[0] * 10 + r.top50[0], p, rounds, r))
    print(f"cfg{i} sp {r.spearman[0]:.3f} top50 {r.top50[0]:.2f} XI {r.bestxi[0]:.1f}  {p}", flush=True)
results.sort(key=lambda x: -x[0])
print("best on 2024-25:", results[0][1])
for score, p, rounds, _ in results[:2]:
    P1, P2 = dict(M.PARAMS, **p), dict(M.HUBER, **{k: v for k, v in p.items() if k != "num_leaves"}, num_leaves=max(7, p["num_leaves"] // 2))
    evaluate(d, feats, lambda tr, f: Model(tr, f, params=(P1, P2), rounds=(rounds, rounds), seeds=3), f"tuned {p}")
evaluate(d, feats, lambda tr, f: Model(tr, f, seeds=3), "current settings, 3 seeds")
