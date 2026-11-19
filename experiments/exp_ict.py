"""Threat (shots), creativity (chance creation), influence as separate features instead of only ICT."""
import pandas as pd
from experiments.harness import load, evaluate, M, Model

d, feats, extra = load()
for c in ("threat", "creativity", "influence"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d, f = M.rolling_features(d, ["season", "element"], ["threat", "creativity", "influence"], (1, 3, 6))
P1, P2 = M.PARAMS, M.HUBER
mk = lambda tr, ff: Model(tr, ff, params=(P1, P2), rounds=(600, 600), seeds=3)
evaluate(d, feats, mk, "current")
evaluate(d, feats + f, mk, "+ threat / creativity / influence")
