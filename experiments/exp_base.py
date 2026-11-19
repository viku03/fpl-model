"""Baseline: current ensemble, with and without the 2022-23 season in training."""
from experiments.harness import load, evaluate, M

d, feats, extra = load(rebuild=True)
ens = lambda tr, f: type("E", (), {"m": M.Ensemble(tr, f), "predict": lambda self, te: self.m.predict(te[f])})()
evaluate(d, feats, ens, "current, no 2022-23", train_from="2023-24")
evaluate(d, feats, ens, "current + 2022-23")
