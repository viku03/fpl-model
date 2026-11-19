"""Other model families vs the tuned LightGBM pair."""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from experiments.harness import load, evaluate, M, Model

d, feats, extra = load()
feats = feats + [c for c in extra if c.startswith("venue_")]
T = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200, feature_fraction=0.5, lambda_l2=20)
P1, P2 = dict(M.PARAMS, **T), dict(M.HUBER, **{k: v for k, v in T.items() if k != "num_leaves"}, num_leaves=7)
lgbm = lambda tr, f: Model(tr, f, params=(P1, P2), rounds=(600, 600), seeds=3)


class Wrap:
    def __init__(self, models, f):
        self.models, self.f = models, f

    def predict(self, te):
        return np.mean([m.predict(te[self.f]) for m in self.models], axis=0)


def cat(tr, f):
    ms = [CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, loss_function=lf, verbose=0, random_seed=1,
                            thread_count=-1).fit(tr[f], tr.total_points) for lf in ("RMSE", "Huber:delta=2")]
    return Wrap(ms, f)


def xgbm(tr, f):
    ms = [xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.5,
                           min_child_weight=50, reg_lambda=20, objective=o, n_jobs=-1, **kw).fit(tr[f], tr.total_points)
          for o, kw in (("reg:squarederror", {}), ("reg:pseudohubererror", {"huber_slope": 2.0}))]
    return Wrap(ms, f)


def rf(tr, f):
    p = dict(boosting="rf", bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.5, num_leaves=63,
             min_child_samples=200, verbose=-1)
    return Wrap([lgb.train(p, lgb.Dataset(tr[f], tr.total_points), 300)], f)


class Blend:
    def __init__(self, parts):
        self.parts = parts

    def predict(self, te):
        return np.mean([p.predict(te) for p in self.parts], axis=0)


if __name__ == "__main__":
  evaluate(d, feats, lgbm, "tuned LightGBM (3 seeds)")
evaluate(d, feats, cat, "CatBoost")
evaluate(d, feats, xgbm, "XGBoost")
evaluate(d, feats, rf, "random forest")
evaluate(d, feats, lambda tr, f: Blend([lgbm(tr, f), cat(tr, f)]), "LightGBM + CatBoost")
evaluate(d, feats, lambda tr, f: Blend([lgbm(tr, f), cat(tr, f), xgbm(tr, f)]), "LightGBM + CatBoost + XGBoost")
