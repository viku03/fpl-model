"""Experiment harness: build the full feature table once (cached), then score model ideas
with walk-forward validation on the last two finished seasons.

    from experiments.harness import load, evaluate
    d, feats = load()
    evaluate(d, feats, make_model)          # make_model(train_df, feats) -> object with .predict(df)
"""
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.argv = sys.argv[:1] + ["6", "--no-validate"]
import model as M  # noqa: E402

CACHE = "data/features.pkl"
TEST_SEASONS = ("2024-25", "2025-26")


def extra_features(d):
    """Candidate features that are cheap to compute on top of model.add_features."""
    extra = []
    # longer window
    d, f = M.rolling_features(d, ["season", "element"], ["total_points", "minutes", "expected_goal_involvements", "bps"], (10,))
    extra += f
    # per-90 rates over the last 6 (a sub's numbers aren't diluted by short appearances)
    mins = d["minutes_r6"].clip(lower=10)
    for c in ("expected_goals", "expected_assists", "expected_goal_involvements", "bps", "defensive_contribution"):
        d[f"{c}_p90_r6"] = d[f"{c}_r6"] / mins * 90
        extra.append(f"{c}_p90_r6")
    # (home / away form tested here first; now part of model.add_features)
    # player's expected goals / assists this match = his share of team xG x bookmaker team goals
    share_g = (d["expected_goals_r6"] / d["team_expected_goals_r8"].clip(lower=0.3)).clip(upper=1)
    share_a = (d["expected_assists_r6"] / d["team_expected_goals_r8"].clip(lower=0.3)).clip(upper=1)
    play = (d["minutes_r6"] / 90).clip(upper=1)
    d["exp_goals_match"] = share_g * d["odds_team_goals"]
    d["exp_assists_match"] = share_a * d["odds_team_goals"]
    d["exp_cs_points"] = d["odds_clean_sheet"] * play * d["pos"].map({1: 4, 2: 4, 3: 1, 4: 0})
    d["exp_attack_points"] = d["exp_goals_match"] * d["pos"].map({1: 10, 2: 6, 3: 5, 4: 4}) + d["exp_assists_match"] * 3
    extra += ["exp_goals_match", "exp_assists_match", "exp_cs_points", "exp_attack_points"]
    return d, extra


def load(rebuild=False):
    if not rebuild and os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    past = pd.concat([M.load_past(s) for s in M.PAST_SEASONS], ignore_index=True)
    d, feats = M.add_features(pd.concat([past, M.load_current()], ignore_index=True))
    d, extra = extra_features(d)
    out = (d, feats, extra)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def best_xi_points(te):
    """Each week, the actual points of the best formation-valid XI by prediction (no budget),
    captain doubled. The closest offline stand-in for 'points per week'."""
    weeks = []
    for _, w in te.groupby(["season", "GW"]):
        w = w.groupby("element").agg(pred=("pred", "sum"), pts=("total_points", "sum"), pos=("pos", "first"))
        w = w.sort_values("pred", ascending=False)
        gk = w[w.pos == 1].head(1)
        out = w[w.pos != 1]
        need = {2: 3, 3: 2, 4: 1}
        picked = pd.concat([out[out.pos == k].head(n) for k, n in need.items()])
        rest = out.drop(picked.index)
        lim = {2: 5, 3: 5, 4: 3}
        for i, r in rest.iterrows():
            if len(picked) >= 10:
                break
            if (picked.pos == r.pos).sum() < lim[r.pos]:
                picked = pd.concat([picked, rest.loc[[i]]])
        xi = pd.concat([gk, picked])
        weeks.append(xi.pts.sum() + xi.sort_values("pred").pts.iloc[-1])
    return float(np.mean(weeks))


def evaluate(d, feats, make_model, label="", seasons=TEST_SEASONS, train_from=None, verbose=True):
    hist = d[~d["future"] & d["total_points"].notna()]
    rows = []
    for s in seasons:
        prior = [p for p in M.PAST_SEASONS if p < s and (train_from is None or p >= train_from)]
        tr = hist[hist.season.isin(prior)]
        te = hist[(hist.season == s) & (hist["n_games"] >= 3)].copy()
        te["pred"] = make_model(tr, feats).predict(te)
        reg = te[te["minutes_r3"] >= 60]
        top = reg.sort_values("pred", ascending=False).groupby("GW").head(50)
        rows.append(dict(season=s, mae=np.abs(te.pred - te.total_points).mean(),
                         rmse=np.sqrt(((te.pred - te.total_points) ** 2).mean()),
                         spearman=spearmanr(reg.pred, reg.total_points)[0],
                         top50=top.total_points.mean(), bestxi=best_xi_points(te)))
    r = pd.DataFrame(rows)
    if verbose:
        print(f"{label:34s} " + "  ".join(f"{s[2:]}: MAE {x.mae:.3f} sp {x.spearman:.3f} top50 {x.top50:.2f} XI {x.bestxi:.1f}"
                                           for s, x in zip(r.season, r.itertuples())))
    return r


class Model:
    def __init__(self, tr, f, params=(M.PARAMS, M.HUBER), rounds=(600, 800), by_pos=False, seeds=1, weight=None):
        self.f, self.by_pos = f, by_pos
        groups = [(k, tr[tr.pos == k]) for k in (1, 2, 3, 4)] if by_pos else [(0, tr)]
        self.models = {}
        for k, part in groups:
            w = weight(part) if weight else None
            ms = []
            for s in range(seeds):
                for p, r in zip(params, rounds):
                    ms.append(lgb.train(dict(p, seed=s), lgb.Dataset(part[f], part["total_points"], weight=w), r))
            self.models[k] = ms

    def predict(self, te):
        if not self.by_pos:
            return np.mean([m.predict(te[self.f]) for m in self.models[0]], axis=0)
        out = np.zeros(len(te))
        for k, ms in self.models.items():
            idx = (te.pos == k).values
            if idx.any():
                out[idx] = np.mean([m.predict(te.loc[idx, self.f]) for m in ms], axis=0)
        return out
