"""Train a LightGBM points model on past FPL seasons and predict the next HORIZON gameweeks.

Usage: python model.py [horizon]   (default 6)
"""
import glob
import json
import re
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POS = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}  # "AM" (2024-25 assistant manager) is dropped
STATS = ["total_points", "minutes", "starts", "expected_goals", "expected_assists",
         "expected_goal_involvements", "expected_goals_conceded", "bps", "bonus",
         "ict_index", "defensive_contribution", "saves", "clean_sheets", "goals_scored", "assists"]
PRIOR = ["total_points", "minutes", "starts", "expected_goal_involvements", "bps"]
PAST_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
CURRENT_SEASON = "2026-27"
HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 6
VALIDATE = "--no-validate" not in sys.argv

try:
    boot = json.load(open("data/boot.json"))
    fixtures = json.load(open("data/fix.json"))
except FileNotFoundError:  # no downloaded data (e.g. CI): the pure functions still import and test
    boot, fixtures = {"events": [{"id": 1, "is_next": True}], "teams": [], "elements": []}, []
next_gw = next(e["id"] for e in boot["events"] if e["is_next"])
FUTURE_GWS = [g for g in range(next_gw, next_gw + HORIZON) if g <= 38]
team_name = {t["id"]: t["name"] for t in boot["teams"]}

# ESPN club names -> FPL club names
ESPN = {"Manchester City": "Man City", "Manchester United": "Man Utd", "Tottenham Hotspur": "Spurs",
        "Nottingham Forest": "Nott'm Forest", "Brighton & Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves",
        "Newcastle United": "Newcastle", "West Ham United": "West Ham", "AFC Bournemouth": "Bournemouth",
        "Leeds United": "Leeds", "Leicester City": "Leicester", "Luton Town": "Luton", "Sheffield United": "Sheffield Utd"}


def prev_season(season):
    a = int(season[:4]) - 1
    return f"{a}-{str(a + 1)[2:]}"


def load_prior(season):
    """Last season's per-90 numbers for each player, keyed by the stable FPL player code.
    Per 90 rather than per start, so a sub with one start doesn't look like a superstar."""
    r = pd.read_csv(f"data/raw_{prev_season(season)}.csv")
    nineties = (r["minutes"] / 90).clip(lower=1)
    out = pd.DataFrame({"code": r["code"]})
    for c in PRIOR:
        if c not in r:  # older files lack some columns (e.g. 2021-22 has no starts or xGI)
            out[f"prior_{c}"] = np.nan
        else:
            out[f"prior_{c}"] = r[c] / 38 if c in ("minutes", "starts") else r[c] / nineties
    out["prior_minutes_total"] = r["minutes"]
    return out


def load_past(season):
    d = pd.read_csv(f"data/gw_{season}.csv")
    d = d[d["position"].isin(POS)].copy()
    d["season"] = season
    d["pos"] = d["position"].map(POS)
    # opponent = the other team playing in the same fixture
    teams = d.groupby("fixture")["team"].unique()
    d["opp"] = [next((t for t in teams[f] if t != tm), None) for f, tm in zip(d["fixture"], d["team"], strict=False)]
    if "defensive_contribution" not in d:
        d["defensive_contribution"] = np.nan
    # 2022-23 only started recording xG in GW16; before that it's stored as 0, which means unknown
    xg_cols = ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
    no_xg = d.groupby("GW")["expected_goals"].transform(lambda s: (s.fillna(0) == 0).all())
    d.loc[no_xg, xg_cols] = np.nan
    code = pd.read_csv(f"data/raw_{season}.csv")[["id", "code"]]
    return d.merge(code, left_on="element", right_on="id", how="left").drop(columns="id")


def load_current():
    rows = []
    els = {e["id"]: e for e in boot["elements"]}
    fx = {f["id"]: f for f in fixtures}
    for path in glob.glob("data/es/*.json"):
        for h in json.load(open(path))["history"]:
            e = els[h["element"]]
            # the club they played for in that match, not their current one (summer transfers)
            f = fx[h["fixture"]]
            if not (f["finished"] or f["finished_provisional"]):
                continue  # live or not yet played: minutes so far would look like a benching
            played_for = f["team_h"] if h["was_home"] else f["team_a"]
            h.update(name=e["web_name"], pos=e["element_type"], team=team_name[played_for], code=e["code"],
                     opp=team_name[h["opponent_team"]], GW=h["round"])
            rows.append(h)
    d = pd.DataFrame(rows)
    d["season"] = CURRENT_SEASON
    # future fixtures as rows with an unknown target
    fut = []
    for f in fixtures:
        if f["event"] not in FUTURE_GWS:
            continue
        for side, other, home in (("team_h", "team_a", True), ("team_a", "team_h", False)):
            for e in boot["elements"]:
                if e["team"] == f[side]:
                    fut.append(dict(element=e["id"], code=e["code"], name=e["web_name"], pos=e["element_type"],
                                    team=team_name[f[side]], opp=team_name[f[other]], GW=f["event"],
                                    fixture=f["id"], was_home=home, kickoff_time=f["kickoff_time"],
                                    value=e["now_cost"], season=CURRENT_SEASON, future=True))
    return pd.concat([d, pd.DataFrame(fut)], ignore_index=True)


def rolling_features(d, keys, cols, windows, prefix=""):
    """Rolling means of past games only. Future rows all get the stats as of the last real game."""
    real = d[~d["future"]].sort_values(keys + ["kickoff_time"])
    g = real.groupby(keys)
    out = {}
    for c in cols:
        for w in windows:
            name = f"{prefix}{c}_r{w}" if w else f"{prefix}{c}_std"
            roll = (g[c].rolling(w, min_periods=1) if w else g[c].expanding()).mean()
            roll = roll.reset_index(level=list(range(len(keys))), drop=True)
            out[name] = roll
    after = pd.DataFrame(out, index=real.index)  # stats including this game
    before = after.groupby([real[k] for k in keys]).shift(1)  # stats before this game
    d = d.join(before)
    last = after.join(real[keys]).groupby(keys).last()
    fut = d["future"]
    d.loc[fut, list(out)] = d.loc[fut, keys].merge(last, left_on=keys, right_index=True, how="left")[list(out)].values
    return d, list(out)


def congestion(d):
    """Days of rest and European / cup games around each Premier League match."""
    pl = d[["season", "team", "kickoff_time"]].drop_duplicates().assign(comp="pl")
    try:
        cups = pd.read_csv("data/cups.csv")
        cups = cups.assign(team=cups["team"].replace(ESPN), kickoff_time=cups["date"],
                           comp=np.where(cups["league"].str.startswith("uefa"), "euro", "cup"))
        cups = cups[cups["team"].isin(set(pl["team"]))][["team", "kickoff_time", "comp"]]
    except FileNotFoundError:
        cups = pd.DataFrame(columns=["team", "kickoff_time", "comp"])
    allm = pd.concat([pl[["team", "kickoff_time", "comp"]], cups], ignore_index=True)
    allm["t"] = pd.to_datetime(allm["kickoff_time"], utc=True, format="ISO8601")
    allm = allm.drop_duplicates(["team", "t"]).sort_values(["team", "t"])
    feats = []
    for team, grp in allm.groupby("team"):
        t = grp["t"].values
        is_euro = (grp["comp"] == "euro").values
        is_cup = (grp["comp"] != "pl").values
        for i in np.where(grp["comp"].values == "pl")[0]:
            day = np.timedelta64(1, "D")
            prev = (t[i] - t[i - 1]) / day if i > 0 else np.nan
            nxt = (t[i + 1] - t[i]) / day if i + 1 < len(t) else np.nan
            win = lambda lo, hi, mask: int(((t - t[i] > lo * day) & (t - t[i] < hi * day) & mask).sum())
            feats.append(dict(team=team, t=grp["t"].iloc[i], rest_days=min(prev, 14), days_to_next=min(nxt, 14),
                              euro_prev5=win(-5, 0, is_euro), euro_next5=win(0, 5, is_euro),
                              cup_prev5=win(-5, 0, is_cup & ~is_euro), cup_next5=win(0, 5, is_cup & ~is_euro),
                              euro_last30=win(-30, 0, is_euro)))
    f = pd.DataFrame(feats)
    d["t"] = pd.to_datetime(d["kickoff_time"], utc=True, format="ISO8601")
    d = d.merge(f, on=["team", "t"], how="left").drop(columns="t")
    return d, ["rest_days", "days_to_next", "euro_prev5", "euro_next5", "cup_prev5", "cup_next5", "euro_last30"]


# FPL and football-data spell some clubs differently; key() maps both to one name for matching
SAME_CLUB = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
             "Hull City": "Hull", "Coventry City": "Coventry", "Ipswich Town": "Ipswich"}


def key(team):
    return team.map(lambda t: SAME_CLUB.get(t, t))


def market_ratings(o, season):
    """Attack and defence ratings for each club from the odds of its recent matches:
    log(expected goals) = home advantage + attack(team) - defence(opponent).
    This season's matches count most, with older ones fading out; last season fills in early on."""
    from sklearn.linear_model import Ridge
    rows = []
    for s, w in ((season, 1.0), (prev_season(season), 0.3)):
        part = o[o["season"] == s].reset_index(drop=True)
        decay = 0.97 ** (len(part) / 10 - np.arange(len(part)) / 10)  # ~10 matches per gameweek
        for r, dw in zip(part.itertuples(), decay, strict=False):
            rows.append((r.home, r.away, 1, np.log(r.home_goals), w * dw))
            rows.append((r.away, r.home, 0, np.log(r.away_goals), w * dw))
    t = pd.DataFrame(rows, columns=["att", "def", "home", "y", "w"])
    X = pd.concat([pd.get_dummies(t["att"], prefix="a"), -pd.get_dummies(t["def"], prefix="d"), t[["home"]]],
                  axis=1).astype(float)
    reg = Ridge(alpha=1.0).fit(X, t["y"], sample_weight=t["w"])

    def predict(att, dfn, home):
        q = pd.DataFrame(0.0, index=range(len(att)), columns=X.columns)
        for i, (a, b, h) in enumerate(zip(att, dfn, home, strict=False)):
            if f"a_{a}" in q:
                q.at[i, f"a_{a}"] = 1
            if f"d_{b}" in q:
                q.at[i, f"d_{b}"] = -1
            q.at[i, "home"] = h
        return np.exp(reg.predict(q))
    return predict


def odds_features(d, odds=None):
    """Bookmaker-implied goals for and against in each match. Matches without odds yet
    (usually anything past next week) are estimated from the market's team ratings."""
    if odds is not None:  # the season replay passes only the odds known at the time
        o = odds
    else:
        try:
            o = pd.read_csv("data/odds_goals.csv")
        except FileNotFoundError:
            o = pd.DataFrame(columns=["season", "home", "away", "home_goals", "away_goals"])
    home = np.where(d["was_home"] == 1, key(d["team"]), key(d["opp"]))
    away = np.where(d["was_home"] == 1, key(d["opp"]), key(d["team"]))
    m = pd.DataFrame({"season": d["season"].values, "home": home, "away": away}).merge(
        o, on=["season", "home", "away"], how="left")
    own = np.where(d["was_home"] == 1, m["home_goals"], m["away_goals"])
    opp = np.where(d["was_home"] == 1, m["away_goals"], m["home_goals"])
    d["odds_team_goals"], d["odds_opp_goals"] = own, opp
    need = d["odds_team_goals"].isna()
    for season in d.loc[need, "season"].unique():
        rows = need & (d["season"] == season)
        predict = market_ratings(o, season)
        team, opp_team, h = key(d.loc[rows, "team"]).values, key(d.loc[rows, "opp"]).values, d.loc[rows, "was_home"].values
        d.loc[rows, "odds_team_goals"] = predict(team, opp_team, h)
        d.loc[rows, "odds_opp_goals"] = predict(opp_team, team, 1 - h)
    d["odds_clean_sheet"] = np.exp(-d["odds_opp_goals"])
    if odds is None:
        print(f"  odds found for {np.isfinite(own).mean():.0%} of rows, rest estimated from market ratings")
    return d, ["odds_team_goals", "odds_opp_goals", "odds_clean_sheet"]


def add_features(d, odds=None):
    d = d.copy()
    d["future"] = d["future"].fillna(False).astype(bool) if "future" in d else False
    for c in STATS + ["value"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["was_home"] = d["was_home"].astype(float)
    d = d.reset_index(drop=True)
    feats = ["pos", "value", "was_home"]

    d, f = rolling_features(d, ["season", "element"], STATS, (1, 3, 6, 0))
    feats += f
    # home and away form separately (some players are much better at home)
    d, f = rolling_features(d, ["season", "element", "was_home"], ["total_points"], (5,), prefix="venue_")
    feats += f
    real = d[~d["future"]]
    n = real.groupby(["season", "element"]).cumcount()
    d["n_games"] = n.reindex(d.index)
    last_n = real.groupby(["season", "element"]).size().rename("cnt")
    fut = d["future"]
    d.loc[fut, "n_games"] = d.loc[fut, ["season", "element"]].merge(
        last_n, left_on=["season", "element"], right_index=True, how="left")["cnt"].fillna(0).values
    feats.append("n_games")

    # last season's numbers, which matter most in the first few weeks
    prior = pd.concat([load_prior(s).assign(season=s) for s in PAST_SEASONS + (CURRENT_SEASON,)])
    d = d.merge(prior, on=["season", "code"], how="left")
    feats += [c for c in prior.columns if c.startswith("prior_")]

    # team attack / defence from summed player xG per fixture (last 8 matches)
    t = (real.groupby(["season", "fixture", "team", "opp", "kickoff_time"])["expected_goals"].sum().reset_index())
    against = t.rename(columns={"team": "opp", "opp": "team", "expected_goals": "xga"})[["season", "fixture", "team", "xga"]]
    t = t.merge(against, on=["season", "fixture", "team"]).assign(future=False)
    futt = d[fut][["season", "fixture", "team", "opp", "kickoff_time"]].drop_duplicates().assign(future=True)
    t = pd.concat([t, futt], ignore_index=True)
    t, tf = rolling_features(t, ["season", "team"], ["expected_goals", "xga"], (8,), prefix="team_")
    tf_df = t[["season", "fixture", "team"] + tf].drop_duplicates(["season", "fixture", "team"])
    d = d.merge(tf_df, on=["season", "fixture", "team"], how="left")
    of = tf_df.rename(columns={"team": "opp", tf[0]: "opp_xg", tf[1]: "opp_xga"})
    d = d.merge(of, on=["season", "fixture", "opp"], how="left")
    feats += tf + ["opp_xg", "opp_xga"]

    d, cf = congestion(d)
    feats += cf
    d, odf = odds_features(d, odds)
    feats += odf
    return d, feats


# Settings picked by random search on 2024-25 and confirmed on 2025-26 (experiments/exp_tune.py):
# small trees with strong regularisation, since a single match is mostly noise.
TUNED = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200, feature_fraction=0.5, lambda_l2=20)
PARAMS = dict(objective="regression", bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0, **TUNED)
# The squared-error model ranks well but overrates the top picks; the Huber one ranks better
# (big hauls don't drag it around) but underrates them. Their average does best on both.
HUBER = dict(PARAMS, objective="huber", alpha=2.0, num_leaves=7)
SEEDS = 3  # averaging a few differently-seeded copies smooths out random noise in the trees


class Ensemble:
    def __init__(self, df, feats, weight=None):
        self.feats = feats
        data = lambda: lgb.Dataset(df[feats], df["total_points"], weight=weight)
        self.models = [lgb.train(dict(p, seed=s), data(), 600) for s in range(SEEDS) for p in (PARAMS, HUBER)]

    def predict(self, X):
        return np.mean([m.predict(X[self.feats] if isinstance(X, pd.DataFrame) else X) for m in self.models], axis=0)


HAUL = 10  # a "big score": the kind of week that moves you up a mini-league
HAUL_PARAMS = dict(PARAMS, objective="binary")


def train_haul(df, feats):
    """Chance of a big score (10+ points) in a match. Used to value differentials by their upside."""
    return lgb.train(HAUL_PARAMS, lgb.Dataset(df[feats], (df["total_points"] >= HAUL).astype(int)), 400)


def train(df, feats, weight=None):
    return Ensemble(df, feats, weight)


def report(te, label):
    reg = te[te["minutes_r3"] >= 60]
    top = reg.sort_values("pred", ascending=False).groupby("GW").head(20)
    print(f"  {label:28s} MAE {np.abs(te.pred - te.total_points).mean():.3f}  "
          f"starters Spearman {spearmanr(reg.pred, reg.total_points)[0]:.3f}  "
          f"top-20 picks avg {top.total_points.mean():.2f}")


def smooth_calibration(pred, actual, bins=25):
    """Calibration curve: average prediction vs average actual points in equal-sized bins, forced
    to rise, joined with straight lines. Smooth (unlike a step-shaped isotonic fit), so players a
    little apart stay a little apart and the planner can still tell them apart on price."""
    q = pd.qcut(pd.Series(pred).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"p": np.asarray(pred), "a": np.asarray(actual), "q": q.values}).groupby("q").mean()
    x, y = df["p"].values, np.maximum.accumulate(df["a"].values)
    y = y + np.arange(len(y)) * 1e-3  # strictly rising
    return x, y


def calibrate(calib, pred, pos):
    """Map raw predictions to expected points with each position's calibration curve.
    Beyond the last bin the curve continues at the same slope instead of flattening."""
    pred, pos = np.asarray(pred, dtype=float), np.asarray(pos)
    out = pred.copy()
    for k, (x, y) in calib.items():
        sel = pos == k
        p = pred[sel]
        r = np.interp(p, x, y)
        slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-6)
        r = np.where(p > x[-1], y[-1] + (p - x[-1]) * slope, r)
        out[sel] = np.maximum(r, 0)
    return out


def apply_availability(out, elements, events, next_gw):
    """Adjust predictions (index player id, columns gameweek numbers) for availability: FPL's
    chance of playing for next week, 'expected back' dates for injuries, a one-week gap for
    suspensions, and 0 for players who have left or aren't available."""
    out = out.copy()
    els = pd.DataFrame(elements).set_index("id").reindex(out.index)
    deadline = {e["id"]: datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) for e in events}
    chance = els["chance_of_playing_next_round"].fillna(100) / 100
    out[next_gw] *= chance
    for i, e in els.iterrows():
        if e["status"] in ("u", "n"):
            out.loc[i] = 0
        elif e["status"] in ("i", "s", "d") and chance[i] == 0:
            back = expected_back(e["news"])
            for g in out.columns:
                if back is not None and deadline[g] < back:
                    out.at[i, g] = 0
                elif back is None and g == next_gw + 1 and e["status"] == "i":
                    out.at[i, g] *= 0.5  # injured with no return date: unlikely to be back straight away
    return out


def expected_back(news):
    """Date from FPL news like 'Expected back 22 Nov', else None."""
    m = re.search(r"[Ee]xpected back (\d{1,2} \w{3})", news or "")
    if not m:
        return None
    now = datetime.now(timezone.utc)
    d = datetime.strptime(f"{m.group(1)} {now.year}", "%d %b %Y").replace(tzinfo=timezone.utc)
    return d.replace(year=now.year + 1) if (now - d).days > 60 else d  # "2 Jan" written in December


if __name__ == "__main__":
    past = pd.concat([load_past(s) for s in PAST_SEASONS], ignore_index=True)
    alld, feats = add_features(pd.concat([past, load_current()], ignore_index=True))
    hist = alld[~alld["future"] & alld["total_points"].notna()]

    # Walk-forward: each past season predicted by a model trained only on the seasons before it.
    # Those honest predictions also calibrate the final ones: the trees shrink toward the average
    # (a 5.3 prediction actually averaged 6.2), so a calibration curve maps predicted -> actual points.
    # It only stretches the numbers; the order of players never changes.
    import pickle
    if VALIDATE:
        print("Validation (players with 3+ prior games):")
        oof = []
        for i in range(1, len(PAST_SEASONS)):
            tr = hist[hist.season.isin(PAST_SEASONS[:i])]
            te = hist[hist.season == PAST_SEASONS[i]]
            te = te.assign(pred=train(tr, feats).predict(te[feats]))
            oof.append(te[["pred", "total_points", "pos"]])
            if i == len(PAST_SEASONS) - 1:
                from sklearn.metrics import roc_auc_score
                hp = train_haul(tr, feats).predict(te[feats])
                big = (te["total_points"] >= HAUL).astype(int)
                base = te["total_points_r6"].fillna(0)
                print(f"  big-score (10+) model on {PAST_SEASONS[i]}: AUC {roc_auc_score(big, hp):.3f} "
                      f"(last-6 average {roc_auc_score(big, base):.3f}); "
                      f"predicted rate {hp.mean():.3f} vs actual {big.mean():.3f}")
            te = te[te["n_games"] >= 3]
            report(te, f"model on {PAST_SEASONS[i]}")
            report(te.assign(pred=te["total_points_r6"].fillna(0)), f"last-6 avg on {PAST_SEASONS[i]}")
        oof = pd.concat(oof)
        # one fit per position: keepers' points are lumpier (clean sheet or not), so a 4.5 keeper
        # prediction is worth less than a 4.5 midfielder one
        calib = {k: smooth_calibration(g.pred, g.total_points) for k, g in oof.groupby("pos")}
        pickle.dump(calib, open("data/calibration.pkl", "wb"))
        oof["cal"] = calibrate(calib, oof.pred, oof.pos)
        for lo, hi in ((2, 3), (3, 4), (4, 5), (5, 7)):
            b = oof[(oof.cal > lo) & (oof.cal <= hi)]
            print(f"  calibrated {lo}-{hi} pts: predicted {b.cal.mean():.2f}, actual {b.total_points.mean():.2f} "
                  f"(raw model said {b.pred.mean():.2f})")
    calib = pickle.load(open("data/calibration.pkl", "rb"))

    # final model on everything played so far, all seasons weighted equally (weighting this season
    # up was tested by replaying 2024-25 and 2025-26 from GW9 and made predictions worse)
    final = train(hist, feats)
    fut = alld[alld["future"]].copy()
    fut["pred"] = calibrate(calib, final.predict(fut[feats]), fut["pos"])
    # per-fixture stats the planner quotes in its report to explain each pick
    detail = ["element", "GW", "opp", "was_home", "pred", "odds_team_goals", "odds_opp_goals", "odds_clean_sheet",
              "minutes_r6", "starts_r6", "total_points_r6", "expected_goal_involvements_r6", "expected_goals_conceded_r6",
              "euro_prev5", "euro_next5"]
    fut[detail].to_csv("predictions_detail.csv", index=False)
    out = fut.pivot_table(index="element", columns="GW", values="pred", aggfunc="sum").reindex(columns=FUTURE_GWS).fillna(0)

    out = apply_availability(out, boot["elements"], boot["events"], next_gw)
    out.columns = [f"gw{c}" for c in out.columns]
    out.to_csv("predictions.csv")

    # chance of a big score per player per gameweek (two matches in a week: either one)
    fut["haul"] = train_haul(hist, feats).predict(fut[feats])
    haul = fut.groupby(["element", "GW"])["haul"].apply(lambda p: 1 - np.prod(1 - p)).unstack()
    haul = haul.reindex(index=out.index, columns=FUTURE_GWS).fillna(0)
    haul = haul.where(out.values > 0, 0)  # unavailable players can't haul
    haul.columns = out.columns
    haul.to_csv("predictions_haul.csv")
    print(f"Saved predictions.csv for GW{FUTURE_GWS[0]}-{FUTURE_GWS[-1]}")
