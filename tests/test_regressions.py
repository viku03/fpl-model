"""One test per bug found so far, so none of them can come back.
Each test names the bug it guards against."""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.argv = sys.argv[:1] + ["6", "--no-validate"]
import model as M  # noqa: E402
import team_news  # noqa: E402
from solver import solve  # noqa: E402
from state import purchase_from_history, real_squad_gw  # noqa: E402
from tests.test_solver import CFG, league, squad  # noqa: E402

FIX = {f["id"]: f for f in json.load(open("data/fix.json"))}


@pytest.fixture(scope="module")
def current():
    return M.load_current()


# --- data loading ---------------------------------------------------------------------------

def test_live_matches_not_used_as_finished(current):
    """Bug: mid-gameweek, live/unplayed matches had 0 minutes and halved every prediction."""
    real = current[current["future"] != True]  # noqa: E712
    assert all(FIX[f]["finished"] or FIX[f]["finished_provisional"] for f in real["fixture"].unique())


def test_summer_signings_credited_to_club_they_played_for(current):
    """Bug: a player's early-season games were counted for his new club after a transfer."""
    real = current[current["future"] != True].sample(500, random_state=0)  # noqa: E712
    for r in real.itertuples():
        f = FIX[r.fixture]
        assert r.team == M.team_name[f["team_h"] if r.was_home else f["team_a"]]


def test_assistant_manager_rows_dropped():
    """Bug: 2024-25 'AM' (assistant manager chip) rows were trained on as midfielders."""
    d = M.load_past("2024-25")
    assert set(d["pos"].unique()) <= {1, 2, 3, 4}
    assert "AM" not in set(d["position"])


def test_missing_xg_is_unknown_not_zero():
    """Bug: 2022-23 stored xG as 0 before GW16, which reads as 'no chances'."""
    d = M.load_past("2022-23")
    assert d[d.GW < 16]["expected_goals"].isna().all()
    assert d[d.GW >= 16]["expected_goals"].notna().any()


def test_prior_season_per_90_not_per_start():
    """Bug: last season's numbers were divided by starts, so a sub with 1 start looked elite.
    Also: 2021-22 has no starts / xGI columns and crashed."""
    p = M.load_prior("2023-24")  # uses 2022-23 totals
    assert (p["prior_total_points"].dropna() < 20).all()
    p21 = M.load_prior("2022-23")  # uses 2021-22, which lacks some columns
    assert p21["prior_expected_goal_involvements"].isna().all()


# --- features -------------------------------------------------------------------------------

def test_future_weeks_share_latest_form():
    """Bug: weeks 2-6 of the plan saw different form numbers from week 1."""
    d = pd.DataFrame({"season": "s", "element": 1, "total_points": [4.0, 8.0, np.nan, np.nan, np.nan],
                      "kickoff_time": [f"2026-01-0{i}" for i in range(1, 6)],
                      "future": [False, False, True, True, True]})
    out, _ = M.rolling_features(d, ["season", "element"], ["total_points"], (3, 6))
    fut = out[out.future]
    assert fut["total_points_r3"].nunique() == 1 and fut["total_points_r6"].nunique() == 1


def test_missing_odds_estimate_separates_strong_and_weak_teams():
    """Bug: the fallback for matches without odds barely told City from an average side."""
    rows = []
    for h, a, hg, ag in [("Strong", "Mid", 2.4, 0.8), ("Mid", "Strong", 0.9, 2.1), ("Weak", "Mid", 1.0, 1.6),
                         ("Mid", "Weak", 1.8, 0.9), ("Strong", "Weak", 2.8, 0.6), ("Weak", "Strong", 0.7, 2.5)] * 3:
        rows.append(dict(season="2026-27", home=h, away=a, home_goals=hg, away_goals=ag))
    predict = M.market_ratings(pd.DataFrame(rows), "2026-27")
    strong_away, weak_home = predict(["Strong"], ["Weak"], [0])[0], predict(["Weak"], ["Strong"], [1])[0]
    assert strong_away > 1.8 and weak_home < 1.0


# --- calibration and availability -----------------------------------------------------------

def test_calibration_rescales_without_ties():
    """Bugs: predictions were ~15-20% too low; the first fix (isotonic) gave many players the exact
    same number so the planner couldn't tell a £7m player from a £5m one."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 5, 20000)
    actual = raw * 1.2 + rng.normal(0, 1, raw.size)
    calib = {3: M.smooth_calibration(raw, actual)}
    test = np.linspace(1, 4.5, 50)
    out = M.calibrate(calib, test, np.full(50, 3))
    assert np.all(np.diff(out) > 0)  # strictly rising: no ties
    assert out[-1] == pytest.approx(4.5 * 1.2, abs=0.35)


def boot_like():
    now = datetime.now(timezone.utc)
    events = [{"id": g, "deadline_time": (now + timedelta(days=7 * (g - 10))).isoformat()} for g in range(10, 16)]
    return events


def test_suspension_only_misses_next_week():
    """Bug: suspended players were set to 0 for the whole 6-week plan."""
    out = pd.DataFrame({g: [5.0] for g in range(10, 16)}, index=[1])
    el = [{"id": 1, "status": "s", "chance_of_playing_next_round": 0, "news": "Suspended until 15 Nov"}]
    res = M.apply_availability(out, el, boot_like(), 10)
    assert res.at[1, 10] == 0 and res.at[1, 12] == 5.0


def test_injury_with_return_date():
    """'Expected back' dates zero the weeks before the return, not after."""
    back = (datetime.now(timezone.utc) + timedelta(days=15)).strftime("%d %b")
    out = pd.DataFrame({g: [5.0] for g in range(10, 16)}, index=[1])
    el = [{"id": 1, "status": "i", "chance_of_playing_next_round": 0, "news": f"Hamstring - Expected back {back}"}]
    res = M.apply_availability(out, el, boot_like(), 10)
    assert res.at[1, 10] == 0 and res.at[1, 11] == 0 and res.at[1, 14] == 5.0


def test_player_who_left_is_zero():
    out = pd.DataFrame({g: [5.0] for g in range(10, 16)}, index=[1])
    el = [{"id": 1, "status": "u", "chance_of_playing_next_round": None, "news": "Joined X on loan"}]
    assert (M.apply_availability(out, el, boot_like(), 10).loc[1] == 0).all()


# --- your squad -----------------------------------------------------------------------------

def test_free_hit_week_uses_previous_squad():
    """Bug: the importer read your temporary free hit team as your real squad."""
    assert real_squad_gw(12, "freehit") == 11
    assert real_squad_gw(12, "bboost") == 12 and real_squad_gw(12, None) == 12


def test_purchase_price_from_week_you_joined():
    """Bug: players held since joining used their GW1 price, not the price when you joined."""
    hist = [{"round": 1, "value": 60}, {"round": 5, "value": 63}, {"round": 6, "value": 64}]
    assert purchase_from_history(hist, 5, 64) == 63
    assert purchase_from_history(hist, 1, 64) == 60


# --- planner --------------------------------------------------------------------------------

def test_bench_boost_still_starts_best_xi():
    """Bug: in a bench boost week the planner listed random players as starters (Haaland benched)."""
    players, pred = league()
    owned = {i: 5.0 for i in squad(players)}
    star = next(i for i in owned if players.pos[i] == 4)
    pred.loc[star] = 9.0
    plan = solve(pred, players, owned, 0.0, 1, {"bboost": [(1, 19)]}, CFG, active=("bboost",))
    assert plan.weeks[0].chip == "bboost" and star in plan.weeks[0].xi


def test_team_news_times():
    """Expected news times follow press-conference days and are compared against the deadline."""
    sun = pd.Timestamp("2026-11-08 14:00", tz="Europe/London")
    sat = pd.Timestamp("2026-11-07 15:00", tz="Europe/London")
    assert team_news.expected_time(sun).day_name() == "Friday"
    assert team_news.expected_time(sat).day_name() == "Friday"
    deadline = pd.Timestamp("2026-11-06 18:30", tz="Europe/Dublin")
    early = pd.Timestamp("2026-11-05 15:00", tz="Europe/Dublin")
    sym, _ = team_news.news_state({"updated": False, "kickoff": sun}, deadline, early)
    assert sym == "⏳"
    late = pd.Timestamp("2026-11-06 18:00", tz="Europe/Dublin")
    sym, txt = team_news.news_state({"updated": False, "kickoff": sun}, deadline, late)
    assert sym == "✅" and "no changes" in txt


@pytest.mark.slow
def test_planner_runs_end_to_end():
    """Bug: a variable in the report loop shadowed a function and crashed the report.
    Runs the real planner on the current data and checks the report comes out."""
    r = subprocess.run([sys.executable, "planner.py"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DEADLINE" in r.stdout and "Team news check" in r.stdout
    # Bug: "free transfers 4" read as 4 transfers being made when only 3 were; the line must say
    # how many are made and how many free transfers are used vs saved
    import re
    for made, used, avail, saved in re.findall(
            r"\*\*(\d+) transfer\(s\)\*\* · free transfers: using (\d+) of (\d+), saving (\d+)", r.stdout):
        assert int(used) + int(saved) == int(avail) and int(used) <= int(made)
