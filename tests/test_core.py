"""Tests for the rules and maths the plan depends on. Run: .venv/bin/python -m pytest -q"""
import sys

import numpy as np
import pandas as pd
import pytest

sys.argv = sys.argv[:1] + ["6", "--no-validate"]
import model as M  # noqa: E402
import odds  # noqa: E402
from state import free_transfers, selling_price  # noqa: E402


def weeks(*transfers, start=1):
    return [{"event": start + i, "event_transfers": t} for i, t in enumerate(transfers)]


class TestFreeTransfers:
    def test_first_week_gives_one(self):
        assert free_transfers(weeks(0), {}) == 1

    def test_banks_one_per_week_up_to_five(self):
        assert free_transfers(weeks(0, 0, 0), {}) == 3
        assert free_transfers(weeks(*[0] * 10), {}) == 5

    def test_using_transfers(self):
        assert free_transfers(weeks(0, 0, 2), {}) == 1  # had 2, used 2, +1
        assert free_transfers(weeks(0, 0, 1), {}) == 2

    def test_hits_never_go_below_zero(self):
        assert free_transfers(weeks(0, 4), {}) == 1  # had 1, made 4 (-12), +1

    def test_wildcard_keeps_banked_but_no_weekly_bonus(self):
        # banked 4 going into a wildcard week -> still 4 the week after (matches the FPL app)
        w = weeks(0, 1, 1, 0, 0, 0, 15)
        assert free_transfers(w, {7: "wildcard"}) == 4

    def test_late_joiner_starts_fresh(self):
        assert free_transfers(weeks(15, 0, start=5), {}) == 2


class TestSellingPrice:
    def test_half_the_rise_rounded_down(self):
        assert selling_price(50, 51) == 50
        assert selling_price(50, 52) == 51
        assert selling_price(50, 53) == 51

    def test_falls_in_full(self):
        assert selling_price(50, 48) == 48


class TestExpectedBack:
    def test_parses_date(self):
        d = M.expected_back("Hamstring injury - Expected back 22 Nov")
        assert (d.month, d.day) == (11, 22)

    def test_no_date(self):
        assert M.expected_back("Knock - 75% chance of playing") is None
        assert M.expected_back("") is None


class TestOddsConversion:
    @pytest.mark.parametrize("lh,la", [(1.5, 1.1), (2.4, 0.7), (0.9, 1.6)])
    def test_recovers_known_goal_rates(self, lh, la):
        from scipy.stats import poisson
        g = np.arange(11)
        grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        p_home, p_away = np.tril(grid, -1).sum(), np.triu(grid, 1).sum()
        p_over = grid[np.add.outer(g, g) > 2].sum()
        h, a = odds.goal_rates(p_home, p_away, p_over)
        assert h == pytest.approx(lh, abs=0.05) and a == pytest.approx(la, abs=0.05)


class TestRollingFeatures:
    def frame(self):
        return pd.DataFrame({
            "season": ["s"] * 5, "element": [1] * 5,
            "kickoff_time": ["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29"],
            "total_points": [2.0, 10.0, 6.0, np.nan, np.nan],
            "future": [False, False, False, True, True]})

    def test_no_leakage(self):
        d, f = M.rolling_features(self.frame(), ["season", "element"], ["total_points"], (3,))
        # a match's features only use earlier matches
        assert np.isnan(d.loc[0, "total_points_r3"])
        assert d.loc[1, "total_points_r3"] == 2.0
        assert d.loc[2, "total_points_r3"] == 6.0

    def test_future_rows_get_latest_form(self):
        d, f = M.rolling_features(self.frame(), ["season", "element"], ["total_points"], (3,))
        assert d.loc[3, "total_points_r3"] == pytest.approx(6.0)
        assert d.loc[4, "total_points_r3"] == pytest.approx(6.0)  # same for every future week


class TestClubNames:
    def test_fpl_and_bookmaker_names_match(self):
        o = pd.read_csv("data/odds_goals.csv")
        odds_names = set(o["home"])
        for season in M.PAST_SEASONS:
            fpl = set(M.key(pd.read_csv(f"data/gw_{season}.csv")["team"]))
            missing = fpl - odds_names
            assert not missing, f"{season}: no odds name for {missing}"
        current = set(M.key(pd.Series(M.team_name.values())))
        assert not current - odds_names, current - odds_names

    def test_espn_names_cover_every_european_club(self):
        cups = pd.read_csv("data/cups.csv")
        euro = set(cups[cups["league"].str.startswith("uefa")]["team"].replace(M.ESPN))
        fpl = set()
        for season in M.PAST_SEASONS:
            fpl |= set(pd.read_csv(f"data/gw_{season}.csv")["team"])
        fpl |= set(M.team_name.values())
        english_in_europe = {"Arsenal", "Aston Villa", "Brighton", "Chelsea", "Crystal Palace", "Liverpool", "Man City",
                             "Man Utd", "Newcastle", "Nott'm Forest", "Spurs", "West Ham"}
        assert english_in_europe & fpl <= euro | (english_in_europe - fpl)


class TestSettings:
    def test_env_overrides_ids(self, monkeypatch):
        from settings import load_config
        monkeypatch.setenv("FPL_ENTRY_ID", "123")
        monkeypatch.setenv("FPL_LEAGUE_ID", "456")
        cfg = load_config()
        assert cfg["entry_id"] == 123 and cfg["league_id"] == 456

    def test_example_config_has_no_personal_ids(self):
        import json
        ex = json.load(open("config.example.json"))
        assert ex["entry_id"] is None and ex["league_id"] is None
        assert ex["banned_teams"] and "transfer_cost" in ex
