"""Solver tests on a small made-up league."""
import pandas as pd

from solver import check_week, solve

CFG = {"banned_teams": ["LIV"], "min_players_from_team": {"MUN": 1}, "discount": 1.0, "hit_cost": 4,
       "value_of_saved_free_transfer": 0.0, "chip_reserve_points": {"wildcard": 99, "freehit": 99, "bboost": 99, "3xc": 99},
       "players_per_position": 30, "solver_seconds": 30, "transfer_cost": 0.0}
CLUBS = ["ARS", "CHE", "MUN", "LIV", "TOT", "NEW", "AVL", "BHA"]


def league():
    rows, pid = [], 0
    for pos, n in ((1, 6), (2, 14), (3, 14), (4, 8)):
        for k in range(n):
            pid += 1
            rows.append(dict(id=pid, pos=pos, club=CLUBS[pid % len(CLUBS)], price=5.0))
    players = pd.DataFrame(rows).set_index("id")
    pred = pd.DataFrame({"gw1": 3.0, "gw2": 3.0}, index=players.index)
    return players, pred


def squad(players):
    out = []
    for pos, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
        cand = players[(players.pos == pos) & (players.club != "LIV")]
        picked = []
        for i, r in cand.iterrows():
            if sum(players.club[j] == r.club for j in out + picked) < 3:
                picked.append(i)
            if len(picked) == n:
                break
        out += picked
    assert any(players.club[i] == "MUN" for i in out)
    return out


def test_plan_follows_rules():
    players, pred = league()
    owned = {i: 5.0 for i in squad(players)}
    plan = solve(pred, players, owned, 0.0, 1, {}, CFG)
    assert plan.status == "Optimal"
    for w in plan.weeks:
        assert check_week(w, players, CFG) == []


def tiny_upgrade_setup(gain):
    players, pred = league()
    mine = squad(players)
    owned = {i: 5.0 for i in mine}
    out_id = next(i for i in mine if players.pos[i] == 3 and players.club[i] != "MUN")
    in_id = next(i for i in players.index if players.pos[i] == 3 and i not in owned
                 and players.club[i] == players.club[out_id])
    pred.loc[in_id] = 3.0 + gain / 2  # spread over the 2 weeks
    return players, pred, owned, in_id


def test_transfer_cost_skips_tiny_gains():
    players, pred, owned, in_id = tiny_upgrade_setup(0.3)
    plan = solve(pred, players, owned, 0.0, 1, {}, dict(CFG, transfer_cost=1.0))
    assert plan.weeks[0].ins == []


def test_clear_upgrade_is_taken():
    players, pred, owned, in_id = tiny_upgrade_setup(3.0)
    plan = solve(pred, players, owned, 0.0, 1, {}, dict(CFG, transfer_cost=1.0))
    assert plan.weeks[0].ins == [in_id]


def test_never_picks_banned_club():
    players, pred = league()
    pred.loc[players.club == "LIV"] = 20.0  # even if Liverpool players look amazing
    plan = solve(pred, players, {}, 100.0, 1, {}, CFG, free_first=True)
    assert not any(players.club[i] == "LIV" for i in plan.weeks[0].squad)


# --- mini-league differentials --------------------------------------------------------------

from rivals import risk_settings  # noqa: E402


def test_risk_grows_when_far_behind_with_few_weeks_left():
    now = risk_settings(gap=100, weeks_left=27)
    late = risk_settings(gap=100, weeks_left=8)
    close = risk_settings(gap=20, weeks_left=27)
    assert 3 <= now["max_differentials"] <= 4
    assert late["max_differentials"] > now["max_differentials"] and late["risk"] > now["risk"]
    assert close["max_differentials"] < now["max_differentials"]


def test_leading_turns_protective():
    s = risk_settings(gap=-15, weeks_left=10)
    assert s["max_differentials"] == 1 and s["risk"] < 0


def diff_setup(cap, risk=1.0):
    players, pred = league()
    owned = {i: 5.0 for i in squad(players)}
    haul = pred * 0 + 0.05
    # every player nobody else owns except the current squad, which rivals all own
    ownership = {i: 1.0 for i in owned}
    rivals = dict(ownership=ownership, captaincy={}, risk=risk, max_differentials=cap, owned_by_at_most=1 / 7)
    return players, pred, owned, haul, rivals


def test_differential_cap_respected():
    players, pred, owned, haul, rivals = diff_setup(cap=2)
    pred[:] = 3.0
    for i in players.index:
        if i not in owned:
            pred.loc[i] = 3.5  # differentials slightly better everywhere
    plan = solve(pred, players, owned, 0.0, 5, {}, CFG, free_first=True, haul=haul, rivals=rivals)
    for w in plan.weeks:
        assert sum(rivals["ownership"].get(i, 0) <= 1 / 7 for i in w.xi) <= 2


def test_equal_points_prefers_the_differential():
    players, pred, owned, haul, rivals = diff_setup(cap=4)
    out_id = next(i for i in owned if players.pos[i] == 3 and players.club[i] != "MUN")
    in_id = next(i for i in players.index if players.pos[i] == 3 and i not in owned
                 and players.club[i] == players.club[out_id])
    haul.loc[[out_id, in_id]] = 0.3
    plan = solve(pred, players, owned, 0.0, 1, {}, CFG, haul=haul, rivals=rivals)
    assert in_id in plan.weeks[0].squad  # same expected points, but rivals don't have him


def test_chip_never_left_to_expire():
    """A chip whose window ends inside the plan must be used (it's worth nothing afterwards)."""
    players, pred = league()
    owned = {i: 5.0 for i in squad(players)}
    plan = solve(pred, players, owned, 0.0, 1, {"3xc": [(1, 2)]}, CFG)
    assert any(w.chip == "3xc" for w in plan.weeks)
