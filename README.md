# FPL model

[![tests](https://github.com/viku03/fpl-model/actions/workflows/tests.yml/badge.svg)](https://github.com/viku03/fpl-model/actions/workflows/tests.yml)

Predicts Fantasy Premier League points with a LightGBM model and plans transfers, captains and chips for your actual squad.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # requirements-experiments.txt adds CatBoost / XGBoost for experiments/
cp config.example.json config.json            # then set entry_id and league_id
```

`entry_id` is the number in your FPL points page URL (`fantasy.premierleague.com/entry/<ID>/event/<GW>`), `league_id` the one in your mini-league's URL. Instead of editing the file you can set `FPL_ENTRY_ID` and `FPL_LEAGUE_ID`. `config.json` and `squad.json` are git-ignored.

The first run downloads past seasons (vaastav/Fantasy-Premier-League), odds and fixtures into `data/`; see "Each new season" for the file names.

## Weekly use

```
./run.sh             # pulls your squad (entry_id in config.json) and plans
./run.sh --manual    # uses squad.json as it is
```

The plan prints and is saved to `reports/plan-GW<next gw>-<date>.md`, with the stats behind every transfer, captain and chip (fixture, bookmaker clean sheet / goal odds, xGI, recent points, minutes and starts). Run it once a gameweek has finished and before you make any transfers (FPL hides upcoming transfers until the deadline). Re-running before the deadline is fine too, e.g. after Friday's press conferences, to pick up injury news.

## Files

| File | What it does |
|---|---|
| `refresh_data.sh` | Downloads prices, injury flags, fixtures and every player's match history from the FPL API |
| `fetch_cups.py` | Downloads Champions League, Europa, Conference, FA Cup and League Cup matches from ESPN into `data/cups.csv` (cached; the first run takes ~11 min, weekly runs a few seconds) |
| `odds.py` | Downloads market-average match odds (home/draw/away, over/under 2.5) from football-data.co.uk, a free site that publishes them as CSV for analysis, and converts them to expected goals for each side in `data/odds_goals.csv` |
| `model.py` | Trains on 4 past seasons (2022-23 to 2025-26) + this season so far (finished matches only), calibrates the predictions to real points per position and predicts points per player for the next `horizon` gameweeks into `predictions.csv`. Prints walk-forward validation each run |
| `rivals.py` | Pulls your mini-league rivals' squads and captains, the gap to first and weeks left, and sets how many differentials to allow and how much risk to take (automatic) |
| `state.py` | Builds `squad.json` from your team ID: 15 players with selling prices, bank, free transfers, chips used |
| `squad.json` | Your current squad. Edit by hand if needed (e.g. transfers you've made before the deadline, which FPL hides until it passes) |
| `config.json` | Your rules and planner settings |
| `solver.py` | The optimisation: plans the whole horizon at once (transfers, -4 hits, free transfer banking, captain, bench order, chips) and checks every plan against the rules |
| `planner.py` | Loads your squad and predictions, calls `solver.py`, writes the report |
| `optimise.py` | Old one-off wildcard picker (ignores your current squad) |

## What the model uses

- Each player's last 1, 3 and 6 games and season average for points, minutes, starts, xG, xA, xGI, xGC, bonus, BPS, ICT, defensive contributions, saves, clean sheets, goals and assists
- Last season's per-game numbers (matched by FPL player code), which matter most in the first few weeks
- Team and opponent attack/defence: xG for and against over the last 8 matches
- Bookmaker odds: expected goals for and against and clean sheet chance for each match. Odds usually only exist for the next gameweek, so later matches use the market's own team attack/defence ratings, worked out from the odds of each club's recent matches
- Fixture congestion: rest days, days to the next match, European or cup games in the 5 days before/after, European games in the last 30 days
- Home and away form separately (last 5 home / last 5 away games)
- Home/away, position, price
- Availability: FPL chance of playing for next week, "Expected back" dates for injuries, players who have left get 0

It averages six LightGBM models: a squared-error and a Huber model (less thrown by rare big hauls), each trained with 3 different random seeds. Settings were picked by random search on 2024-25 and confirmed on 2025-26: small, heavily regularised trees, since one match is mostly noise.

The raw trees shrink towards the average (a raw 4.8 actually averaged 5.3), so the final predictions go through a smooth **calibration curve per position**, learned from the walk-forward predictions. It only rescales the numbers; the order of players is unchanged.

## Mini-league differentials

Your league position only moves on players you own and rivals don't. Each run, `rivals.py` reads your league (`league_id` in config.json) and the planner:

- counts expected points the same whoever owns a player (on average that's what moves you), but
- gives a starter a bonus of `risk x 10 x P(10+ points) x (1 - 2 x share of rivals owning him)`: positive for players rivals don't have, negative for ones most of them do (the same for captaincy),
- allows at most `max_differentials` starters owned by 0-1 rivals.

Both numbers are automatic from the gap to first and the weeks left: about 3-4 differentials when ~100 behind with 27 weeks left, more if you're still far behind late on, fewer when close, and protective (favouring players rivals own) when you're leading. Override them under `differentials` in config.json. The report's "Mini-league view" lists your differentials, their big-score chance, and popular players you don't own.

The big-score chance comes from a second LightGBM model trained on whether a player scored 10+ (AUC 0.856 vs 0.815 for recent form on 2025-26, predicted rate 1.4% vs 1.7% actual).

## Rules it knows

- 15-man squad, 2/5/5/3, max 3 per club, valid formation, selling price = purchase + half the rise.
- Free transfers: +1 per week, max 5, kept through a wildcard or free hit. Extra transfers cost 4 points.
- Chips come from the FPL API: one wildcard, free hit, bench boost and triple captain for GW1-19 and a fresh set for GW20-38. Unused first-half chips are lost after GW19.
- `banned_teams` (Liverpool) and `min_players_from_team` (1 Man Utd) apply every week, including free hit squads.

## Settings in config.json

- `horizon`: gameweeks to plan ahead (6 is a good balance; predictions get weaker further out).
- `discount`: later weeks count a bit less because predictions are less certain.
- `chip_reserve_points`: how many extra points a chip must earn before the planner uses it now instead of saving it. This only applies while the chip's window lasts beyond the horizon, so near GW19 it drops to 0 and the planner uses up any chips left.
- `value_of_saved_free_transfer`: credit for ending the horizon with transfers banked.

## Each new season

Add the finished season to `PAST_SEASONS` and set `CURRENT_SEASON` in `model.py`, then download its two files:
```
curl -o data/gw_<season>.csv https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/<season>/gws/merged_gw.csv
curl -o data/raw_<season>.csv https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/<season>/players_raw.csv
```

## Tests and experiments

Tests run on every push (GitHub Actions); the 8 that need the downloaded `data/` are skipped there and run locally with `test.py`. `tests/test_core.py` covers the rules and maths, `tests/test_solver.py` the optimiser on a made-up league, and `tests/test_regressions.py` has one test per bug found so far. Every new bug fix gets a test there.

```
.venv/bin/python test.py            # every test (~20 s), including a full planner run
.venv/bin/python test.py --quick    # skip the full planner run
PYTHONPATH=. .venv/bin/python -m experiments.exp_ideas   # re-run any experiment (features cached in data/features.pkl)
```

`experiments/backtest.py` replays a whole past season week by week with the planner making every decision (predictions only use what was known before each deadline) and scores it on what actually happened. That's how the planner settings in config.json were chosen:

```
PYTHONPATH=. .venv/bin/python -m experiments.backtest preds 2025-26 2024-25   # once, ~5 min
PYTHONPATH=. .venv/bin/python -m experiments.backtest run 2025-26,2024-25     # every setting in CONFIGS, ~15 min
```

`experiments/harness.py` builds every feature once and scores an idea on 2024-25 and 2025-26 with 5 measures, including the actual points of the best XI the model would have picked each week. Gains under ~1.5 on that measure are noise (the same model re-run varies that much).

## Tried and left out (didn't clearly beat the current model in walk-forward tests)

- Separate models per position (as in OpenFPL): worse on every measure, each position has too little data
- CatBoost (worse), XGBoost (equal), random forest and LightGBM + random forest blends (better at the very top on one measure, worse prediction error)
- Weighting hauls (5+) more in training
- 10-game windows, per-90 rates, player expected goals from odds x xG share: no clear gain
- Penalty / corner / free-kick taker flags (prompted by top managers owning Groß at 98%): no gain, set-piece value already shows up in xG and xA
- Counting this season's games more than past seasons: worse when replaying 2024-25 and 2025-26 from GW9

- Separate minutes model (P(0 / 1-59 / 60+ minutes) x points for each): predicts who plays very well but the final points were no better and training took twice as long
- Player's share of team xG as a feature
- Weighting recent seasons more
