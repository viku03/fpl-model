# How the FPL model works

A plain-English tour of every file, the full flow of one run, and the ideas behind it.

---

## 1. The big picture

The question is: *which 15 players, captain and chips score the most points over the next few weeks, given the squad, money and transfers I actually have?*

That splits into two separate problems:

1. **Prediction**: how many points will each player score in each upcoming match? This is a machine-learning problem, handled by `model.py`.
2. **Decision**: given those predictions and the FPL rules (budget, 3 per club, free transfers, chips), which moves give the most total points? This is an optimisation problem, handled by `planner.py`.

They're kept separate so each can be improved and tested on its own. The model never needs to know FPL rules, and the planner never needs to know any football.

```
                  ┌────────────────── DATA ──────────────────┐
FPL API ──────────► refresh_data.sh ─► data/boot.json  (players, prices, injuries, deadlines, chips)
                  │                  ├► data/fix.json   (all fixtures)
                  │                  └► data/es/*.json  (every player's match-by-match history)
ESPN ─────────────► fetch_cups.py ───► data/cups.csv    (European + cup matches)
football-data.uk ─► odds.py ─────────► data/odds_goals.csv (bookmaker expected goals per match)
vaastav GitHub ───► (one-off) ───────► data/gw_*.csv, data/raw_*.csv (past seasons)
                  └──────────────────────────────────────────┘
                                   │
                                   ▼
                     model.py  ──► predictions.csv          (points per player per GW)
                               └─► predictions_detail.csv   (stats used to explain picks)
                                   │
FPL API (your team) ► state.py ─► squad.json                (your 15, bank, free transfers, chips)
                                   │
                    config.json ───┤                        (your rules: no LIV, ≥1 MUN, …)
                                   ▼
                     planner.py ─► reports/plan-GW<n>-<date>.md
```

`run.sh` runs all of these in order.

---

## 2. One run, step by step (`./run.sh`)

| Step | Command | Time | What happens |
|---|---|---|---|
| 1 | `refresh_data.sh` | ~30 s | Downloads the latest FPL data: prices, injury flags, fixtures and every player's match history |
| 2 | `fetch_cups.py` | ~10 s | Updates the European and cup fixtures (past days are cached, so only recent and upcoming days are fetched) |
| 3 | `odds.py` | ~10 s | Downloads bookmaker odds and turns them into expected goals for each team in each match |
| 4 | `model.py 6` | ~70 s | Builds features, re-runs the validation, trains the final model and predicts the next 6 gameweeks |
| 5 | `state.py <id>` | ~2 s | Pulls your squad, selling prices, bank, free transfers and chips used from FPL (skipped with `--manual`) |
| 6 | `planner.py` | ~40 s | Solves for the best transfers, captain, bench and chips over 6 weeks and writes the report |

If step 2 or 3 fails (site down), the run carries on with the last saved file. If step 1 fails it stops, because everything depends on fresh FPL data.

---

## 3. The data and where it comes from

All sources are free and public. Nothing needs a login or an API key.

### FPL API (`fantasy.premierleague.com/api/…`)
The same endpoints the official app uses. No login is needed for these:

| Endpoint | Gives | Used by |
|---|---|---|
| `bootstrap-static/` | Every player (price, position, team, injury `status`, `chance_of_playing_next_round`, `news`), every gameweek (deadlines), chip rules | everything |
| `fixtures/` | Every match: teams, kickoff, finished or not, FPL difficulty | model, planner |
| `element-summary/<id>/` | One player's match-by-match stats this season: minutes, goals, xG, xA, bonus, BPS, defensive contributions, price | model |
| `entry/<id>/history/` | Your points per week, transfers made, chips used | state |
| `entry/<id>/event/<gw>/picks/` | Your 15 players for a gameweek and your bank | state |
| `entry/<id>/transfers/` | Every transfer with the price paid | state (selling prices) |

`refresh_data.sh` fetches the element-summary for all ~670 players, 16 at a time in parallel.

### Past seasons: vaastav/Fantasy-Premier-League (GitHub)
A well-known open dataset that archives the FPL API every week. It's used for:
- `gw_<season>.csv`: one row per player per match for 2022-23 to 2025-26. This is the training data. (2022-23 only has xG from GW16; earlier games are treated as unknown, not 0.)
- `raw_<season>.csv`: season totals per player, used for "last season" features.

These are downloaded once. They don't change.

### ESPN (`site.api.espn.com`)
ESPN's public scoreboard feed, which their website uses. It only accepts one day per request, so `fetch_cups.py` loops over every day since July 2023 for 5 competitions (Champions League, Europa League, Conference League, FA Cup, League Cup). Days older than a week are cached in `data/cups/` and never fetched again.

### football-data.co.uk
A free site that publishes bookmaker odds as CSV files specifically for people to analyse. We use the **market average** odds (the average across many bookmakers, which is more reliable than any single one) for home/draw/away and over/under 2.5 goals.

---

## 4. File by file

### `refresh_data.sh`
A shell script that downloads `bootstrap-static` and `fixtures`, then every player's `element-summary`, in parallel with `xargs -P 16`. It overwrites last week's files.

### `fetch_cups.py`
- `fetch(league, day)`: downloads one day's matches for one competition, with 3 retries. It skips the download if the day is more than a week old and already cached.
- The rest loops over all (competition, day) pairs with 16 threads, then flattens the results into `data/cups.csv` with columns `date, league, team, status`.

**Why:** a European game midweek means tired legs and rotation at the weekend. The Premier League data alone can't see it.

### `odds.py`
- `goal_rates(p_home, p_away, p_over)`: finds the home and away goal rates (λ) that reproduce the bookmakers' probabilities. It assumes each team's goals follow a **Poisson distribution**, the standard model for football scores. Then it searches (with `scipy.optimize.minimize`) for the two λ values whose implied home-win, away-win and over-2.5 chances best match the odds.
- `main()`: downloads each season's CSV and the upcoming fixtures, removes the bookmaker margin (odds add up to about 105%, so the probabilities are scaled to 100%), runs `goal_rates` for every match and saves `data/odds_goals.csv`.

**Why:** bookmakers combine team news, injuries, lineups and their own models. Their odds are one of the best predictors available. From the goal rates we get each team's **expected goals** and **clean sheet chance** (`e^(−opponent's λ)`).

Example: Man City vs Brighton came out at 2.19 – 1.26.

### `model.py`: the prediction model
This is the biggest file. Main parts, in the order they run:

**Loading**
- `load_past(season)`: reads a past season. It drops "AM" rows (the 2024-25 assistant manager chip, which aren't players), works out each row's opponent from the fixture, and attaches the player's permanent `code` so they can be matched across seasons.
- `load_current()`: builds the same table for this season from `data/es/*.json`. Two important details:
  - It only uses **finished** matches. Live matches would show partial minutes and look like the player was benched.
  - It uses the club the player played for **in that match**, not their current club, so summer signings are handled correctly.

  It also creates **future rows**: one per player per upcoming fixture, with the stats left blank. Those are what the model predicts.
- `load_prior(season)`: last season's per-game numbers (points, minutes, starts, xGI, BPS) for each player.

**Features** (what the model looks at for each player in each match)
- `rolling_features(...)`: for 15 stats (points, minutes, starts, xG, xA, xGI, xGC, BPS, bonus, ICT, defensive contributions, saves, clean sheets, goals, assists), the average over the **last 1, 3 and 6 games** and the **season so far**. It only ever uses games *before* the one being predicted. Using the match's own result would be "leakage": the model would look great in testing and be useless for real predictions. Future rows all get the stats as of the last real game.
- Team strength: each team's xG for and against over its last 8 matches, for both the player's team and the opponent.
- `congestion(d)`: for every Premier League match, the days since the team's last game (any competition), days to its next one, and European or cup games in the 5 days before or after.
- `odds_features(d)`: bookmaker expected goals for and against and clean sheet chance. Future matches usually don't have odds yet, so `market_ratings()` works out an attack and defence rating for every club from the odds of its recent matches (log expected goals = home advantage + attack − opponent's defence, with recent matches counting more) and predicts the missing ones from those ratings.
- Plus position, price, home/away and number of games played so far.

That's around 90 features in total.

**Training**
- `Ensemble`: trains **six LightGBM models** (squared error + Huber, each with 3 seeds) with tuned settings and averages them (explained in section 5).
- `smooth_calibration` / `calibrate`: the trees shrink towards the average, so walk-forward predictions are compared with what actually happened, per position, in 25 bins. The resulting curve rescales new predictions (a raw 4.8 → 5.3). Keepers get their own curve because their points are lumpier.
- `report(te, label)`: prints validation numbers: prediction error (MAE), ranking accuracy among regular starters (Spearman), and the average points of each week's top-20 predicted picks.
- **Walk-forward validation** (in `__main__`): predicts 2024-25 using a model trained only on 2023-24, then predicts 2025-26 using 2023-24 and 2024-25. This mimics real use, where you only ever know the past. Every change to the model is kept only if it improves these numbers.
- The final model is trained on everything, all seasons weighted equally (weighting this season up was tested and made predictions worse).

**Availability adjustments** (after predicting)
- Next week's prediction is multiplied by FPL's `chance_of_playing_next_round`.
- `expected_back(news)` reads "Expected back 22 Nov" from the news text and sets the player to 0 until then.
- Players who have left the league (`u`) or aren't available (`n`) get 0 everywhere.

**Output**
- `predictions.csv`: one row per player, one column per gameweek. Double gameweeks are summed and blank gameweeks are 0.
- `predictions_detail.csv`: the stats for each fixture, which the report quotes.

### `state.py`: your current squad
- `main(entry_id)` fetches your history, picks and transfers, then:
  - **Free transfers**: replays the season. You start with 1 in GW2, get +1 each week (max 5), use them on transfers, and keep them when you play a wildcard or free hit.
  - **Squad**: your last gameweek's picks. If that week was a free hit, it uses the week before, because free hit squads revert.
  - **Selling prices**: FPL's rule is purchase price + half of any rise, rounded down. A fall counts in full. Purchase prices come from your transfer history (ignoring free-hit buys), or the price in the week you joined for players you've had since the start.
  - **Chips used**, from your history.
- Saves `squad.json`, which you can also edit by hand (`./run.sh --manual`).

### `config.json`: your settings
| Setting | Meaning |
|---|---|
| `entry_id` | Your FPL team ID |
| `banned_teams` | `["LIV"]`: never picked |
| `min_players_from_team` | `{"MUN": 1}`: always at least one |
| `horizon` | Gameweeks to plan ahead (6) |
| `discount` | Each week further out counts 10% less, because predictions get less certain |
| `hit_cost` | 4 points per extra transfer |
| `value_of_saved_free_transfer` | Credit for ending the plan with transfers banked, so it doesn't burn them for tiny gains |
| `chip_reserve_points` | Extra points a chip must earn to be used now rather than saved. This only applies while the chip's window lasts beyond the plan, so it drops to 0 as GW19 approaches |
| `players_per_position` | Candidate pool size (the top 25 per position plus your squad), which keeps the solver fast |

### `planner.py`: the decision maker
This is a **mixed-integer linear program** (MILP), solved with the free CBC solver through `pulp`. The idea: describe every decision as a yes/no variable, write the FPL rules as equations those variables must satisfy, and ask the solver for the combination with the most points. Unlike trying options one by one, it's guaranteed to find the best plan under the rules.

**Decision variables** (for every candidate player and every gameweek in the plan):
- `s`: in your squad · `x`: in the starting XI · `c`: captain · `t3`: triple captain · `bb`: bench player counted by bench boost · `f`: in the free-hit squad
- `tin` / `tout`: bought / sold
- `chip[k, g]`: chip k played in gameweek g
- `ft`, `used_ft`, `hits`, `bank`: free transfers available and used, -4 hits taken, money in the bank

**Rules (constraints):**
- Squad continuity: this week's squad = last week's + bought − sold
- Money: bank = last bank + selling prices − buying prices, and never below 0
- 15 players, 2/5/5/3, max 3 per club, no Liverpool, at least 1 Man Utd (also applies to free-hit squads)
- Valid XI: 11 players, 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD; one captain who must be starting
- Transfers: number bought ≤ free transfers used + hits, unless a wildcard is played. A free hit means no changes to the real squad
- Free transfers next week ≤ this week − used + 1, max 5
- Chips: only inside their FPL window (GW1–19 or GW20–38), each at most once per window, at most one chip per week. Already-used chips are blocked

**Objective (what it maximises):**
```
Σ over weeks of discount^n × (XI points + captain points + triple-captain extra + bench-boost bench points + small bench weight)
 − 4 × hits − chip reserve cost + value of banked free transfers
```

**Report:** `why(i, g)` builds the stats line for each player (fixture, clean sheet chance or xGI, team expected goals, recent points, minutes and starts, European flag). The report has the deadline in Dublin time at the top, then each week's chip, transfers with their stats and points gained, team table, bench order and captain reasoning.

### `optimise.py`
The original one-off wildcard picker. It ignores your current squad. It's kept for reference, and the planner replaces it.

---

## 5. Why LightGBM?

### What it is
**LightGBM** is a *gradient-boosted decision tree* library, made by Microsoft.

- A **decision tree** is a flowchart of questions: "Started last game? → Yes → Opponent concedes > 1.5 xG? → Yes → predict 5.1". One tree on its own is crude.
- **Boosting** builds hundreds of small trees one after another. Each new tree is trained to fix the errors the previous trees still make. Tree 1 learns "regular starters score more", tree 50 learns "...unless they're a defender against a strong attack", and so on. The final prediction is the sum of all trees.
- "Light" refers to clever tricks (histogram binning, growing the most useful branch first) that make it very fast. Training on ~90,000 rows takes seconds on a laptop.

### Why it suits this problem
- **Tabular data:** this is a table of numbers (form, xG, odds, rest days). On data like this, boosted trees (LightGBM, XGBoost, CatBoost) consistently beat neural networks, both in research and in Kaggle competitions. Neural nets do best on images, text and audio.
- **Interactions come for free:** "clean sheet odds matter for defenders but not forwards" or "rest days only matter for teams in Europe". Trees learn these on their own, while a linear model would need each one written by hand.
- **Missing values:** a new signing has no "last season" stats, and a promoted team's players have no Premier League history. LightGBM handles blanks natively.
- **Small data:** about 3 seasons × 380 matches × ~30 players. That's plenty for trees, and small for deep learning.
- **Fast and cheap:** no GPU, about a minute for everything.

XGBoost or CatBoost would give practically the same results. The choice between them matters much less than the features and the validation.

### The two-model average
- Model 1 uses **squared error** (the usual choice). It predicts the average well, but rare 15-point hauls pull it around, so it **overrates** its top picks (about +0.5 points).
- Model 2 uses **Huber loss**, which treats big errors more gently, so hauls don't drag it around. It **ranks** players better but **underrates** the top ones (about −0.7).
- Their **average** ranked best on both test seasons, and its predictions for top players were within about 0.15 points of what actually happened.

### How good is it?
In walk-forward tests:

| | 2024-25 | 2025-26 |
|---|---|---|
| Model's weekly top-20 picks, actual avg points | 4.76 | 4.45 |
| "Last 6 games average" approach | 4.22 | 3.86 |

So it picks players who score about 0.5–0.6 points more per player per week than a sensible form-based approach. FPL is very noisy, though: even the best model can't predict a single game well. The ranking accuracy among starters is about 0.27–0.35, which is normal for FPL, and the edge comes from making slightly better decisions every week, all season.

---

## 6. Tests

`tests/test_core.py` (run `.venv/bin/python -m pytest -q tests`) checks free-transfer counting (including the wildcard rule and late joiners), selling prices, the odds → goals conversion, that rolling features never use the match being predicted, injury return dates, and that club names match between FPL, the bookmakers and ESPN in every season. The planner also checks its own plan against every rule on each run and stops if anything is off.

## 7. Things that were tried and left out
Each was tested with walk-forward validation and dropped because it didn't clearly help on both seasons:
- A separate minutes model (P(0 / 1–59 / 60+ minutes) × points for each): very good at predicting minutes, but no better on final points, and twice as slow
- Player's share of his team's xG
- Weighting recent seasons more

---

## 8. Bugs found and fixed (for reference)
- Live matches read as finished, which halved predictions mid-gameweek
- 2024-25 assistant manager rows trained on as midfielders
- Summer signings' early games credited to their new club
- Weeks 2–6 of the plan saw slightly different form numbers than week 1
- Suspended players zeroed for the whole 6-week plan
- Free-hit squad imported as your real squad
- Wrong purchase price for players held since you joined
- Bench boost weeks showing random "starters"
- Predictions ~15-20% too low (fixed with calibration)
- This season's games weighted 3x without testing (made predictions worse)
- Last season's numbers divided by starts, so subs looked like superstars (now per 90)
- ESPN throttling left 1,040 days of European fixtures missing
- Free transfers wrong in wildcard weeks and for late joiners
- Missing odds estimated with a model that barely told City from an average side
