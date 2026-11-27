"""Download bookmaker match odds from football-data.co.uk (free CSVs published for analysis)
and turn them into expected goals for each side, saved to data/odds_goals.csv.

Each match's market-average home/draw/away and over/under 2.5 odds are de-margined into
probabilities, then we find the home and away goal rates (independent Poisson) that best
reproduce them.
"""
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

SEASONS = {"2022-23": "2223", "2023-24": "2324", "2024-25": "2425", "2025-26": "2526", "2026-27": "2627"}
CURRENT = "2026-27"
BASE = "https://www.football-data.co.uk"
# football-data names -> one key used for matching (see key() in model.py)
GOALS = np.arange(11)


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    open(path, "wb").write(urllib.request.urlopen(req, timeout=30).read())


def goal_rates(p_home, p_away, p_over):
    def loss(lam):
        h, a = poisson.pmf(GOALS, lam[0]), poisson.pmf(GOALS, lam[1])
        grid = np.outer(h, a)
        ph, pa = np.tril(grid, -1).sum(), np.triu(grid, 1).sum()
        tot = np.add.outer(GOALS, GOALS)
        po = grid[tot > 2].sum()
        return (ph - p_home) ** 2 + (pa - p_away) ** 2 + (po - p_over) ** 2
    return minimize(loss, [1.4, 1.1], bounds=[(0.05, 5), (0.05, 5)], method="L-BFGS-B").x


def main(refresh_past=False):
    import os
    frames = []
    for season, code in SEASONS.items():
        path = f"data/odds_{code}.csv"
        if season == CURRENT or refresh_past or not os.path.exists(path):
            download(f"{BASE}/mmz4281/{code}/E0.csv", path)
        frames.append(pd.read_csv(path, encoding="utf-8-sig").assign(season=season))
    download(f"{BASE}/fixtures.csv", "data/odds_fixtures.csv")
    up = pd.read_csv("data/odds_fixtures.csv", encoding="utf-8-sig")
    frames.append(up[up["Div"] == "E0"].assign(season=CURRENT))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"])
    d = d.drop_duplicates(["season", "HomeTeam", "AwayTeam"], keep="last")

    inv = 1 / d[["AvgH", "AvgD", "AvgA"]].values
    p1x2 = inv / inv.sum(axis=1, keepdims=True)
    ou = 1 / d[["Avg>2.5", "Avg<2.5"]].values
    p_over = ou[:, 0] / ou.sum(axis=1)
    lam = np.array([goal_rates(h, a, o) for (h, _, a), o in zip(p1x2, p_over, strict=False)])
    out = pd.DataFrame({"season": d["season"].values, "home": d["HomeTeam"].values, "away": d["AwayTeam"].values,
                        "home_goals": lam[:, 0].round(3), "away_goals": lam[:, 1].round(3)})
    out.to_csv("data/odds_goals.csv", index=False)
    print(f"Saved data/odds_goals.csv: {len(out)} matches with odds")


if __name__ == "__main__":
    main()
