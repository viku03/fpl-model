"""Build docs/backtest.png from the season replays (experiments/backtest.py "current" and
"form baseline" runs). Run from the repo root: .venv/bin/python docs/make_charts.py"""
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL, BASELINE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated, CVD-safe)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
for ax, season in zip(axes, ("2024-25", "2025-26"), strict=True):
    model = pd.read_csv(f"data/backtest_{season}_current.csv")
    base = pd.read_csv(f"data/backtest_{season}_form_baseline.csv")
    ax.set_facecolor(SURFACE)
    lines = ((model, MODEL, "Model + planner", "-"), (base, BASELINE, "Last-6 form + planner", "--"))
    for log, color, label, style in lines:
        cum = log["pts"].cumsum()
        ax.plot(log["gw"], cum, color=color, lw=2, ls=style, label=label, solid_capstyle="round")
        ax.annotate(f"{cum.iloc[-1]:,.0f}", (log["gw"].iloc[-1], cum.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=10, fontweight="bold")
    gap = model["pts"].sum() - base["pts"].sum()
    ax.set_title(f"{season}: +{gap:,.0f} points", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Gameweek", color=MUTED)
    ax.set_xlim(1, 41)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Cumulative points", color=MUTED)
axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
fig.suptitle("Season replay: the same planner with model predictions vs recent-form predictions",
             x=0.01, ha="left", color=INK, fontsize=13)
fig.text(0.01, -0.02, "Each week uses only information available before that deadline; scored on actual FPL points "
         "(captain, autosubs, chips, -4 hits).", color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig("docs/backtest.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("Saved docs/backtest.png")
