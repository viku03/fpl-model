"""Penalty / corner / free-kick takers as features (from FPL's taker order: end-of-season snapshot for
past seasons, live for this one). Prompted by top managers owning Groß at 98% while the model didn't."""
import json
import numpy as np
import pandas as pd
from experiments.harness import load, evaluate, M, Model

d, feats, extra = load()
COLS = ["penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"]
frames = []
for s in M.PAST_SEASONS:
    r = pd.read_csv(f"data/raw_{s}.csv")[["id"] + COLS].rename(columns={"id": "element"})
    frames.append(r.assign(season=s))
cur = pd.DataFrame(json.load(open("data/boot.json"))["elements"])[["id"] + COLS].rename(columns={"id": "element"})
frames.append(cur.assign(season=M.CURRENT_SEASON))
sp = pd.concat(frames)
sp["pen_taker"] = (sp.penalties_order == 1).astype(float)
sp["pen_backup"] = (sp.penalties_order == 2).astype(float)
sp["corner_taker"] = (sp.corners_and_indirect_freekicks_order <= 2).astype(float)
sp["fk_taker"] = (sp.direct_freekicks_order <= 2).astype(float)
sp["set_piece_score"] = sp.pen_taker * 2 + sp.pen_backup * 0.5 + sp.corner_taker + sp.fk_taker
new = ["pen_taker", "pen_backup", "corner_taker", "fk_taker", "set_piece_score"]
d = d.merge(sp[["season", "element"] + new], on=["season", "element"], how="left")
d[new] = d[new].fillna(0)
mk = lambda tr, f: Model(tr, f, params=(M.PARAMS, M.HUBER), rounds=(600, 600), seeds=3)
evaluate(d, feats, mk, "current")
evaluate(d, feats + new, mk, "+ set-piece takers")
