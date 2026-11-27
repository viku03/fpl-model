"""LightGBM + random forest blends (random forest ranked the top players best)."""
from experiments.exp_models import evaluate, lgbm, rf, feats, d


class W:
    def __init__(self, a, b, wa):
        self.a, self.b, self.wa = a, b, wa

    def predict(self, te):
        return self.wa * self.a.predict(te) + (1 - self.wa) * self.b.predict(te)


for wa in (0.5, 0.7):
    evaluate(d, feats, lambda tr, f: W(lgbm(tr, f), rf(tr, f), wa), f"{wa:.0%} LightGBM + {1-wa:.0%} RF")
