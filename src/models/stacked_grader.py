"""
stacked_grader.py  (v2)
------------------------
Replaces the untuned 50/50 "hybrid" with a LEARNED combiner that is fitted WITHOUT leakage.

  level 0  any set of base graders (adapters with fit(rows)/predict(rows) -> objects with .score)
  level 1  non-negative linear regression, no intercept, on the base graders' score FRACTIONS

Leakage control (the classic stacking mistake is fitting the combiner on predictions the base models
made on their own training rows): the level-1 training set is built from OUT-OF-FOLD predictions,
using leave-one-question-out inside the rows passed to fit().  At predict time each base grader is
re-fitted on ALL training rows.  Cost: (n_questions + 1) fits per base grader - cheap when the
expensive features are cached (see model_adapters.py).

The learned weights are exposed (`weights_`) so the combination is explainable, e.g.
{"tfidf": 0.0, "lexical": 0.3, "nli_bi": 0.7}.
"""
from typing import Callable, Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression


class StackedGrader:
    def __init__(self, factories: Dict[str, Callable[[], object]], group_key: str = "qid"):
        self.factories, self.group_key = factories, group_key
        self.weights_: Dict[str, float] = {}

    @staticmethod
    def _fracs(preds, rows):
        return np.array([p.score / float(r["max_marks"]) for p, r in zip(preds, rows)])

    def fit(self, rows: List[dict]):
        names = list(self.factories)
        groups = sorted({r[self.group_key] for r in rows})
        if len(groups) < 3:
            raise ValueError("need >= 3 groups (questions) to build out-of-fold predictions")
        oof = np.zeros((len(rows), len(names)))
        for g in groups:
            tr = [r for r in rows if r[self.group_key] != g]
            idx = [i for i, r in enumerate(rows) if r[self.group_key] == g]
            te = [rows[i] for i in idx]
            for j, n in enumerate(names):
                oof[idx, j] = self._fracs(self.factories[n]().fit(tr).predict(te), te)
        y = np.array([float(r["human_score"]) / float(r["max_marks"]) for r in rows])
        self.meta = LinearRegression(positive=True, fit_intercept=False).fit(oof, y)
        self.weights_ = {n: round(float(w), 3) for n, w in zip(names, self.meta.coef_)}
        self.bases = {n: self.factories[n]().fit(rows) for n in names}   # final base models on ALL rows
        self.names = names
        return self

    def predict(self, rows: List[dict]):
        F = np.column_stack([self._fracs(self.bases[n].predict(rows), rows) for n in self.names])
        frac = np.clip(self.meta.predict(F), 0, 1)
        return [(round(float(f) * float(r["max_marks"]) * 2) / 2) for f, r in zip(frac, rows)], F
