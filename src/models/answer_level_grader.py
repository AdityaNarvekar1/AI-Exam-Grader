"""
answer_level_grader.py  (v2)
------------------------------
For datasets that only have an ANSWER-LEVEL grade (Mohler, SciEntsBank ...): no per-point labels exist,
so the per-point classifier of FeatureGrader cannot be trained.  Instead:

  1. reference answer -> pseudo-rubric (one point per reference sentence)       [load_external.pseudo_rubric]
  2. the same per-point features as FeatureGrader (lexical / bi-encoder / NLI)
  3. aggregate over points (mean and min of every feature)  -> Ridge regression -> score fraction

Length is deliberately NOT a feature (it is the confound this whole v2 effort is about).
Interface: fit(rows) / predict(rows) -> list of objects with .score ; used by run_external_benchmark.py.
"""
import json
from typing import List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from models.feature_grader import FeatureGrader
except ImportError:  # pragma: no cover
    from feature_grader import FeatureGrader


class AnswerLevelGrader:
    def __init__(self, nli=None, bi=None, use_lexical=True, alpha: float = 10.0, cache: Optional[dict] = None):
        self.fg = FeatureGrader(nli=nli, bi=bi, use_lexical=use_lexical, use_factcheck=False, cache=cache)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def _x(self, row) -> np.ndarray:
        rubric = json.loads(row["rubric"])
        vecs = np.array([v for v, _ in self.fg._answer_features(rubric, row["student_answer"] or "")])
        return np.concatenate([vecs.mean(axis=0), vecs.min(axis=0)])

    def fit(self, rows: List[dict]):
        X = np.array([self._x(r) for r in rows])
        y = np.array([float(r["human_score"]) / float(r["max_marks"]) for r in rows])
        self.model.fit(X, y)
        return self

    def predict_fraction(self, rows: List[dict]) -> np.ndarray:
        return np.clip(self.model.predict(np.array([self._x(r) for r in rows])), 0, 1)
