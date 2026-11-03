"""
graders.py  (v2)
-----------------
A uniform adapter layer so every grader - trivial, lexical, spaCy, NLI, bi-encoder, LLM judge -
is evaluated by the SAME code under the SAME protocols.

    adapter.fit(rows)            # rows = dataset dicts (rubric stored as a JSON string)
    adapter.predict(rows)        # -> List[GradeOut]

GradeOut carries the score plus (optionally) per-point credit and a confidence, so the same
benchmark can compute QWK/MAE, point-level F1, fact-check behaviour and risk-coverage.

Model-backed adapters (spaCy / NLI / bi-encoder / LLM) are imported lazily, so the benchmark
runs on a laptop with only scikit-learn when those packages are absent.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "models"))

from baseline_grader import TFIDFBaselineGrader               # noqa: E402
from lexical_rubric_grader import LexicalRubricGrader         # noqa: E402


@dataclass
class GradeOut:
    score: float
    confidence: Optional[float] = None
    point_credit: Optional[Dict[str, bool]] = None
    point_prob: Optional[Dict[str, float]] = None
    contradicted_points: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _r05(x: float) -> float:
    return round(x * 2) / 2


def _nwords(r) -> int:
    return len((r["student_answer"] or "").split())


class ConstMean:
    name = "const_mean_fraction"

    def fit(self, rows):
        self.f = float(np.mean([float(r["human_score"]) / float(r["max_marks"]) for r in rows]))
        return self

    def predict(self, rows):
        return [GradeOut(_r05(self.f * float(r["max_marks"]))) for r in rows]


class LengthOnly:
    """Linear regression on log(1+word count) ONLY.  Any real model must beat this."""
    name = "length_only"

    def fit(self, rows):
        X = np.log1p([[_nwords(r)] for r in rows])
        y = [float(r["human_score"]) / float(r["max_marks"]) for r in rows]
        self.m = LinearRegression().fit(X, y)
        return self

    def predict(self, rows):
        X = np.log1p([[_nwords(r)] for r in rows])
        f = np.clip(self.m.predict(X), 0, 1)
        return [GradeOut(_r05(float(fr) * float(r["max_marks"]))) for fr, r in zip(f, rows)]


class TfidfBaseline:
    name = "tfidf_baseline (original Week-2)"

    def fit(self, rows):
        self.g = TFIDFBaselineGrader().fit(rows)
        return self

    def predict(self, rows):
        out = []
        for r in rows:
            res = self.g.grade_one(r["question"], r["reference_answer"], None, r["student_answer"], float(r["max_marks"]))
            out.append(GradeOut(res.score, res.confidence))
        return out


class LexicalRubric:
    name = "lexical_rubric (v2, rubric-aware, no embeddings)"

    def fit(self, rows):
        self.g = LexicalRubricGrader().fit(rows)
        return self

    def predict(self, rows):
        out = []
        for r, res in zip(rows, self.g.grade_batch(rows)):
            out.append(GradeOut(res.score,
                                point_credit={p.point_id: p.credited for p in res.points},
                                point_prob={p.point_id: p.prob for p in res.points}))
        return out


REGISTRY = {"const": ConstMean, "length": LengthOnly, "tfidf": TfidfBaseline, "lexical": LexicalRubric}


def register(name, cls):
    REGISTRY[name] = cls


def load_optional_adapters():
    """Registers model-backed adapters whose dependencies are installed (silently skips others)."""
    try:
        import model_adapters  # noqa: F401  (registers spacy/nli/biencoder/judge adapters on import)
    except Exception as e:  # pragma: no cover
        print(f"[graders] model-backed adapters unavailable: {e}")
