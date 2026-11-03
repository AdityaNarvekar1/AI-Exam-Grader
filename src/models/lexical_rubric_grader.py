"""
lexical_rubric_grader.py  (v2)
--------------------------------
A RUBRIC-AWARE lexical baseline with no embeddings and no downloads.

Why: the Week 2 baseline compares the answer to the *reference answer only* and ignores the
rubric completely, so "embedding model vs baseline" compared a rubric-aware model with a
rubric-blind one.  To claim a transformer helps, it has to beat the best *lexical*
rubric-aware method, which is this one.

Per rubric point it computes
    recall1  - share of the point's content-word stems found in the answer
    recall2  - share of the point's content-word bigrams (stem pairs) found in the answer
    best_sentence_recall - recall1 restricted to the single best sentence (rewards locality,
                           which penalises keyword stuffing spread over a word salad)
and a logistic regression (fit on point-level coverage labels) turns them into P(covered).
Interface mirrors the other graders: fit(rows) / grade_one(...).
"""

import json
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression

STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
        "those", "of", "in", "on", "at", "to", "for", "and", "or", "but", "with", "as", "by", "it", "its",
        "from", "into", "such", "which", "who", "whom", "their", "there", "than", "then", "so", "if",
        "each", "also", "can", "will", "would", "should", "not", "no", "we", "you", "your", "they",
        "has", "have", "had", "do", "does", "did", "when", "while", "about", "between", "because",
        "meaning", "given", "correct", "example"}
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def stem(w: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def stems(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    return [stem(t) for t in toks if t not in STOP and len(t) > 1]


def _bigrams(seq):
    return {(a, b) for a, b in zip(seq, seq[1:])}


def point_features(point_text: str, answer: str) -> List[float]:
    p = stems(point_text)
    if not p or not answer.strip():
        return [0.0, 0.0, 0.0]
    a = stems(answer)
    aset = set(a)
    r1 = sum(w in aset for w in set(p)) / len(set(p))
    pb = _bigrams(p)
    r2 = len(pb & _bigrams(a)) / len(pb) if pb else 0.0
    best = 0.0
    for s in SENT_RE.split(answer):
        ss = set(stems(s))
        if ss:
            best = max(best, sum(w in ss for w in set(p)) / len(set(p)))
    return [r1, r2, best]


@dataclass
class LexPoint:
    point_id: str
    marks: float
    prob: float
    credited: bool


@dataclass
class LexResult:
    score: float
    max_marks: float
    points: List[LexPoint]


class LexicalRubricGrader:
    def __init__(self, threshold: float = 0.5):
        self.clf = LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000)
        self.threshold = threshold
        self._fitted = False

    def fit(self, rows: List[dict]):
        X, y = [], []
        for r in rows:
            covered = set(json.loads(r["points_covered"]))
            for p in json.loads(r["rubric"]):
                X.append(point_features(p["text"], r["student_answer"]))
                y.append(int(p["id"] in covered))
        self.clf.fit(np.array(X), np.array(y))
        self._fitted = True
        return self

    def grade_one(self, rubric: List[dict], student_answer: str, max_marks: float) -> LexResult:
        assert self._fitted, "call fit() first"
        pts = []
        for p in rubric:
            prob = float(self.clf.predict_proba(np.array([point_features(p["text"], student_answer)]))[0][1])
            pts.append(LexPoint(p["id"], float(p["marks"]), prob, prob >= self.threshold))
        total = sum(x.marks for x in pts if x.credited)
        return LexResult(round(min(total, max_marks) * 2) / 2, max_marks, pts)

    def grade_batch(self, rows: List[dict]) -> List[LexResult]:
        return [self.grade_one(json.loads(r["rubric"]), r["student_answer"], float(r["max_marks"])) for r in rows]
