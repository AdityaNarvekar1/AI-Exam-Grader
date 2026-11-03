"""
baseline_grader.py
-------------------
Group 2 - AI Exam Answer Grader
Week-2 BASELINE model (per Section 6, Requirement 2 of the course handout:
"a relatively simple NLP approach" such as TF-IDF / keyword matching /
cosine similarity). This is intentionally simple - it is the reference
point that Week 4's embedding/LLM-based "improved approach" must beat.

Pipeline
--------
1. Preprocess (lowercase, strip punctuation, remove stopwords) reference
   and student answers.
2. Feature A - TF-IDF cosine similarity between student answer and the
   reference answer (classic bag-of-words semantic overlap).
3. Feature B - Keyword-overlap ratio: fraction of "content words" (non-
   stopword tokens) in the reference answer that also appear in the
   student answer (classic rubric/keyword-matching baseline).
4. A simple Linear Regression combines [A, B] -> predicted fraction of
   max_marks, fitted on the TRAIN split only. This keeps the model a
   "baseline" (no embeddings, no LLM) while giving it a small, transparent,
   trainable calibration step, as required by the dataset train/test split.
5. predicted_score = clip(predicted_fraction, 0, 1) * max_marks, rounded
   to the nearest 0.5 mark.

This module is written so it can later be wrapped directly as the
Week-6 API endpoint (POST /evaluate-answer) with no redesign - see
`grade_one()` for the single-example interface described in the handout's
Section 9/10 I/O philosophy.
"""

import re
import string
import json
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "with", "as", "by", "it", "its", "from", "into",
    "such", "which", "who", "whom", "their", "there", "than", "then",
    "so", "if", "each", "also", "can", "will", "would", "should", "not",
    "no", "we", "you", "your", "i", "they", "he", "she", "his", "her",
    "has", "have", "had", "do", "does", "did", "over", "when", "while",
    "about", "between", "because",
}


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_words(text: str) -> set:
    tokens = preprocess(text).split()
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def keyword_overlap_ratio(reference: str, student: str) -> float:
    ref_words = content_words(reference)
    stu_words = content_words(student)
    if not ref_words:
        return 0.0
    return len(ref_words & stu_words) / len(ref_words)


@dataclass
class BaselineGraderResult:
    qid: str
    student_id: str
    score: float
    max_marks: float
    confidence: float
    tfidf_similarity: float
    keyword_overlap: float
    feedback: str


class TFIDFBaselineGrader:
    """
    Approach A (Week 2 baseline): TF-IDF cosine similarity + keyword overlap,
    calibrated with a linear regressor fit on the labelled train split.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.regressor = LinearRegression()
        self._fitted = False

    def _fit_vectorizer(self, all_texts: List[str]):
        self.vectorizer.fit(all_texts)

    def _tfidf_sim(self, reference: str, student: str) -> float:
        vecs = self.vectorizer.transform([preprocess(reference), preprocess(student)])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)

    def fit(self, train_rows: List[dict]):
        """train_rows: dicts with reference_answer, student_answer, human_score, max_marks."""
        all_texts = [preprocess(r["reference_answer"]) for r in train_rows] + \
                    [preprocess(r["student_answer"]) for r in train_rows]
        self._fit_vectorizer(all_texts)

        X, y = [], []
        for r in train_rows:
            sim = self._tfidf_sim(r["reference_answer"], r["student_answer"])
            kw = keyword_overlap_ratio(r["reference_answer"], r["student_answer"])
            X.append([sim, kw])
            y.append(float(r["human_score"]) / float(r["max_marks"]))
        self.regressor.fit(np.array(X), np.array(y))
        self._fitted = True
        return self

    def grade_one(self, question: str, reference_answer: str, rubric,
                  student_answer: str, max_marks: float,
                  qid: str = "", student_id: str = "") -> BaselineGraderResult:
        if not self._fitted:
            raise RuntimeError("Call .fit(train_rows) before grading.")

        sim = self._tfidf_sim(reference_answer, student_answer)
        kw = keyword_overlap_ratio(reference_answer, student_answer)
        pred_fraction = float(self.regressor.predict(np.array([[sim, kw]]))[0])
        pred_fraction = max(0.0, min(1.0, pred_fraction))
        score = round(pred_fraction * max_marks * 2) / 2

        # crude, transparent confidence: how much the two signals agree
        confidence = round(1.0 - abs(sim - kw), 2)

        if pred_fraction >= 0.85:
            feedback = "Answer closely matches the reference answer in content and vocabulary."
        elif pred_fraction >= 0.55:
            feedback = "Answer covers some key points but is missing important concepts from the reference answer."
        elif pred_fraction >= 0.25:
            feedback = "Answer only partially overlaps with the expected content; most key concepts are missing."
        else:
            feedback = "Answer shows little to no overlap with the reference answer/rubric."

        return BaselineGraderResult(
            qid=qid, student_id=student_id, score=score, max_marks=max_marks,
            confidence=confidence, tfidf_similarity=round(sim, 3),
            keyword_overlap=round(kw, 3), feedback=feedback,
        )

    def grade_batch(self, rows: List[dict]) -> List[BaselineGraderResult]:
        return [
            self.grade_one(
                question=r["question"], reference_answer=r["reference_answer"],
                rubric=r.get("rubric"), student_answer=r["student_answer"],
                max_marks=float(r["max_marks"]), qid=r["qid"],
                student_id=r["student_id"],
            )
            for r in rows
        ]


if __name__ == "__main__":
    # Minimal smoke test
    demo_train = [{
        "reference_answer": "A stack is LIFO. A queue is FIFO.",
        "student_answer": "A stack is LIFO, last in first out.",
        "human_score": 3, "max_marks": 5,
    }]
    g = TFIDFBaselineGrader().fit(demo_train)
    r = g.grade_one("Q", demo_train[0]["reference_answer"], None,
                     demo_train[0]["student_answer"], 5, "Q1", "S1")
    print(r)
