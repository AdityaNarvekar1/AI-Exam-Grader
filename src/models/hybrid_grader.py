"""
hybrid_grader.py
-----------------
Group 2 - AI Exam Answer Grader
The default grading approach of the API (Week 6): combines the TF-IDF
baseline and the per-concept embedding grader.

Rule (fixed in advance - the 50/50 weights were NOT tuned):
  score      = mean(baseline, embedding), rounded to the nearest 0.5 mark
               EXCEPT when the fact-check layer detects a contradicted rubric
               point: then score = min(baseline, embedding), because the
               baseline only measures vocabulary and cannot see factual
               errors, so it must not dilute the check.
  confidence = 1 - |baseline - embedding| / max_marks   (agreement between
               two independent approaches; NOT a calibrated probability.
               Week 6 evaluation: Spearman +0.59 with lower error on
               unseen questions.)
  needs_human_review = confidence < review_threshold (0.7)
                       OR a fact-check contradiction was detected.

Keeping this in one module means the API, the evaluation scripts and the
robustness checks all apply exactly the same rule.
"""

import json
from dataclasses import dataclass
from typing import List

REVIEW_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class HybridResult:
    score: float
    max_marks: float
    confidence: float
    needs_human_review: bool
    baseline_score: float
    embedding_score: float
    any_contradiction: bool
    feedback: str
    point_matches: list


class HybridGrader:
    def __init__(self, baseline, embedding, review_threshold: float = REVIEW_CONFIDENCE_THRESHOLD):
        self.baseline = baseline
        self.embedding = embedding
        self.review_threshold = review_threshold

    def grade_one(self, question: str, reference_answer: str, rubric: List[dict],
                  student_answer: str, max_marks: float) -> HybridResult:
        base = self.baseline.grade_one(question=question, reference_answer=reference_answer,
                                       rubric=rubric, student_answer=student_answer,
                                       max_marks=max_marks)
        emb = self.embedding.grade_one(reference_answer=reference_answer, rubric=rubric,
                                       student_answer=student_answer, max_marks=max_marks)
        any_contradiction = any(m.fact_check == "contradicted" for m in emb.point_matches)
        if any_contradiction:
            score = min(base.score, emb.score)
        else:
            score = round((base.score + emb.score) / 2 * 2) / 2
        score = min(score, max_marks)
        confidence = round(max(0.0, 1 - abs(base.score - emb.score) / max_marks), 2)
        return HybridResult(
            score=score, max_marks=max_marks, confidence=confidence,
            needs_human_review=bool(confidence < self.review_threshold or any_contradiction),
            baseline_score=base.score, embedding_score=emb.score,
            any_contradiction=any_contradiction, feedback=emb.feedback,
            point_matches=emb.point_matches,
        )

    def grade_batch(self, rows: List[dict]) -> List[HybridResult]:
        """rows: dataset rows (rubric stored as a JSON string)."""
        return [self.grade_one(r["question"], r["reference_answer"], json.loads(r["rubric"]),
                               r["student_answer"], float(r["max_marks"])) for r in rows]
