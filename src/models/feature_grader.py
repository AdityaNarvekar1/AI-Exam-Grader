"""
feature_grader.py  (v2)  -  learned, evidence-grounded rubric grader
----------------------------------------------------------------------
Replaces "one cosine -> one logistic regression -> hard threshold" with a small, calibrated,
inspectable classifier over several INDEPENDENT signals per rubric point:

    lexical   (lex_r1, lex_r2, lex_best_sent)            always available, no downloads
    bi-enc    (bi_max, bi_top2, bi_full, bi_margin)      optional  - BiEncoderPointScorer
    NLI       (ent_*, contra_*)                          optional  - NLIPointScorer
    rule veto (fact_binding_checker, whole-answer scope) optional  - hard veto, not a feature

Per point:  P(covered) = calibrated logistic regression on standardised features.
Score:      mode="hard"      sum of marks with P >= threshold     (interpretable credit/no-credit)
            mode="expected"  sum of marks * P                     (smooth; lower variance)
Confidence: 1 - marks_at_risk / max_marks,  marks_at_risk = sum_i marks_i * min(P_i, 1-P_i)
            i.e. the expected marks lost to wrong credit decisions - a quantity that has a meaning,
            unlike "how much two correlated models disagree".  Validate it with
            analyze_selective_prediction.py before trusting any threshold.

Everything the grader decided is returned (features, best evidence window, veto reasons), so
every mark can be explained and audited.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from models.fact_binding_checker import apply_fact_check_text
    from models.lexical_rubric_grader import point_features as lex_point_features
    from models.text_utils import possible_injection, split_sentences
    from models import nli_scorer as NLI, biencoder_scorer as BI
except ImportError:  # pragma: no cover  (script-style path)
    from fact_binding_checker import apply_fact_check_text
    from lexical_rubric_grader import point_features as lex_point_features
    from text_utils import possible_injection, split_sentences
    import nli_scorer as NLI
    import biencoder_scorer as BI

LEX_NAMES = ["lex_r1", "lex_r2", "lex_best_sent"]


@dataclass
class PointResult:
    point_id: str
    point_text: str
    marks: float
    prob: float
    credited: bool
    vetoed: bool = False
    veto_reason: str = ""
    best_window: str = ""
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureGradeResult:
    score: float
    max_marks: float
    confidence: float
    marks_at_risk: float
    needs_review: bool
    review_reasons: List[str]
    points: List[PointResult]
    feedback: str

    @property
    def point_credit(self):
        return {p.point_id: p.credited for p in self.points}

    @property
    def point_prob(self):
        return {p.point_id: p.prob for p in self.points}

    @property
    def contradicted_points(self):
        return [p.point_id for p in self.points if p.vetoed]


class FeatureGrader:
    def __init__(self, nli: Optional["NLI.NLIPointScorer"] = None, bi: Optional["BI.BiEncoderPointScorer"] = None,
                 use_lexical: bool = True, use_factcheck: bool = True, nli_veto: Optional[float] = None,
                 mode: str = "hard", threshold: float = 0.5, C: float = 1.0,
                 review_threshold: float = 0.75, cache: Optional[dict] = None):
        assert mode in ("hard", "expected")
        self.nli, self.bi = nli, bi
        self.use_lexical, self.use_factcheck, self.nli_veto = use_lexical, use_factcheck, nli_veto
        self.mode, self.threshold, self.review_threshold = mode, threshold, review_threshold
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=C, max_iter=3000))
        self.cache = cache if cache is not None else {}
        self._fitted = False

    # ------------------------------------------------------------------ features
    @property
    def feature_names(self) -> List[str]:
        n = []
        if self.use_lexical:
            n += LEX_NAMES
        if self.bi is not None:
            n += BI.FEATURE_NAMES
        if self.nli is not None:
            n += NLI.FEATURE_NAMES
        return n

    def _answer_features(self, rubric: List[dict], answer: str):
        """One (vector, aux) per rubric point.  Cached on (answer, point texts)."""
        key = (answer, tuple(p["text"] for p in rubric), tuple(self.feature_names),
               getattr(getattr(self.nli, "backend", None), "model_name", None),
               getattr(getattr(self.bi, "backend", None), "model_name", None))
        if key in self.cache:
            return self.cache[key]
        bi_feats = self.bi.features_for_answer([p["text"] for p in rubric], answer) if self.bi else None
        out = []
        for i, p in enumerate(rubric):
            vec, aux = [], {"best_window": ""}
            if self.use_lexical:
                vec += lex_point_features(p["text"], answer)
            if self.bi:
                vec += [bi_feats[i][k] for k in BI.FEATURE_NAMES]
            if self.nli:
                nf = self.nli.point_features(p["text"], answer, claims=p.get("claims"))
                vec += [nf[k] for k in NLI.FEATURE_NAMES]
                aux["best_window"] = nf["best_window"]
                aux["nli"] = {k: nf[k] for k in NLI.FEATURE_NAMES}
            out.append((vec, aux))
        self.cache[key] = out
        return out

    # ----------------------------------------------------------------------- fit
    def fit(self, rows: List[dict]):
        X, y = [], []
        for r in rows:
            rubric = json.loads(r["rubric"]) if isinstance(r["rubric"], str) else r["rubric"]
            covered = set(json.loads(r["points_covered"]) if isinstance(r["points_covered"], str) else r["points_covered"])
            for (vec, _), p in zip(self._answer_features(rubric, r["student_answer"] or ""), rubric):
                X.append(vec)
                y.append(int(p["id"] in covered))
        self.clf.fit(np.array(X), np.array(y))
        self._fitted = True
        return self

    # --------------------------------------------------------------------- predict
    def grade_one(self, rubric: List[dict], student_answer: str, max_marks: float) -> FeatureGradeResult:
        assert self._fitted, "call fit() first"
        answer = student_answer or ""
        feats = self._answer_features(rubric, answer)
        probs = self.clf.predict_proba(np.array([v for v, _ in feats]))[:, 1] if rubric else np.array([])
        pts, reasons = [], []
        for p, (vec, aux), pr in zip(rubric, feats, probs):
            vetoed, why = False, ""
            if self.use_factcheck:
                fc = apply_fact_check_text(answer, p)
                if fc is not None and fc.status == "contradicted":
                    vetoed, why = True, fc.reason
            nli = aux.get("nli")
            if (not vetoed) and self.nli_veto is not None and nli:
                if nli["contra_max"] >= self.nli_veto and nli["ent_max"] < 0.5:
                    vetoed, why = True, f"NLI: answer contradicts this point (p_contra={nli['contra_max']:.2f})"
                elif nli["contra_max"] >= self.nli_veto and nli["ent_max"] >= 0.5:
                    reasons.append(f"self_contradiction_possible:{p['id']}")
            credited = (pr >= self.threshold) and not vetoed
            pts.append(PointResult(p["id"], p["text"], float(p["marks"]), float(0.0 if vetoed else pr), credited,
                                   vetoed, why, aux.get("best_window", ""),
                                   dict(zip(self.feature_names, [float(x) for x in vec]))))
        if self.mode == "hard":
            total = sum(x.marks for x in pts if x.credited)
        else:
            total = sum(x.marks * x.prob for x in pts)
        score = round(min(total, max_marks) * 2) / 2
        risk = sum(x.marks * min(x.prob, 1 - x.prob) for x in pts)
        conf = float(np.clip(1 - risk / max_marks, 0, 1))
        if any(x.vetoed for x in pts):
            reasons.append("fact_check_contradiction")
        if possible_injection(answer):
            reasons.append("possible_prompt_injection")
        if not split_sentences(answer):
            reasons.append("empty_answer")
        if conf < self.review_threshold:
            reasons.append("low_confidence")
        return FeatureGradeResult(score, max_marks, round(conf, 3), round(risk, 3), bool(reasons), reasons, pts,
                                  self._feedback(pts))

    @staticmethod
    def _feedback(pts: List[PointResult]) -> str:
        got = [p for p in pts if p.credited]
        bad = [p for p in pts if p.vetoed]
        miss = [p for p in pts if not p.credited and not p.vetoed]
        parts = []
        if got:
            parts.append("Covered: " + "; ".join(f"{p.point_text}" for p in got) + ".")
        if bad:
            parts.append("Factual error on: " + "; ".join(p.point_text for p in bad) + ".")
        if miss:
            parts.append("Missing or unclear: " + "; ".join(p.point_text for p in miss) + ".")
        return " ".join(parts) if parts else "No rubric points to assess."

    def grade_batch(self, rows: List[dict]) -> List[FeatureGradeResult]:
        out = []
        for r in rows:
            rubric = json.loads(r["rubric"]) if isinstance(r["rubric"], str) else r["rubric"]
            out.append(self.grade_one(rubric, r["student_answer"], float(r["max_marks"])))
        return out
