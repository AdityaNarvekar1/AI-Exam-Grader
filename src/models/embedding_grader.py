"""
embedding_grader.py
--------------------
Group 2 - AI Exam Answer Grader
Week-3 CORE NLP MODEL ("Improved Approach" per Section 6, Requirement 2:
examples given are "Transformer embeddings / BERT-based model / sentence
transformers / RAG / LLM-based approach / hybrid NLP pipeline").

Design
------
Rather than a single whole-answer similarity score (as in the Week-2
baseline), this pipeline does concept-level semantic matching:

  For every rubric point (e.g. "Tuples are hashable / usable as dict keys"),
  find the single best-matching SENTENCE inside the student's answer using
  pretrained word-embedding cosine similarity, then classify - via a
  logistic regression calibrated on the labelled train split - whether that
  concept is genuinely covered.

This directly targets the two failure modes surfaced in the Week-2 error
analysis:
  - paraphrase_trap: a correct concept in different words should now get a
    high embedding similarity even with near-zero lexical overlap.
  - keyword_stuffing: an incoherent list of rubric vocabulary should NOT
    automatically produce a high per-sentence similarity, because isolated
    keywords compose into a very different sentence vector than a full,
    coherent explanatory sentence.

Embeddings used: spaCy en_core_web_md (300-dim, GloVe-trained static word
vectors, averaged per sentence by spaCy). This is a genuine pretrained
embedding model (not TF-IDF), obtained via spaCy's GitHub-hosted model
release - the environment this was built in could not reach
huggingface.co, so a full sentence-transformer/BERT model was not
downloadable here. `SentenceTransformerBackend` below implements the swap
to all-MiniLM-L6-v2 and is a drop-in replacement for `EmbeddingBackend`
(same two methods) - use it via `get_backend("sentence_transformer")` on a
machine with unrestricted internet access.

Interface mirrors baseline_grader.py's grade_one()/grade_batch()/fit() so
either model can be dropped behind the same Week-6 API without redesign.
"""

import json
from dataclasses import dataclass
from typing import List

import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression

try:
    from models.fact_binding_checker import apply_fact_check
except ImportError:  # running this file directly (python embedding_grader.py)
    from fact_binding_checker import apply_fact_check


class EmbeddingBackend:
    """Pretrained word-vector backend (spaCy en_core_web_md). Works offline
    once the model is installed; used as the Week 3 default because it was
    the only pretrained option reachable from the sandbox this was built in."""

    def __init__(self, model_name: str = "en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def sentences_with_vectors(self, text: str):
        doc = self.nlp(text)
        out = []
        for sent in doc.sents:
            s = sent.text.strip()
            if s and sent.vector_norm > 0:
                out.append((s, sent.vector / sent.vector_norm))
        if not out and doc.vector_norm > 0:
            out = [(text.strip(), doc.vector / doc.vector_norm)]
        return out

    def vector(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        if doc.vector_norm == 0:
            return doc.vector
        return doc.vector / doc.vector_norm


class SentenceTransformerBackend:
    """
    Drop-in replacement for EmbeddingBackend using a real sentence-transformer
    (contextual, not just averaged static word vectors). Needs unrestricted
    internet on first run to download the model from huggingface.co - not
    available in the sandbox this project was built in, so this class is
    untested end-to-end here; the code below is correct and ready to run.

    Setup (one-time):
        pip install sentence-transformers
        python -c "from sentence_transformers import SentenceTransformer; \
                    SentenceTransformer('all-MiniLM-L6-v2')"   # downloads + caches the model

    Usage - identical interface to EmbeddingBackend, so no other file needs
    to change:
        grader = EmbeddingSemanticGrader(backend=SentenceTransformerBackend())
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run:\n"
                "    pip install sentence-transformers\n"
                "then retry."
            ) from e

        self.model = SentenceTransformer(model_name)

        # Sentence-transformers only embeds text; it doesn't split paragraphs
        # into sentences. Use spaCy's rule-based sentencizer for that (no
        # downloaded model or vectors needed - this works fully offline).
        self._sent_nlp = spacy.blank("en")
        self._sent_nlp.add_pipe("sentencizer")

    def _split_sentences(self, text: str) -> List[str]:
        doc = self._sent_nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        return sents if sents else [text.strip()]

    def sentences_with_vectors(self, text: str):
        sents = self._split_sentences(text)
        vecs = self.model.encode(sents, normalize_embeddings=True)
        return list(zip(sents, vecs))

    def vector(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]


def get_backend(name: str = "spacy"):
    """
    Factory so callers/CLI scripts can switch embedding backends without
    editing any other code.
        name="spacy"                -> EmbeddingBackend (en_core_web_md; default, sandbox-safe)
        name="sentence_transformer" -> SentenceTransformerBackend (all-MiniLM-L6-v2; needs internet)
    """
    if name == "spacy":
        return EmbeddingBackend()
    if name == "sentence_transformer":
        return SentenceTransformerBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'spacy' or 'sentence_transformer'.")


@dataclass
class PointMatch:
    point_id: str
    point_text: str
    marks: float
    best_sentence: str
    similarity: float
    coverage_prob: float
    credited: bool
    fact_check: str = "not_applicable"  # "confirmed" | "contradicted" | "neutral" | "not_applicable"


@dataclass
class EmbeddingGraderResult:
    qid: str
    student_id: str
    score: float
    max_marks: float
    confidence: float
    point_matches: List[PointMatch]
    feedback: str


class EmbeddingSemanticGrader:
    """Approach 'Improved' (Week 3/4): per-rubric-point semantic concept matching."""

    def __init__(self, backend: EmbeddingBackend = None, scoring_mode: str = "hard",
                 threshold: float = 0.5):
        """
        scoring_mode:
          "hard" - binary credit/no-credit per point at probability >= threshold
                   (default; matches the intrinsic precision/recall/F1 evaluation).
          "soft" - continuous credit = marks * coverage_prob per point, summed.
                   Trades some interpretability ("was this point covered, yes
                   or no") for smoother, less binary-boundary-sensitive scores.
        threshold: decision cutoff for "hard" mode. Default 0.5 (Week 3/4);
                   Week 5 tunes this on a held-out validation split instead
                   of assuming it - see src/evaluation/tune_threshold.py.
        """
        self.backend = backend or EmbeddingBackend()
        self.clf = LogisticRegression(class_weight="balanced")
        self._fitted = False
        self.scoring_mode = scoring_mode
        self.threshold = threshold

    def _point_best_match(self, point_text: str, student_answer: str):
        point_vec = self.backend.vector(point_text)
        best_sent, best_sim = "", -1.0
        for sent, vec in self.backend.sentences_with_vectors(student_answer):
            sim = float(np.dot(point_vec, vec))
            if sim > best_sim:
                best_sim, best_sent = sim, sent
        if not best_sent:  # no sentences at all (e.g. empty answer): -1.0 was a leaked sentinel
            best_sim = 0.0
        return best_sent, best_sim

    def fit(self, train_rows: List[dict]):
        """
        Builds a POINT-LEVEL training set from the row-level dataset: every
        (rubric point, student answer) pair becomes one training example,
        labelled 1/0 using the dataset's ground-truth `points_covered`.
        """
        X, y = [], []
        for r in train_rows:
            rubric = json.loads(r["rubric"])
            covered = set(json.loads(r["points_covered"]))
            for p in rubric:
                _, sim = self._point_best_match(p["text"], r["student_answer"])
                X.append([sim])
                y.append(1 if p["id"] in covered else 0)
        self.clf.fit(np.array(X), np.array(y))
        self._fitted = True
        return self

    def predict_point(self, point_id, point_text, marks, student_answer,
                       point_meta: dict = None) -> PointMatch:
        """
        point_meta: the full rubric-point dict (optionally carrying
        "binding" or "cues" keys - see fact_binding_checker.py). When
        present, a detected contradiction VETOES credit regardless of how
        high the similarity/probability was - this is what fixes the
        DS01/ML01-style factual-inversion failures documented in the
        Week 5 report.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(train_rows) before grading.")
        sent, sim = self._point_best_match(point_text, student_answer)
        prob = float(self.clf.predict_proba(np.array([[sim]]))[0][1])
        credited = prob >= self.threshold
        fact_check_status = "not_applicable"

        if point_meta is not None:
            check = apply_fact_check(sent, point_meta)
            if check is not None:
                fact_check_status = check.status
                if check.status == "contradicted":
                    credited = False  # veto, regardless of similarity/probability

        return PointMatch(point_id=point_id, point_text=point_text, marks=marks,
                           best_sentence=sent, similarity=round(sim, 3),
                           coverage_prob=round(prob, 3), credited=credited,
                           fact_check=fact_check_status)

    def grade_one(self, reference_answer, rubric, student_answer, max_marks,
                  qid: str = "", student_id: str = "") -> EmbeddingGraderResult:
        matches = [self.predict_point(p["id"], p["text"], p["marks"], student_answer, point_meta=p)
                   for p in rubric]
        if self.scoring_mode == "soft":
            total = sum(m.marks * m.coverage_prob for m in matches)
        else:
            total = sum(m.marks for m in matches if m.credited)
        score = round(min(total, max_marks) * 2) / 2
        confidence = round(float(np.mean([m.coverage_prob for m in matches])), 2) if matches else 0.0

        contradicted = [m.point_text for m in matches if m.fact_check == "contradicted"]
        missed = [m.point_text for m in matches if not m.credited and m.fact_check != "contradicted"]
        feedback_parts = []
        if contradicted:
            feedback_parts.append("Contains a factual error on: " + "; ".join(contradicted))
        if missed:
            feedback_parts.append("Missing or unclear on: " + "; ".join(missed))
        feedback = " ".join(feedback_parts) if feedback_parts else \
            "All rubric concepts are addressed (in the student's own words or paraphrased)."

        return EmbeddingGraderResult(qid=qid, student_id=student_id, score=score,
                                      max_marks=max_marks, confidence=confidence,
                                      point_matches=matches, feedback=feedback)

    def grade_batch(self, rows: List[dict]) -> List[EmbeddingGraderResult]:
        results = []
        for r in rows:
            rubric = json.loads(r["rubric"])
            results.append(self.grade_one(
                reference_answer=r["reference_answer"], rubric=rubric,
                student_answer=r["student_answer"], max_marks=float(r["max_marks"]),
                qid=r["qid"], student_id=r["student_id"],
            ))
        return results


if __name__ == "__main__":
    # Minimal smoke test - paraphrase should now score well.
    rubric = [{"id": "p1", "text": "List is mutable, tuple is immutable", "marks": 2.0}]
    train_rows = [{
        "rubric": json.dumps(rubric),
        "student_answer": "A list can be changed after creation, a tuple cannot.",
        "points_covered": json.dumps(["p1"]),
    }, {
        "rubric": json.dumps(rubric),
        "student_answer": "Python has many data types used for programming.",
        "points_covered": json.dumps([]),
    }]
    g = EmbeddingSemanticGrader().fit(train_rows)
    r = g.grade_one("", rubric,
                     "One of these two collection types can have its contents "
                     "rearranged after you build it, whereas the other is locked "
                     "in place the moment you create it.", 2.0, "PY01", "S_demo")
    print(r.score, r.confidence, r.feedback)
