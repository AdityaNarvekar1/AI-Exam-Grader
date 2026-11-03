"""
biencoder_scorer.py  (v2)
--------------------------
Doing the bi-encoder (sentence-transformer) route PROPERLY.  The Week 3 MiniLM experiment
failed for avoidable reasons, not because "transformers don't help":

  1. all-MiniLM-L6-v2 is trained for SYMMETRIC sentence similarity; a terse rubric point
     matched against an answer sentence is an ASYMMETRIC retrieval problem.  Retrieval models
     (bge / e5 / gte / multi-qa) are built for it and expect query/passage prefixes.
  2. Only the single best sentence was used; a point spread over two sentences was lost.
     Here: windows of 1-3 sentences.
  3. One scalar (the cosine) went into a one-feature logistic regression.  Here the cosine is
     one of several features (see feature_grader.py).

Features per (rubric point, answer)  [cosine, so roughly in -1..1; usually 0..1]
    bi_max      best cosine over all windows
    bi_top2     mean of the two best windows
    bi_full     cosine to the whole answer
    bi_margin   bi_max minus the mean cosine of the *other* points' best window - separates "this
                answer is about the topic" (all points high) from "this answer covers THIS point"

Backends: SentenceTransformerBackend (real) | StubBiEncoder (hashed bag-of-stems; tests only).

Suggested models:  BAAI/bge-base-en-v1.5  (query prefix applied automatically),
                   intfloat/e5-base-v2     ("query: " / "passage: "),
                   Alibaba-NLP/gte-base-en-v1.5, multi-qa-mpnet-base-dot-v1,
                   and sentence-transformers/all-MiniLM-L6-v2 as the Week-3 control.
For Hinglish / multilingual answers try intfloat/multilingual-e5-base or BAAI/bge-m3.
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from models.text_utils import split_sentences, make_windows
except ImportError:  # pragma: no cover
    from text_utils import split_sentences, make_windows

FEATURE_NAMES = ["bi_max", "bi_top2", "bi_full", "bi_margin"]


def default_prefixes(model_name: str):
    n = model_name.lower()
    if "bge" in n and "m3" not in n:
        return "Represent this sentence for searching relevant passages: ", ""
    if "e5" in n:
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None,
                 query_prefix: Optional[str] = None, passage_prefix: Optional[str] = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        dq, dp = default_prefixes(model_name)
        self.qp = dq if query_prefix is None else query_prefix
        self.pp = dp if passage_prefix is None else passage_prefix
        self.batch_size = batch_size

    def _enc(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        return np.asarray(self.model.encode([prefix + t for t in texts], normalize_embeddings=True,
                                            batch_size=self.batch_size, show_progress_bar=False))

    def encode_queries(self, texts):
        return self._enc(texts, self.qp)

    def encode_passages(self, texts):
        return self._enc(texts, self.pp)


class StubBiEncoder:
    """Hashed bag-of-stems embedding.  FOR TESTS ONLY - not a model."""
    model_name = "stub-biencoder"
    dim = 256

    def _vec(self, t):
        v = np.zeros(self.dim)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[hash(w[:5]) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return np.array([self._vec(t) for t in texts]).reshape(len(texts), self.dim)

    encode_passages = encode_queries


class BiEncoderPointScorer:
    def __init__(self, backend, max_window: int = 3):
        self.backend, self.max_window = backend, max_window

    def answer_windows(self, answer: str) -> List[str]:
        sents = split_sentences(answer)
        return make_windows(sents, self.max_window) if sents else []

    def features_for_answer(self, points: List[str], answer: str) -> List[Dict[str, float]]:
        """All points at once (shares one passage encoding).  Returns one feature dict per point."""
        windows = self.answer_windows(answer)
        if not windows:
            return [{k: 0.0 for k in FEATURE_NAMES} for _ in points]
        W = self.backend.encode_passages(windows)
        Q = self.backend.encode_queries(points)
        S = Q @ W.T                                         # [n_points, n_windows]
        full = S[:, -1]                                     # last window == whole answer
        best = S.max(axis=1)
        top2 = np.sort(S, axis=1)[:, -2:].mean(axis=1) if S.shape[1] > 1 else best
        out = []
        for i in range(len(points)):
            others = np.delete(best, i)
            margin = float(best[i] - others.mean()) if len(others) else 0.0
            out.append({"bi_max": float(best[i]), "bi_top2": float(top2[i]), "bi_full": float(full[i]),
                        "bi_margin": margin})
        return out
