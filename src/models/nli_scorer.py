"""
nli_scorer.py  (v2)  -  the transformer upgrade the project was missing
-------------------------------------------------------------------------
Reframes "did the student cover this rubric point?" as what it really is: TEXTUAL ENTAILMENT.

    premise    = a window of the student's answer (1-3 consecutive sentences, or the whole answer)
    hypothesis = the rubric point (and, separately, its atomic claims)

A cross-encoder reads premise and hypothesis TOGETHER, so word order, negation and role
assignment are visible to it - exactly what averaged GloVe vectors (and MiniLM cosine) cannot see.
It also returns a CONTRADICTION probability, which generalises the hand-written fact-binding
rules to any question: "a stack is FIFO" contradicts "a stack is LIFO" without anyone authoring
a binding.

Features produced per (rubric point, answer)  [all in 0..1]
    ent_max          best entailment of the WHOLE point over all windows
    ent_claim_mean   mean over atomic claims of (best entailment over windows)   -> partial coverage
    ent_claim_min    min  over atomic claims of the same                          -> "everything is said"
    contra_max       highest contradiction prob of the whole point over any window
    contra_at_best   contradiction prob in the window that best entails the point
    ent_minus_contra ent_max - contra_max

Backends
    HFNLIBackend    real model (needs `pip install torch transformers`; downloads from the Hub)
    StubNLIBackend  deterministic lexical stand-in used by the unit tests and for dry runs;
                    NOT a model - never report numbers from it

Suggested models (any 3-way NLI classifier works; label order is read from the model config):
    cross-encoder/nli-deberta-v3-base        small, strong, CPU-friendly       <- start here
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli   larger, stronger
    facebook/bart-large-mnli                 the classic; heavier
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # package-style and script-style imports both work
    from models.text_utils import split_sentences, make_windows, atomic_claims
except ImportError:  # pragma: no cover
    from text_utils import split_sentences, make_windows, atomic_claims

FEATURE_NAMES = ["ent_max", "ent_claim_mean", "ent_claim_min", "contra_max", "contra_at_best", "ent_minus_contra"]


class HFNLIBackend:
    """Cross-encoder NLI.  probs() -> array [n, 3] with columns (entailment, neutral, contradiction)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: Optional[str] = None,
                 batch_size: int = 16, max_length: int = 512):
        import torch  # lazy: not needed unless this backend is used
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                              and torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.batch_size, self.max_length = batch_size, max_length
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.idx = {}
        for i, lab in id2label.items():
            if "entail" in lab:
                self.idx["ent"] = i
            elif "contra" in lab:
                self.idx["con"] = i
            elif "neutral" in lab:
                self.idx["neu"] = i
        if set(self.idx) != {"ent", "con", "neu"}:
            raise ValueError(f"Cannot map NLI labels from model config: {id2label}. "
                             "Pass a 3-way NLI model (entailment/neutral/contradiction).")

    def probs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = []
        t = self.torch
        with t.no_grad():
            for i in range(0, len(premises), self.batch_size):
                enc = self.tok(list(premises[i:i + self.batch_size]), list(hypotheses[i:i + self.batch_size]),
                               truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                p = t.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                out.append(p[:, [self.idx["ent"], self.idx["neu"], self.idx["con"]]])
        return np.vstack(out) if out else np.zeros((0, 3))


class StubNLIBackend:
    """Deterministic lexical stand-in (token recall + naive negation flip).  FOR TESTS ONLY."""
    model_name = "stub-nli"

    def probs(self, premises, hypotheses):
        import re
        rows = []
        for p, h in zip(premises, hypotheses):
            pw, hw = set(re.findall(r"[a-z0-9]+", p.lower())), set(re.findall(r"[a-z0-9]+", h.lower()))
            rec = len(pw & hw) / max(1, len(hw))
            neg = ("not" in pw or "never" in pw) != ("not" in hw or "never" in hw)
            ent = max(0.0, rec - (0.6 if neg and rec > 0.4 else 0.0))
            con = 0.8 if (neg and rec > 0.4) else 0.02
            neu = max(0.0, 1.0 - ent - con)
            s = ent + neu + con
            rows.append([ent / s, neu / s, con / s])
        return np.array(rows).reshape(-1, 3)


class NLIPointScorer:
    def __init__(self, backend, max_window: int = 3, use_claims: bool = True):
        self.backend, self.max_window, self.use_claims = backend, max_window, use_claims

    def point_features(self, point_text: str, answer: str, claims: Optional[List[str]] = None) -> Dict[str, object]:
        sents = split_sentences(answer)
        if not sents:
            return {**{k: 0.0 for k in FEATURE_NAMES}, "best_window": "", "claims": []}
        windows = make_windows(sents, self.max_window)
        claims = claims if claims is not None else (atomic_claims(point_text) if self.use_claims else [point_text])
        hyps = [point_text] + [c for c in claims if c != point_text]
        pairs_p = [w for h in hyps for w in windows]
        pairs_h = [h for h in hyps for _ in windows]
        P = self.backend.probs(pairs_p, pairs_h).reshape(len(hyps), len(windows), 3)
        ent, con = P[:, :, 0], P[:, :, 2]
        whole_ent, whole_con = ent[0], con[0]
        best_w = int(np.argmax(whole_ent))
        claim_best = ent[1:].max(axis=1) if len(hyps) > 1 else np.array([whole_ent.max()])
        f = {
            "ent_max": float(whole_ent.max()),
            "ent_claim_mean": float(claim_best.mean()),
            "ent_claim_min": float(claim_best.min()),
            "contra_max": float(whole_con.max()),
            "contra_at_best": float(whole_con[best_w]),
        }
        f["ent_minus_contra"] = f["ent_max"] - f["contra_max"]
        return {**f, "best_window": windows[best_w], "claims": claims}
