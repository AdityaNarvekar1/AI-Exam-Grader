"""
model_adapters.py  (v2)
------------------------
Registers the model-backed graders with graders.REGISTRY so run_benchmark.py can evaluate them under
exactly the same protocols as the trivial/lexical baselines.  Imported lazily by
graders.load_optional_adapters(); heavy imports (torch, transformers, spaCy ...) happen only when an
adapter is actually USED, so a missing dependency never breaks the other graders.

ENVIRONMENT VARIABLES (set before running run_benchmark.py)
  G2_NLI_MODEL     default cross-encoder/nli-deberta-v3-base       ("stub" = test stand-in, NOT a model)
  G2_BI_MODEL      default BAAI/bge-base-en-v1.5                   ("stub" = test stand-in)
  G2_NLI_VETO      e.g. 0.9  -> veto a point when contradiction prob >= value and entailment < 0.5 (default: off)
  G2_DEVICE        cuda | cpu | mps (default: auto)
  G2_LLM_BACKEND   ollama | openai | hf            G2_LLM_MODEL  e.g. qwen2.5:7b-instruct
  G2_LLM_URL       ollama host or OpenAI-compatible base URL      G2_LLM_VOTES  default 1
  G2_LLM_NOREF     "1" = do not show the reference answer to the judge

REGISTERED NAMES
  spacy_embedding, spacy_hybrid     the ORIGINAL Week 3-6 models (needs spaCy + en_core_web_md)
  feat_lex                          FeatureGrader, lexical features only (+ whole-answer veto) - no downloads
  bi, nli, nli_bi                   FeatureGrader with bi-encoder / NLI / both       (+ "_exp": expected-score mode)
  judge                             LLM judge (rubric-grounded, evidence-verified)
  stack_base                        learned stack of tfidf + lexical                  - no downloads
  stack_nli                         learned stack of tfidf + lexical + nli_bi
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(HERE))

import graders as G                      # noqa: E402
from graders import GradeOut            # noqa: E402

_SINGLETON = {}
_FEATURE_CACHE = {}   # shared by every FeatureGrader in the process: LOQO would otherwise recompute 15x


def _nli_scorer():
    if "nli" not in _SINGLETON:
        import nli_scorer as N
        name = os.environ.get("G2_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
        if name == "stub":
            print("[adapters] WARNING: G2_NLI_MODEL=stub - results are NOT from a real model")
            be = N.StubNLIBackend()
        else:
            be = N.HFNLIBackend(name, device=os.environ.get("G2_DEVICE"))
        _SINGLETON["nli"] = N.NLIPointScorer(be)
    return _SINGLETON["nli"]


def _bi_scorer():
    if "bi" not in _SINGLETON:
        import biencoder_scorer as B
        name = os.environ.get("G2_BI_MODEL", "BAAI/bge-base-en-v1.5")
        if name == "stub":
            print("[adapters] WARNING: G2_BI_MODEL=stub - results are NOT from a real model")
            be = B.StubBiEncoder()
        else:
            be = B.SentenceTransformerBackend(name, device=os.environ.get("G2_DEVICE"))
        _SINGLETON["bi"] = B.BiEncoderPointScorer(be)
    return _SINGLETON["bi"]


# ------------------------------------------------------------------------------ spaCy (original models)
def _spacy_backend():
    if "spacy" not in _SINGLETON:
        from embedding_grader import get_backend
        _SINGLETON["spacy"] = get_backend("spacy")
    return _SINGLETON["spacy"]


class SpacyEmbedding:
    name = "spacy_embedding (original Week-3 model)"

    def fit(self, rows):
        from embedding_grader import EmbeddingSemanticGrader
        self.g = EmbeddingSemanticGrader(backend=_spacy_backend()).fit(rows)
        return self

    def predict(self, rows):
        out = []
        for res in self.g.grade_batch(rows):
            out.append(GradeOut(res.score, res.confidence,
                                {m.point_id: m.credited for m in res.point_matches},
                                {m.point_id: m.coverage_prob for m in res.point_matches},
                                [m.point_id for m in res.point_matches if m.fact_check == "contradicted"]))
        return out


class SpacyHybrid:
    name = "spacy_hybrid (original Week-6 default)"

    def fit(self, rows):
        from baseline_grader import TFIDFBaselineGrader
        from embedding_grader import EmbeddingSemanticGrader
        from hybrid_grader import HybridGrader
        b = TFIDFBaselineGrader().fit(rows)
        e = EmbeddingSemanticGrader(backend=_spacy_backend()).fit(rows)
        self.g = HybridGrader(b, e)
        return self

    def predict(self, rows):
        out = []
        for res in self.g.grade_batch(rows):
            out.append(GradeOut(res.score, res.confidence,
                                {m.point_id: m.credited for m in res.point_matches},
                                {m.point_id: m.coverage_prob for m in res.point_matches},
                                [m.point_id for m in res.point_matches if m.fact_check == "contradicted"],
                                {"needs_review": res.needs_human_review}))
        return out


# ------------------------------------------------------------------------------ FeatureGrader family
class _FeatureAdapter:
    use_nli = use_bi = False
    mode = "hard"
    name = "feature"

    def fit(self, rows):
        from feature_grader import FeatureGrader
        veto = os.environ.get("G2_NLI_VETO")
        self.g = FeatureGrader(nli=_nli_scorer() if self.use_nli else None, bi=_bi_scorer() if self.use_bi else None,
                               nli_veto=float(veto) if (veto and self.use_nli) else None,
                               mode=self.mode, cache=_FEATURE_CACHE).fit(rows)
        return self

    def predict(self, rows):
        return [GradeOut(r.score, r.confidence, r.point_credit, r.point_prob, r.contradicted_points,
                         {"needs_review": r.needs_review, "reasons": r.review_reasons})
                for r in self.g.grade_batch(rows)]


def _mk(cls_name, nli, bi, mode, label):
    return type(cls_name, (_FeatureAdapter,), {"use_nli": nli, "use_bi": bi, "mode": mode, "name": label})


FeatLex = _mk("FeatLex", False, False, "hard", "feature grader: lexical only")
Bi = _mk("Bi", False, True, "hard", "feature grader: lexical + bi-encoder")
Nli = _mk("Nli", True, False, "hard", "feature grader: lexical + NLI")
NliBi = _mk("NliBi", True, True, "hard", "feature grader: lexical + NLI + bi-encoder")
FeatLexE = _mk("FeatLexE", False, False, "expected", "feature grader (expected score): lexical only")
BiE = _mk("BiE", False, True, "expected", "feature grader (expected score): lexical + bi-encoder")
NliE = _mk("NliE", True, False, "expected", "feature grader (expected score): lexical + NLI")
NliBiE = _mk("NliBiE", True, True, "expected", "feature grader (expected score): lexical + NLI + bi-encoder")


# ------------------------------------------------------------------------------ LLM judge
class JudgeAdapter:
    name = "llm_judge"

    def _judge(self):
        if "judge" not in _SINGLETON:
            import llm_judge as J
            kind = os.environ.get("G2_LLM_BACKEND", "ollama")
            model = os.environ.get("G2_LLM_MODEL", "qwen2.5:7b-instruct")
            url = os.environ.get("G2_LLM_URL")
            if kind == "ollama":
                be = J.OllamaBackend(model, url or "http://localhost:11434")
            elif kind == "openai":
                be = J.OpenAICompatBackend(url or "http://localhost:8000/v1", model)
            elif kind == "hf":
                be = J.HFBackend(model)
            else:
                raise ValueError(f"unknown G2_LLM_BACKEND {kind}")
            safe = "".join(c if c.isalnum() else "_" for c in f"{kind}_{model}")
            be = J.CachedBackend(be, str(ROOT / "results" / "v2" / f"llm_cache_{safe}.jsonl"))
            _SINGLETON["judge"] = J.LLMJudge(be, n_votes=int(os.environ.get("G2_LLM_VOTES", "1")),
                                             include_reference=os.environ.get("G2_LLM_NOREF") != "1")
        return _SINGLETON["judge"]

    def fit(self, rows):          # zero-shot: nothing to fit
        return self

    def predict(self, rows):
        j = self._judge()
        out = []
        for r, res in zip(rows, j.grade_batch(rows)):
            out.append(GradeOut(res.score, res.confidence, res.point_credit, res.point_prob, res.contradicted_points,
                                {"needs_review": res.needs_review, "reasons": res.review_reasons,
                                 "parse_failed": res.parse_failed}))
        return out


# ------------------------------------------------------------------------------ learned stacks
class _StackAdapter:
    name = "stack"
    members = ()

    def fit(self, rows):
        from stacked_grader import StackedGrader
        self.s = StackedGrader({n: G.REGISTRY[n] for n in self.members}).fit(rows)
        return self

    def predict(self, rows):
        scores, _ = self.s.predict(rows)
        return [GradeOut(sc, extra={"weights": self.s.weights_}) for sc in scores]


class StackBase(_StackAdapter):
    name = "stack(tfidf + lexical)"
    members = ("tfidf", "lexical")


class StackNli(_StackAdapter):
    name = "stack(tfidf + lexical + nli_bi)"
    members = ("tfidf", "lexical", "nli_bi")


for _n, _c in {"spacy_embedding": SpacyEmbedding, "spacy_hybrid": SpacyHybrid, "feat_lex": FeatLex, "bi": Bi, "nli": Nli,
               "nli_bi": NliBi, "feat_lex_exp": FeatLexE, "bi_exp": BiE, "nli_exp": NliE, "nli_bi_exp": NliBiE,
               "judge": JudgeAdapter, "stack_base": StackBase, "stack_nli": StackNli}.items():
    G.register(_n, _c)
