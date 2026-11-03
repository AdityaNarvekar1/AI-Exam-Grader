"""
text_utils.py  (v2)
--------------------
Dependency-free text helpers shared by the NLI, bi-encoder and LLM-judge modules.

  split_sentences     regex sentence splitter (no spaCy needed)
  make_windows        1..k consecutive-sentence windows (+ the whole answer): a rubric point
                      is often spread over two sentences, which "best single sentence"
                      matching (Week 3-6) cannot see
  atomic_claims       splits a compound rubric point ("X, while Y") into separately checkable
                      claims, so a point can earn PARTIAL credit instead of all-or-nothing
  is_meta_point       detects rubric points that describe the answer instead of asserting a fact
                      ("a correct example is given") - NLI cannot verify those; the LLM judge can
  possible_injection  cheap flag for answers that address the grader instead of the question
"""

import re
from typing import List

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|,\s*(?:and|while|whereas|but|which|so|because)\b|\b(?:while|whereas)\b)\s*",
                           re.IGNORECASE)
_META = re.compile(r"\b(a|an)\s+(correct|valid|appropriate|suitable|relevant|good)\s+"
                   r"(illustrative\s+)?(example|illustration|diagram|analogy)\b|"
                   r"\bis\s+(given|provided|included|shown|described)\b|\bexample\s+is\s+(given|provided)\b",
                   re.IGNORECASE)
_INJECT = re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(rubric|instructions?|above|previous|grader|grading)\b|"
                     r"\b(award|give|assign)\b.{0,30}\b(full|maximum|max|10/10|100)\b.{0,15}\b(marks?|score|points?)\b|"
                     r"\bthis answer is (perfect|correct|complete)\b|\bas an ai\b|\bsystem prompt\b",
                     re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    s = [x.strip() for x in SENT_RE.split(text or "") if x and x.strip()]
    return s or ([text.strip()] if text and text.strip() else [])


def make_windows(sentences: List[str], max_len: int = 3, include_full: bool = True) -> List[str]:
    """All windows of 1..max_len consecutive sentences, plus the full answer."""
    out, seen = [], set()
    n = len(sentences)
    for k in range(1, max_len + 1):
        for i in range(0, n - k + 1):
            w = " ".join(sentences[i:i + k])
            if w not in seen:
                seen.add(w)
                out.append(w)
    full = " ".join(sentences)
    if include_full and full and full not in seen:
        out.append(full)
    return out


def atomic_claims(point_text: str, min_words: int = 3) -> List[str]:
    """
    Heuristic decomposition of a compound rubric point.  Always returns >=1 claim; claims
    shorter than `min_words` are merged back so we never hand NLI a fragment like "tuples".
    """
    parts = [p.strip(" ,.") for p in _CLAUSE_SPLIT.split(point_text) if p and p.strip(" ,.")]
    merged: List[str] = []
    for p in parts:
        if merged and len(p.split()) < min_words:
            merged[-1] = merged[-1] + ", " + p
        elif len(p.split()) < min_words and not merged:
            merged.append(p)
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[0].split()) < min_words:
        merged = [merged[0] + ", " + merged[1]] + merged[2:]
    return merged or [point_text.strip()]


def is_meta_point(point_text: str) -> bool:
    return bool(_META.search(point_text or ""))


def possible_injection(answer: str) -> bool:
    return bool(_INJECT.search(answer or ""))


def norm_for_match(s: str) -> str:
    """Lowercase, unify quotes/dashes, strip punctuation, collapse whitespace (for quote verification)."""
    s = (s or "").lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
