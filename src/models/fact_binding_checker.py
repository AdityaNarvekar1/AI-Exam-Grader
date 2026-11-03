"""
fact_binding_checker.py  (v2 - bug-fixed and hardened)
--------------------------------------------------------
Group 2 - AI Exam Answer Grader

A lexical *veto layer* that checks whether the right words are bound to the
right entity ("stack ... LIFO"), because embedding / TF-IDF similarity only
measures vocabulary presence.

What changed versus the Week-5 version (all found by auditing the code):

  1. SUBSTRING MATCHING BUG.  `"mutable" in "immutable"` is True, so
     "a list is immutable" used to be CONFIRMED as "list is mutable".  The
     same bug made "type i" match inside "type ii" and "stack" inside
     "stackoverflow".  All matching is now whole-word (token-boundary), with
     hyphen/space normalisation ("last-in-first-out" == "last in first out")
     and optional plural suffix.
  2. NEGATION.  "a list is NOT mutable" is now read as the opposite value
     (a negated correct term counts as wrong, a negated wrong term as correct).
  3. PROXIMITY BINDING (commas no longer split clauses, so "A Type I error is X, which is a
     false negative" binds correctly; only contrast words and ; : split).  "A stack is FIFO and a queue is LIFO" (no comma) used to
     be "confirmed" because both values occurred in the same clause.  Each
     subject mention is now bound to its NEAREST value mention (a value after
     the subject is preferred over one before it).
  4. WHOLE-ANSWER SCOPE (optional).  `apply_fact_check_text()` scans every
     sentence of the answer, so a contradiction that is not in the single
     best-matching sentence is still caught.
  5. SELF-CONTRADICTION.  An answer that binds the subject to BOTH the correct
     and the competing value is reported as "contradicted" (a human grader
     would not give full credit for a self-contradictory claim).

Still true (and still honest): this is hand-authored metadata per rubric point
and does not generalise automatically.  The general fix is an NLI/LLM
contradiction check - see nli_scorer.py and llm_judge.py, which make this module
a cheap *secondary* guard rather than the only line of defence.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

CLAUSE_SPLIT_RE = re.compile(r"\b(?:while|whereas|but|however|unlike|although)\b|[;:]", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEGATORS = {"not", "no", "never", "cannot", "isnt", "isn't", "doesnt", "doesn't", "dont", "don't",
            "wasnt", "wasn't", "arent", "aren't", "cant", "can't", "wont", "won't", "neither", "nor"}
NEGATION_WINDOW = 3          # negator may sit up to 3 tokens before the value term
BINDING_WINDOW = 16          # a value must be within 16 tokens of its subject to bind (definitions are wordy)
BEFORE_WINDOW = 6            # fallback only: a value BEFORE the subject must be this close


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", text)   # last-in-first-out -> last in first out
    return text


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(_normalise(text))]


def _term_tokens(term: str) -> List[str]:
    return [t for t, _, _ in _tokens(term)]


def _tok_eq(a: str, b: str) -> bool:
    """Token equality allowing a plural -s/-es on the text side ('lists' == 'list')."""
    return a == b or a == b + "s" or a == b + "es"


def find_mentions(tokens: List[Tuple[str, int, int]], terms: List[str]) -> List[Tuple[int, int]]:
    """Whole-word (token-sequence) matches of any term. Returns [(first_tok_idx, last_tok_idx)]."""
    words = [t for t, _, _ in tokens]
    out = []
    for term in terms:
        tt = _term_tokens(term)
        if not tt:
            continue
        n = len(tt)
        for i in range(len(words) - n + 1):
            if all(_tok_eq(words[i + j], tt[j]) for j in range(n)):
                out.append((i, i + n - 1))
    # drop mentions fully contained in a longer mention
    out = sorted(set(out))
    return [m for m in out if not any(o != m and o[0] <= m[0] and m[1] <= o[1] for o in out)]


def _is_negated(words: List[str], first_idx: int) -> bool:
    lo = max(0, first_idx - NEGATION_WINDOW)
    return any(w in NEGATORS for w in words[lo:first_idx])


def _clauses(sentence: str) -> List[str]:
    parts = CLAUSE_SPLIT_RE.split(sentence)
    return [p.strip() for p in parts if p and p.strip()]


def _contains_any(text: str, terms: List[str]) -> bool:
    """Whole-word containment (v2: no longer a raw substring test)."""
    return bool(find_mentions(_tokens(text), terms))


@dataclass
class BindingCheckResult:
    status: str  # "confirmed", "contradicted", "neutral"
    reason: str


class EntityBindingCheck:
    """
    binding = {
        "subject_terms": [...],               # e.g. ["stack"]
        "correct_value_terms": [...],         # e.g. ["lifo", "last in first out"]
        "contradicting_value_terms": [...],   # e.g. ["fifo", "first in first out"]
    }
    """

    def _bind_clause(self, clause: str, binding: dict) -> List[str]:
        """Returns one of 'correct'/'wrong' per subject mention that could be bound."""
        toks = _tokens(clause)
        words = [t for t, _, _ in toks]
        subj = find_mentions(toks, binding["subject_terms"])
        if not subj:
            return []
        good = [(m, "correct") for m in find_mentions(toks, binding["correct_value_terms"])]
        bad = [(m, "wrong") for m in find_mentions(toks, binding.get("contradicting_value_terms", []))]
        values = good + bad
        if not values:
            return []
        verdicts = []
        for s_first, s_last in subj:
            after = [(v_first - s_last, kind, v_first) for (v_first, v_last), kind in values
                     if v_first > s_last and v_first - s_last <= BINDING_WINDOW]
            before = [((s_first - v_last), kind, v_first) for (v_first, v_last), kind in values
                      if v_last < s_first and s_first - v_last <= BEFORE_WINDOW]
            # "X is V" is by far the commonest pattern: a value AFTER the subject always wins;
            # a value BEFORE it ("V is what X does") is only a fallback.
            cand = min(after) if after else (min(before) if before else None)
            if cand is None:
                continue
            _, kind, v_first = cand
            if _is_negated(words, v_first):
                kind = "wrong" if kind == "correct" else "correct"
            verdicts.append(kind)
        return verdicts

    def check(self, sentence: str, binding: dict) -> BindingCheckResult:
        verdicts: List[str] = []
        for clause in (_clauses(sentence) or [sentence]):
            verdicts += self._bind_clause(clause, binding)
        n_wrong, n_right = verdicts.count("wrong"), verdicts.count("correct")
        if n_wrong and n_right:
            return BindingCheckResult("contradicted",
                                       f"Self-contradictory: subject {binding['subject_terms']} is bound "
                                       f"to both the correct and the competing value in: \"{sentence}\"")
        if n_wrong:
            return BindingCheckResult("contradicted",
                                       f"Subject {binding['subject_terms']} appears paired with the "
                                       f"competing value (or the negated correct value), in: \"{sentence}\"")
        if n_right:
            return BindingCheckResult("confirmed", "Correct entity-value binding found.")
        return BindingCheckResult("neutral", "No clear binding signal either way; deferring to the similarity score.")


class ContradictionCuePhrases:
    """
    cues = {"contradicting_phrases": [...], "confirming_phrases": [...]}
    Hand-authored per rubric point; whole-word matching (v2). A specific wrong
    claim outranks a bare topic word ("tradeoff") that may appear in both a
    correct and an inverted sentence.
    """

    def check(self, sentence: str, cues: dict) -> BindingCheckResult:
        toks = _tokens(sentence)
        words = [t for t, _, _ in toks]
        contra = find_mentions(toks, cues.get("contradicting_phrases", []))
        confirm = find_mentions(toks, cues.get("confirming_phrases", []))
        # a negated contradicting phrase ("do NOT always happen together") is not a contradiction
        contra = [m for m in contra if not _is_negated(words, m[0])]
        if contra:
            return BindingCheckResult("contradicted",
                                       f"Contains a phrase suggesting the relationship is inverted: \"{sentence}\"")
        if confirm:
            return BindingCheckResult("confirmed", "Contains a phrase consistent with the correct relationship.")
        return BindingCheckResult("neutral", "No cue phrase matched; deferring to the similarity score.")


def apply_fact_check(sentence: str, rubric_point: dict) -> Optional[BindingCheckResult]:
    """Sentence-scope check (backwards compatible). None if the point has no metadata."""
    if rubric_point.get("binding"):
        return EntityBindingCheck().check(sentence, rubric_point["binding"])
    if rubric_point.get("cues"):
        return ContradictionCuePhrases().check(sentence, rubric_point["cues"])
    return None


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def apply_fact_check_text(answer_text: str, rubric_point: dict) -> Optional[BindingCheckResult]:
    """
    Whole-answer scope (v2): check EVERY sentence, not just the best match.
    Any contradicted sentence vetoes; otherwise any confirmed sentence confirms.
    """
    if not (rubric_point.get("binding") or rubric_point.get("cues")):
        return None
    results = [apply_fact_check(s, rubric_point) for s in (split_sentences(answer_text) or [answer_text])]
    results = [r for r in results if r is not None]
    for r in results:
        if r.status == "contradicted":
            return r
    for r in results:
        if r.status == "confirmed":
            return r
    return BindingCheckResult("neutral", "No clear binding signal either way; deferring to the similarity score.")


# ---------------------------------------------------------------------------
# Scope note: EntityBindingCheck generalises to any rubric point with a clear
# two-way entity<->value contrast (list/tuple, stack/queue, Type I/II, bagging/
# boosting, INNER/LEFT join ...) but still needs the metadata to be authored.
# ContradictionCuePhrases is a per-point proof of concept. For an arbitrary
# question bank use the NLI / LLM contradiction signal instead.
# ---------------------------------------------------------------------------
