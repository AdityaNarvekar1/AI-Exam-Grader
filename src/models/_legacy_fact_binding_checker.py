# FROZEN COPY of the Week-5 checker, kept ONLY so evaluate_fact_check.py can show before/after. Do not use.
"""
fact_binding_checker.py
--------------------------
Group 2 - AI Exam Answer Grader
Week-5 follow-up: a direct fix for the "vocabulary presence is not factual
correctness" limitation documented in the Week 5 report (Section 6) - the
DS01 (stack/queue definitions swapped, scored 5.0/5.0) and ML01 (tradeoff
claim inverted, scored 10.0/10.0) failures.

Why this works where pure similarity doesn't
----------------------------------------------
Embedding similarity (and TF-IDF) measure whether the RIGHT WORDS are
present. They cannot tell that "stack" was bound to "FIFO" instead of
"LIFO" - the sentence vector for a swapped claim is almost identical to
the vector for the correct claim, because it contains the same words.

This module adds a second, lexical/pattern-based pass that checks WHICH
words are bound to WHICH entity, as a veto layer on top of (not a
replacement for) the existing embedding-similarity scorer:

  1. EntityBindingCheck - for rubric points that assert a specific
     "X is Y" binding (e.g. "stack is LIFO"), splits the matched sentence
     into clauses and checks whether the correct entity-value pairing
     appears, or whether the COMPETING (wrong) pairing appears instead.
     If a contradiction is detected, credit is vetoed regardless of how
     high the embedding similarity/probability was. Generalises to any
     rubric point with a clear binary entity<->attribute distinction
     (tested here on DS01, and also applicable to PY01, ST02).

  2. ContradictionCuePhrases - for rubric points asserting a RELATIONSHIP
     (not a simple entity swap), e.g. ML01's bias-variance tradeoff, a
     small hand-authored list of phrases that indicate the relationship
     has been inverted (e.g. "always happen together" vs the correct
     "tradeoff"/"inversely related"). This is intentionally narrower and
     more hand-authored than (1) - see the module-level note on scope at
     the bottom of this file for an honest assessment of its limits.

This is a hybrid NLP pipeline (lexical pattern-matching + embedding
similarity), one of Section 6's own listed "Improved Approach" examples.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

CLAUSE_SPLIT_RE = re.compile(r"\b(?:while|whereas|but|however)\b|[,;]", re.IGNORECASE)


def _clauses(sentence: str) -> List[str]:
    parts = CLAUSE_SPLIT_RE.split(sentence)
    return [p.strip() for p in parts if p.strip()]


def _contains_any(text: str, terms: List[str]) -> bool:
    text_l = text.lower()
    return any(t.lower() in text_l for t in terms)


@dataclass
class BindingCheckResult:
    status: str  # "confirmed", "contradicted", "neutral"
    reason: str


class EntityBindingCheck:
    """
    binding = {
        "subject_terms": [...],          # e.g. ["stack"]
        "correct_value_terms": [...],    # e.g. ["lifo", "last-in-first-out", "last in first out"]
        "contradicting_value_terms": [...],  # e.g. ["fifo", "first-in-first-out", "first in first out"]
    }
    """

    def check(self, sentence: str, binding: dict) -> BindingCheckResult:
        clauses = _clauses(sentence) or [sentence]
        found_contradiction, found_confirmation = False, False
        for clause in clauses:
            has_subject = _contains_any(clause, binding["subject_terms"])
            if not has_subject:
                continue
            has_correct = _contains_any(clause, binding["correct_value_terms"])
            has_wrong = _contains_any(clause, binding.get("contradicting_value_terms", []))
            if has_wrong and not has_correct:
                found_contradiction = True
            elif has_correct:
                found_confirmation = True

        if found_contradiction and not found_confirmation:
            return BindingCheckResult("contradicted",
                                       f"Subject {binding['subject_terms']} appears paired with the "
                                       f"competing value, not the correct one, in: \"{sentence}\"")
        if found_confirmation:
            return BindingCheckResult("confirmed", "Correct entity-value binding found.")
        return BindingCheckResult("neutral", "No clear binding signal either way; deferring to embedding score.")


class ContradictionCuePhrases:
    """
    cues = {
        "contradicting_phrases": [...],  # presence => likely inverted claim
        "confirming_phrases": [...],     # presence => likely correct claim
    }
    Narrower and more hand-authored than EntityBindingCheck - see module
    docstring. Intended for relationship-type rubric points (e.g. a
    tradeoff) where there is no simple two-entity swap to detect.
    """

    def check(self, sentence: str, cues: dict) -> BindingCheckResult:
        has_contradicting = _contains_any(sentence, cues.get("contradicting_phrases", []))
        has_confirming = _contains_any(sentence, cues.get("confirming_phrases", []))
        # Contradicting phrases win even if a generic confirming/topic word
        # (e.g. "tradeoff") is also present: a sentence can name the right
        # topic while still describing it backwards ("the tradeoff means X
        # and Y always happen together" uses the word "tradeoff" but
        # asserts the opposite of one). A specific wrong claim is stronger
        # evidence than a bare topic-word match, so it takes priority.
        if has_contradicting:
            return BindingCheckResult("contradicted",
                                       f"Contains a phrase suggesting the relationship is inverted: \"{sentence}\"")
        if has_confirming:
            return BindingCheckResult("confirmed", "Contains a phrase consistent with the correct relationship.")
        return BindingCheckResult("neutral", "No cue phrase matched; deferring to embedding score.")


def apply_fact_check(sentence: str, rubric_point: dict) -> Optional[BindingCheckResult]:
    """
    Dispatches to the right checker based on which metadata key is present
    on the rubric point (see generate_dataset.py for which points carry
    "binding" or "cues"). Returns None if the point has neither - i.e.
    most rubric points are unaffected and still scored by similarity alone.
    """
    if rubric_point.get("binding"):
        return EntityBindingCheck().check(sentence, rubric_point["binding"])
    if rubric_point.get("cues"):
        return ContradictionCuePhrases().check(sentence, rubric_point["cues"])
    return None


# ---------------------------------------------------------------------------
# Honest scope note (read before assuming this "solves" the limitation):
#
# EntityBindingCheck generalises well to any rubric point with a clear,
# nameable two-way entity<->attribute distinction (list/tuple, stack/queue,
# Type I/Type II error, etc.) - this covers a meaningful fraction of
# factual-inversion risk in this dataset's subjects.
#
# ContradictionCuePhrases does NOT generalise automatically - each rubric
# point needs its own hand-authored phrase list, which does not scale to
# an arbitrary question bank. It is included here as a working proof of
# concept for the relationship-inversion class of error (ML01), not as a
# general solution to that class.
#
# Neither technique catches inversions phrased in totally novel language
# that use neither the expected entity terms nor any of the authored cue
# phrases. The fully general solution remains an LLM/NLI-based fact-check
# component (Section 6's "Approach B"), noted as future work pending API
# access since Week 1.
# ---------------------------------------------------------------------------
