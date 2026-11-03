"""Regression tests for the v2 fact-binding checker (run: pytest tests/test_fact_binding.py -v)."""
import csv, json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
from fact_binding_checker import (EntityBindingCheck, ContradictionCuePhrases, apply_fact_check,
                                  apply_fact_check_text)  # noqa: E402

LIST_B = {"subject_terms": ["list"], "correct_value_terms": ["mutable"], "contradicting_value_terms": ["immutable"]}
STACK_B = {"subject_terms": ["stack"], "correct_value_terms": ["lifo", "last in first out"],
           "contradicting_value_terms": ["fifo", "first in first out"]}
QUEUE_B = {"subject_terms": ["queue"], "correct_value_terms": ["fifo", "first in first out"],
           "contradicting_value_terms": ["lifo", "last in first out"]}
T1_B = {"subject_terms": ["type i", "type 1"], "correct_value_terms": ["false positive"],
        "contradicting_value_terms": ["false negative"]}
T2_B = {"subject_terms": ["type ii", "type 2"], "correct_value_terms": ["false negative"],
        "contradicting_value_terms": ["false positive"]}
chk = EntityBindingCheck()


def status(sentence, binding):
    return chk.check(sentence, binding).status


# ---- the substring bug the audit found -------------------------------------------------
def test_immutable_is_not_mistaken_for_mutable():
    assert status("A list is immutable while a tuple is mutable.", LIST_B) == "contradicted"

def test_correct_mutable_still_confirmed():
    assert status("A list is mutable, while a tuple is immutable.", LIST_B) == "confirmed"
    assert status("Lists are mutable so you can add or remove items.", LIST_B) == "confirmed"

def test_type_i_does_not_match_inside_type_ii():
    # the Type II sentence alone must not be judged against the Type I rubric point
    assert status("A Type II error is a false negative.", T1_B) == "neutral"
    assert status("A Type II error is a false negative.", T2_B) == "confirmed"

# ---- negation -------------------------------------------------------------------------
def test_negated_correct_value_is_wrong():
    assert status("A list is not mutable.", LIST_B) == "contradicted"
    assert status("A list is never mutable.", LIST_B) == "contradicted"

def test_negated_wrong_value_is_correct():
    assert status("A list is not immutable.", LIST_B) == "confirmed"

# ---- proximity binding: swapped values with no comma ---------------------------------
def test_swapped_without_comma_is_caught():
    assert status("A stack is FIFO and a queue is LIFO.", STACK_B) == "contradicted"
    assert status("A stack is FIFO and a queue is LIFO.", QUEUE_B) == "contradicted"

def test_correct_without_comma_is_confirmed():
    assert status("A stack is LIFO and a queue is FIFO.", STACK_B) == "confirmed"
    assert status("A stack is LIFO and a queue is FIFO.", QUEUE_B) == "confirmed"

def test_hyphenated_and_spaced_forms_equivalent():
    assert status("A stack follows Last-In-First-Out order.", STACK_B) == "confirmed"
    assert status("A stack follows first in first out order.", STACK_B) == "contradicted"

def test_dataset_ds01_swapped_answer_vetoed():
    a = ("A stack follows a First-In-First-Out order, similar to people standing in a line waiting for "
         "a bus, while a queue follows a Last-In-First-Out order, similar to a stack of plates where you "
         "take the top one first.")
    assert status(a, STACK_B) == "contradicted" and status(a, QUEUE_B) == "contradicted"

def test_long_definition_sentence_binds_across_distance():
    s = "A Type I error happens when we reject a null hypothesis that is actually true - a false positive - and its probability is alpha."
    assert status(s, T1_B) == "confirmed"
    s2 = "Type I error happens when we fail to reject a null hypothesis that is actually false - a false negative."
    assert status(s2, T1_B) == "contradicted"

# ---- self-contradiction and whole-answer scope ---------------------------------------
def test_self_contradiction_is_contradicted():
    assert status("A list is mutable and also a list is immutable.", LIST_B) == "contradicted"

def test_contradiction_outside_best_sentence_is_caught_with_answer_scope():
    ans = "A list is mutable. Later I realised that a list is immutable once created."
    pt = {"id": "p1", "text": "x", "marks": 1, "binding": LIST_B}
    assert apply_fact_check_text(ans, pt).status == "contradicted"

def test_no_metadata_returns_none():
    assert apply_fact_check("anything", {"id": "p1", "text": "x", "marks": 1}) is None
    assert apply_fact_check_text("anything", {"id": "p1", "text": "x", "marks": 1}) is None

# ---- cue phrases ---------------------------------------------------------------------
CUES = {"confirming_phrases": ["tradeoff", "inversely"],
        "contradicting_phrases": ["always happen together", "both increase"]}

def test_cue_contradiction_wins_over_topic_word():
    r = ContradictionCuePhrases().check("The tradeoff means bias and variance always happen together.", CUES)
    assert r.status == "contradicted"

def test_negated_contradicting_cue_is_not_a_contradiction():
    r = ContradictionCuePhrases().check("Bias and variance do not always happen together; it is a tradeoff.", CUES)
    assert r.status == "confirmed"


# ---- dataset-wide: the veto must never fire on an answer that truly covers the point ---
def _rows():
    with open(ROOT / "data" / "exam_grading_dataset.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def test_no_false_veto_on_any_covered_point_in_the_dataset():
    bad = []
    for r in _rows():
        covered = set(json.loads(r["points_covered"]))
        for p in json.loads(r["rubric"]):
            if p["id"] in covered and (p.get("binding") or p.get("cues")):
                res = apply_fact_check_text(r["student_answer"], p)
                if res.status == "contradicted":
                    bad.append((r["qid"], r["quality_tier"], p["id"], res.reason[:80]))
    assert not bad, bad

def test_known_adversarial_rows_are_still_vetoed():
    hits = set()
    for r in _rows():
        if r["quality_tier"] in ("confidently_wrong", "mixed_with_error"):
            for p in json.loads(r["rubric"]):
                res = apply_fact_check_text(r["student_answer"], p)
                if res is not None and res.status == "contradicted":
                    hits.add((r["qid"], p["id"]))
    assert ("DS01", "p1") in hits and ("DS01", "p2") in hits and ("ML01", "p3") in hits


def test_value_after_a_comma_still_binds_to_its_subject():
    s = "A Type I error is failing to reject a false null hypothesis, which is a false negative with probability beta."
    assert status(s, T1_B) == "contradicted"

def test_comma_separated_correct_pair_is_confirmed():
    s = "A stack is LIFO, last in first out, and a queue is FIFO, first in first out."
    assert status(s, STACK_B) == "confirmed" and status(s, QUEUE_B) == "confirmed"
