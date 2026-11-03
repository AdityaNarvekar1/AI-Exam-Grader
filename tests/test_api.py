"""
test_api.py
-----------
Group 2 - AI Exam Answer Grader
Automated tests for POST /evaluate-answer. Run from the project root:

    pytest tests/test_api.py -v

Starting the app loads spaCy and fits both models (~10-20 s once per test
session). These tests check the API CONTRACT and known behaviours
(including known limitations, stated as such) - they do not re-validate
grading accuracy; that is what the src/evaluation/ scripts are for.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))
from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (loads models)
        yield c


STACK_QUEUE_RUBRIC = [
    {"id": "p1", "text": "A stack follows a last-in-first-out order, and a correct real-world example is given",
     "marks": 2.5,
     "binding": {"subject_terms": ["stack"], "correct_value_terms": ["last-in-first-out", "lifo"],
                 "contradicting_value_terms": ["first-in-first-out", "fifo"]}},
    {"id": "p2", "text": "A queue follows a first-in-first-out order, and a correct real-world example is given",
     "marks": 2.5,
     "binding": {"subject_terms": ["queue"], "correct_value_terms": ["first-in-first-out", "fifo"],
                 "contradicting_value_terms": ["last-in-first-out", "lifo"]}},
]


def payload(student_answer, rubric=None, maximum_score=5, **extra):
    body = {"question": "Explain the difference between a stack and a queue.",
            "reference_answer": "A stack is LIFO; a queue is FIFO.",
            "rubric": STACK_QUEUE_RUBRIC if rubric is None else rubric,
            "student_answer": student_answer, "maximum_score": maximum_score}
    body.update(extra)
    return body


GOOD = ("A stack follows Last-In-First-Out order, like a stack of plates, while a queue "
        "follows First-In-First-Out order, like people waiting in a line.")
INVERTED = ("A stack follows a First-In-First-Out order, like people in a line, while a queue "
            "follows a Last-In-First-Out order, like a stack of plates.")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["approaches_available"]) == {"hybrid", "baseline", "embedding"}


def test_response_contains_the_handout_contract_fields(client):
    """Section 5's example output: question, score, maximum_score, feedback, confidence."""
    d = client.post("/evaluate-answer", json=payload(GOOD)).json()
    for field in ("question", "score", "maximum_score", "feedback", "confidence"):
        assert field in d
    assert 0 <= d["score"] <= d["maximum_score"]
    assert 0 <= d["confidence"] <= 1


def test_default_approach_is_hybrid(client):
    d = client.post("/evaluate-answer", json=payload(GOOD)).json()
    assert d["approach_used"] == "hybrid"
    assert set(d["component_scores"]) == {"baseline", "embedding"}


@pytest.mark.parametrize("approach", ["hybrid", "embedding", "baseline"])
def test_each_approach_runs(client, approach):
    r = client.post("/evaluate-answer", json=payload(GOOD, approach=approach))
    assert r.status_code == 200
    assert r.json()["approach_used"] == approach


def test_fact_check_veto_catches_inverted_definitions(client):
    """The Week 5 DS01 failure: swapped LIFO/FIFO used to score full marks."""
    d = client.post("/evaluate-answer", json=payload(INVERTED, approach="embedding")).json()
    assert d["score"] == 0.0
    assert all(p["fact_check"] == "contradicted" for p in d["point_breakdown"])


def test_inverted_answer_is_flagged_for_human_review_in_hybrid(client):
    d = client.post("/evaluate-answer", json=payload(INVERTED)).json()
    assert d["needs_human_review"] is True


def test_hybrid_defers_to_the_fact_check_instead_of_averaging_it_away(client):
    """Regression test: averaging with the (factual-error-blind) baseline once gave 2.5/5 here."""
    d = client.post("/evaluate-answer", json=payload(INVERTED)).json()
    assert d["score"] == min(d["component_scores"].values()) == 0.0


def test_rubric_marks_exceeding_maximum_score_is_rejected(client):
    """Regression test for a real bug: this used to silently return HTTP 200."""
    bad = payload("x", rubric=[{"id": "p1", "text": "something", "marks": 8}], maximum_score=5)
    assert client.post("/evaluate-answer", json=bad).status_code == 422


def test_missing_required_field_is_rejected(client):
    body = payload(GOOD)
    del body["student_answer"]
    assert client.post("/evaluate-answer", json=body).status_code == 422


def test_empty_rubric_is_rejected(client):
    assert client.post("/evaluate-answer", json=payload(GOOD, rubric=[])).status_code == 422


def test_unknown_approach_is_rejected(client):
    assert client.post("/evaluate-answer", json=payload(GOOD, approach="magic")).status_code == 422


def test_empty_student_answer_does_not_crash_and_scores_zero(client):
    """Also guards the leaked -1.0 similarity sentinel."""
    d = client.post("/evaluate-answer", json=payload("", approach="embedding")).json()
    assert d["score"] == 0.0
    assert all(p["similarity"] >= 0 for p in d["point_breakdown"])


def test_off_topic_answer_scores_low(client):
    d = client.post("/evaluate-answer", json=payload("Our cricket team won the final last week.")).json()
    assert d["score"] <= 1.0


def test_batch_endpoint(client):
    r = client.post("/evaluate-answer/batch", json=[payload(GOOD), payload(INVERTED)])
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_size_limit(client):
    r = client.post("/evaluate-answer/batch", json=[payload(GOOD)] * 201)
    assert r.status_code == 400


def test_known_limitation_rubric_phrasing_sensitivity(client):
    """
    DOCUMENTED LIMITATION, asserted so it can't regress silently into being
    ignored: with a terse rubric and approach='embedding', a correct answer
    can fall below the credit threshold. This test pins the *current*
    behaviour so a future fix is noticed (the assertion will then need
    updating - that is the point). See README 'Known limitations'.
    """
    terse = [{"id": "p1", "text": "A stack follows a last-in-first-out order", "marks": 5}]
    d = client.post("/evaluate-answer", json=payload("A stack is last in first out.", rubric=terse,
                                                      approach="embedding")).json()
    assert d["score"] in (0.0, 5.0)  # all-or-nothing on a single 5-mark point
