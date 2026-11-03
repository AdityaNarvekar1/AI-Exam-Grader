"""
test_demo_presets.py
--------------------
Guards the demonstration against claiming things the system does not do.
The presenter notes in examples/demo_presets.json are QUALITATIVE; these
tests assert the qualitative claims, run against the real models, so a
recalibration that changes behaviour fails here instead of on stage.

    pytest tests/test_demo_presets.py -v
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "api"))
sys.path.insert(0, str(ROOT / "src"))
from main import app  # noqa: E402

PRESETS = {p["id"]: p for p in json.loads((ROOT / "examples" / "demo_presets.json").read_text())}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def run(client, pid):
    p = PRESETS[pid]
    body = {k: p[k] for k in ("question", "reference_answer", "rubric", "student_answer",
                              "maximum_score", "approach")}
    r = client.post("/evaluate-answer", json=body)
    assert r.status_code == 200
    return r.json()


def test_every_preset_is_a_valid_request(client):
    for pid in PRESETS:
        run(client, pid)


def test_correct_answer_full_marks_and_not_flagged(client):
    d = run(client, "correct")
    assert d["score"] == d["maximum_score"] and not d["needs_human_review"]


def test_paraphrase_baseline_low_embedding_higher_and_flagged(client):
    d = run(client, "paraphrase")
    assert d["component_scores"]["baseline"] < d["component_scores"]["embedding"]
    assert d["needs_human_review"]


def test_keyword_salad_embedding_near_zero_and_flagged(client):
    d = run(client, "keyword_salad")
    assert d["component_scores"]["embedding"] <= 1.0 and d["needs_human_review"]


def test_swapped_definitions_vetoed_and_flagged(client):
    d = run(client, "swapped")
    assert d["score"] == 0.0 and d["needs_human_review"]
    assert d["component_scores"]["baseline"] > 2.0   # baseline alone would reward it


def test_known_failure_refusal_is_credited_but_flagged(client):
    d = run(client, "fail_refusal")
    assert d["score"] > 0 and d["needs_human_review"]   # a wrong score, caught by the flag


def test_known_failure_terse_rubric_zero_for_a_correct_answer(client):
    d = run(client, "fail_terse_rubric")
    assert d["score"] == 0.0 and PRESETS["fail_terse_rubric"]["reference_mark"] == 5.0
