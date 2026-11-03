"""
test_integration.py
-------------------
Group 2 - tests for POST /grade-digitized-answers (Group 1 -> Group 2).
Uses the MOCK Group 1 records in examples/ (written by Group 2 in Group 1's
published schema; NOT output from Group 1's real system).

    pytest tests/test_integration.py -v
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
from integration.group1_adapter import merge_records  # noqa: E402

EXAM_KEY = json.loads((ROOT / "examples" / "exam_key_sample.json").read_text())
RECORDS = json.loads((ROOT / "examples" / "group1_sample_output.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def post(client, answers, **extra):
    body = {"exam_id": "T", "exam_key": EXAM_KEY, "answers": answers}
    body.update(extra)
    return client.post("/grade-digitized-answers", json=body)


def by_key(resp):
    return {(r["student_id"], r["question"]): r for r in resp.json()["results"]}


def test_sample_exam_runs_and_summarises(client):
    r = post(client, RECORDS)
    assert r.status_code == 200
    s = r.json()["summary"]
    assert s["n_answers"] == 8          # 9 records, one answer spans two pages
    assert s["n_graded"] == 7 and s["n_skipped"] == 1


def test_multi_page_answer_is_merged_with_minimum_ocr_confidence(client):
    item = by_key(post(client, RECORDS))[("23D002", "Q3")]
    assert item["pages"] == [3, 4]
    assert item["ocr_confidence"] == 0.74
    assert "low_ocr_confidence" in item["review_reasons"]
    assert "Type I" in item["answer_text"] and "Type II" in item["answer_text"]


def test_pages_are_merged_in_page_order_even_if_records_arrive_shuffled():
    recs = [{"student_id": "s", "question": "Q1", "answer_text": "second", "page": 2, "confidence": 0.9},
            {"student_id": "s", "question": "Q1", "answer_text": "first", "page": 1, "confidence": 0.95}]
    m = merge_records(recs)[("s", "Q1")]
    assert m["answer_text"] == "first second" and m["ocr_confidence"] == 0.9


def test_swapped_definitions_are_scored_low_and_flagged_with_the_right_reason(client):
    item = by_key(post(client, RECORDS))[("23D003", "Q1")]
    assert item["grade"]["score"] == 0.0
    assert item["needs_human_review"] and "fact_check_contradiction" in item["review_reasons"]


def test_clean_correct_answer_is_not_flagged(client):
    item = by_key(post(client, RECORDS))[("23D001", "Q1")]
    assert item["grade"]["score"] == 5.0 and item["needs_human_review"] is False


def test_empty_ocr_text_is_flagged_as_possible_recognition_failure(client):
    item = by_key(post(client, RECORDS))[("23D004", "Q2")]
    assert "empty_ocr_text_possible_recognition_failure" in item["review_reasons"]


def test_unknown_question_is_skipped_not_crashed(client):
    item = by_key(post(client, RECORDS))[("23D004", "Q9")]
    assert item["status"] == "skipped_no_exam_key" and item["grade"] is None
    assert item["needs_human_review"] is True


def test_ocr_threshold_is_configurable(client):
    strict = by_key(post(client, RECORDS, ocr_review_threshold=0.99))
    lax = by_key(post(client, RECORDS, ocr_review_threshold=0.0))
    assert "low_ocr_confidence" in strict[("23D001", "Q1")]["review_reasons"]
    assert "low_ocr_confidence" not in lax[("23D003", "Q4")]["review_reasons"]


def test_invalid_exam_key_is_rejected(client):
    bad_key = {"Q1": {"question": "q", "reference_answer": "r", "maximum_score": 5,
                      "rubric": [{"id": "p1", "text": "t", "marks": 9}]}}
    r = client.post("/grade-digitized-answers", json={"exam_key": bad_key, "answers": RECORDS[:1]})
    assert r.status_code == 422


def test_ocr_confidence_out_of_range_is_rejected(client):
    bad = [{"student_id": "s", "question": "Q1", "answer_text": "x", "page": 1, "confidence": 1.7}]
    assert post(client, bad).status_code == 422


def test_demo_ui_is_served(client):
    r = client.get("/demo/")
    assert r.status_code == 200 and "AI Exam Answer Grader" in r.text
