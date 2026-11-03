"""
group1_adapter.py
------------------
Group 2 - AI Exam Answer Grader  <-  Group 1 - Handwritten Examination Digitizer
Week-7 integration (handout Section 22: "Handwritten answer digitization
-> automated grading").

Group 1's published output schema (handout Section 5) is one record per
answer segment:

    {"student_id": "23D001", "question": "Q3", "answer_text": "...",
     "page": 4, "confidence": 0.91}

Three interface facts this adapter has to handle (found by reading that
schema against what Group 2 needs):

  1. Group 1 gives a question NUMBER ("Q3"), not the question text or the
     rubric. Group 2 cannot grade without them, so the caller supplies an
     EXAM KEY mapping "Q3" -> {question, reference_answer, rubric,
     maximum_score}. Neither group produces this today; it is the one new
     artefact the two components need from faculty.
  2. An answer can span pages, and the schema carries `page`, so one
     (student, question) may arrive as several records. They are merged in
     page order; the merged OCR confidence is the MINIMUM of the parts.
  3. Group 1's `confidence` is OCR confidence; Group 2's is grading
     agreement. They measure different things, so they are reported
     separately and either one low sends the answer to a human. (Week 7
     robustness check: under simulated OCR noise the grader's own review
     flag does NOT rise, so OCR confidence cannot be replaced by it.)

WEEK 7 UPDATE - Group 1's REAL output (data/group1_real/ocr_output_5_students.csv) differs from the
published schema, so records are now normalised before merging:
  * the text field is `ocr_answer`, not `answer_text`            (both accepted)
  * `status` = found | not_found; not_found has an empty confidence  -> empty text, confidence 0,
    reason "ocr_answer_not_found"
  * `question_type` = mcq | short. MCQ records are graded here by letter match against the exam key's
    `correct_option`; they never reach the NLP grader.
  * Group 1's own `needs_review` = yes                           -> reason "group1_needs_review"
  * `placed_by_content` = yes (Group 1 assigned the text to a question by matching its content, not by
    the printed question number, so it may sit under the wrong question) -> reason
    "group1_placed_by_content"
The original published schema still works unchanged (examples/ and tests/test_integration.py).
"""

from collections import OrderedDict
from typing import Callable, Dict, List

# Change here if Group 1's real output uses different field names.
FIELD_MAP = {
    "student_id": "student_id",
    "question": "question",
    "answer_text": "answer_text",
    "page": "page",
    "confidence": "confidence",
}
# alternative field names seen in Group 1's real output
ALIASES = {"answer_text": ["ocr_answer"]}

DEFAULT_OCR_REVIEW_THRESHOLD = 0.8
YES = {"yes", "true", "1", "y"}


def _get(record: dict, canonical: str, default=None):
    v = record.get(FIELD_MAP[canonical])
    if v is None:
        for alt in ALIASES.get(canonical, []):
            if record.get(alt) is not None:
                return record[alt]
        return default
    return v


def _num(v, default):
    try:
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def normalise_record(rec: dict) -> dict:
    """Map either schema (published or real) onto one internal record."""
    status = str(rec.get("status") or "found").strip().lower()
    text = (_get(rec, "answer_text", "") or "").strip()
    conf = _num(_get(rec, "confidence"), None)
    if conf is None:
        conf = 0.0 if status == "not_found" else 1.0
    page = _get(rec, "page")
    page = int(_num(page, 0)) if page not in (None, "") else None
    return {
        "student_id": str(_get(rec, "student_id")), "question": str(_get(rec, "question")),
        "answer_text": "" if status == "not_found" else text, "page": page, "confidence": conf,
        "status": status, "question_type": str(rec.get("question_type") or "").strip().lower() or None,
        "group1_needs_review": str(rec.get("needs_review") or "").strip().lower() in YES,
        "placed_by_content": str(rec.get("placed_by_content") or "").strip().lower() in YES,
    }


def load_group1_csv(path) -> List[dict]:
    """Read Group 1's real CSV export into a list of raw records (their file starts with a UTF-8 BOM)."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def grade_mcq(spec: dict, answer_text: str) -> dict:
    """Exact letter match against spec['correct_option'].  Returns a grade dict like the NLP grader's."""
    got = "".join(ch for ch in (answer_text or "").upper() if ch.isalpha())[:1]
    want = str(spec.get("correct_option", "")).strip().upper()[:1]
    mx = float(spec.get("maximum_score", 1))
    ok = bool(got) and got == want
    return {"question": spec.get("question", ""), "score": mx if ok else 0.0, "maximum_score": mx,
            "feedback": f"Selected {got or 'nothing'}; correct option is {want}.",
            "confidence": 1.0, "needs_human_review": False, "approach_used": "mcq_exact_match",
            "selected_option": got, "correct_option": want}


def merge_records(records: List[dict]) -> "OrderedDict[tuple, dict]":
    """Merge multi-page answers: same (student, question) -> one answer."""
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for rec in (normalise_record(r) for r in records):
        key = (str(_get(rec, "student_id")), str(_get(rec, "question")))
        groups.setdefault(key, []).append(rec)

    merged = OrderedDict()
    for key, recs in groups.items():
        recs = sorted(recs, key=lambda r: (r["page"] or 0))
        merged[key] = {
            "student_id": key[0],
            "question": key[1],
            "answer_text": " ".join(r["answer_text"] for r in recs if r["answer_text"]).strip(),
            "pages": [r["page"] for r in recs],
            "ocr_confidence": round(min(float(r["confidence"]) for r in recs), 3),
            "ocr_status": "not_found" if all(r["status"] == "not_found" for r in recs) else "found",
            "question_type": next((r["question_type"] for r in recs if r["question_type"]), None),
            "group1_needs_review": any(r["group1_needs_review"] for r in recs),
            "placed_by_content": any(r["placed_by_content"] for r in recs),
        }
    return merged


def grade_digitized(records: List[dict], exam_key: Dict[str, dict],
                    grade_fn: Callable[[dict, str], dict],
                    ocr_review_threshold: float = DEFAULT_OCR_REVIEW_THRESHOLD) -> dict:
    """
    grade_fn(question_spec, answer_text) -> dict with at least
        score, maximum_score, feedback, confidence, needs_human_review
    (the API passes a function wrapping the selected approach).
    """
    results = []
    for (student_id, qid), ans in merge_records(records).items():
        spec = exam_key.get(qid)
        item = {
            "student_id": student_id, "question": qid, "pages": ans["pages"],
            "ocr_confidence": ans["ocr_confidence"], "answer_text": ans["answer_text"],
            "status": "graded", "grade": None, "review_reasons": [],
        }
        if spec is None:
            item["status"] = "skipped_no_exam_key"
            item["review_reasons"].append("no_exam_key_for_question")
        elif (spec.get("question_type") or ans.get("question_type")) == "mcq":
            item["grade"] = grade_mcq(spec, ans["answer_text"])
        else:
            grade = grade_fn(spec, ans["answer_text"])
            item["grade"] = grade
            contradicted = any(pt.get("fact_check") == "contradicted"
                               for pt in (grade.get("point_breakdown") or []))
            if contradicted:
                item["review_reasons"].append("fact_check_contradiction")
            if grade.get("needs_human_review") and not contradicted:
                item["review_reasons"].append("low_grader_agreement")
        if ans["ocr_status"] == "not_found":
            item["review_reasons"].append("ocr_answer_not_found")
        elif ans["ocr_confidence"] < ocr_review_threshold:
            item["review_reasons"].append("low_ocr_confidence")
        if not ans["answer_text"] and ans["ocr_status"] != "not_found":
            item["review_reasons"].append("empty_ocr_text_possible_recognition_failure")
        if ans["group1_needs_review"] and ans["ocr_status"] != "not_found":
            item["review_reasons"].append("group1_needs_review")
        if ans["placed_by_content"]:
            item["review_reasons"].append("group1_placed_by_content")
        item["needs_human_review"] = bool(item["review_reasons"])
        results.append(item)

    return {
        "results": results,
        "summary": {
            "n_answers": len(results),
            "n_mcq": sum(bool(r["grade"]) and r["grade"].get("approach_used") == "mcq_exact_match" for r in results),
            "n_graded": sum(r["status"] == "graded" for r in results),
            "n_skipped": sum(r["status"] != "graded" for r in results),
            "n_needs_human_review": sum(r["needs_human_review"] for r in results),
        },
    }
