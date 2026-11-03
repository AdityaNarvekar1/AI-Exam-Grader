"""
build_sample_payloads.py
--------------------------
Group 2 - AI Exam Answer Grader
Writes examples/exam_key_sample.json, examples/group1_sample_output.json and
examples/demo_presets.json.

IMPORTANT: group1_sample_output.json is MOCK data. It follows the schema
Group 1's component is specified to emit (handout Section 5) but was written
by Group 2 - it is NOT output from Group 1's system, which was not available.
Each record is designed to exercise one integration case.

demo_presets.json feeds the demo UI. The "expect" text is a QUALITATIVE note for
the presenter, validated against the real models by tests/test_demo_presets.py;
the UI never displays a canned result - every click calls the real API.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "preprocessing"))
from generate_dataset import QUESTIONS, ANSWER_VARIANTS, EXTRA_TEST_CASES  # noqa: E402

Q = {q["qid"]: q for q in QUESTIONS}
# exam question number -> dataset question
EXAM_MAP = {"Q1": "DS01", "Q2": "PY01", "Q3": "ST02", "Q4": "ML01"}

exam_key = {}
for exam_q, dsq in EXAM_MAP.items():
    q = Q[dsq]
    exam_key[exam_q] = {"question": q["question"], "reference_answer": q["reference_answer"],
                        "rubric": q["rubric"], "maximum_score": q["max_marks"]}

txt = lambda dsq, tier: ANSWER_VARIANTS[dsq][tier][0]  # noqa: E731

records = [
    # clean, high-confidence, correct
    {"student_id": "23D001", "question": "Q1", "answer_text": txt("DS01", "excellent"), "page": 1, "confidence": 0.94},
    {"student_id": "23D001", "question": "Q2", "answer_text": txt("PY01", "good"), "page": 1, "confidence": 0.91},
    # OCR-style character errors, moderate confidence
    {"student_id": "23D002", "question": "Q1", "page": 1, "confidence": 0.88,
     "answer_text": "A stack works on Last-In-First-Out order, so the last elernent added is the first one "
                    "removed, like a stack of p1ates where you take from the top. A queue works on "
                    "First-In-First-Out order, so the first element added is the first one removed, like "
                    "people standing in a Iine waiting for a bus."},
    # one answer spanning two pages (merged; confidence = min = 0.74 -> low-OCR flag)
    {"student_id": "23D002", "question": "Q3", "page": 3, "confidence": 0.90,
     "answer_text": "A Type I error happens when we reject a null hypothesis that is actually true - "
                    "a false positive - and its probability is alpha."},
    {"student_id": "23D002", "question": "Q3", "page": 4, "confidence": 0.74,
     "answer_text": "A Type II error happens when we fail to reject a null hypothesis that is actually "
                    "false - a false negative - with probability beta."},
    # fluent but factually inverted (fact-check should fire)
    {"student_id": "23D003", "question": "Q1", "page": 1, "confidence": 0.95,
     "answer_text": "A stack follows a First-In-First-Out order, similar to people standing in a line "
                    "waiting for a bus, while a queue follows a Last-In-First-Out order, similar to a "
                    "stack of plates where you take the top one first."},
    # very low OCR confidence, garbled text
    {"student_id": "23D003", "question": "Q4", "page": 2, "confidence": 0.62,
     "answer_text": "Bias is the err0r from a modeI that is too simp1e which causes underfitting. "
                    "Variance is the error from a model too sensitive to the tra1ning data."},
    # empty OCR output (recognition failure vs genuinely blank is unknowable from the text alone)
    {"student_id": "23D004", "question": "Q2", "answer_text": "", "page": 1, "confidence": 0.30},
    # a question number that is not in the exam key
    {"student_id": "23D004", "question": "Q9", "answer_text": "Some answer to a question we have no key for.",
     "page": 5, "confidence": 0.92},
]

out_dir = ROOT / "examples"
out_dir.mkdir(exist_ok=True)
(out_dir / "exam_key_sample.json").write_text(json.dumps(exam_key, indent=2))
(out_dir / "group1_sample_output.json").write_text(json.dumps(records, indent=2))
print(f"wrote {len(exam_key)} exam-key entries and {len(records)} MOCK Group 1 records to {out_dir}")

# ---------------------------------------------------------------------------
# Demo presets for the UI (demo/index.html loads /examples/demo_presets.json).
# ---------------------------------------------------------------------------
extra = {(c["qid"], c["tier"]): c["student_answer"] for c in EXTRA_TEST_CASES}
_rows = list(csv.DictReader(open(ROOT / "data" / "exam_grading_dataset.csv", encoding="utf-8")))


def _reference_mark(dsq, answer):
    for r in _rows:
        if r["qid"] == dsq and r["student_answer"] == answer:
            return float(r["human_score"])
    return None


def preset(pid, label, dsq, answer, expect, approach="hybrid", failure=False, rubric=None, max_marks=None,
           reference_mark=None):
    q = Q[dsq]
    if reference_mark is None:
        reference_mark = _reference_mark(dsq, answer)
    return {"id": pid, "label": label, "is_known_failure": failure, "expect": expect,
            "reference_mark": reference_mark,
            "question": q["question"], "reference_answer": q["reference_answer"],
            "rubric": rubric or q["rubric"], "maximum_score": max_marks or q["max_marks"],
            "student_answer": answer, "approach": approach}


presets = [
    preset("correct", "Correct answer", "DS01", txt("DS01", "excellent"),
           "Both approaches give full marks and agree, so nothing is flagged."),
    preset("paraphrase", "Correct answer in different words (paraphrase)", "ST01", extra[("ST01", "paraphrase_trap")],
           "Compare component scores: the baseline cannot see a paraphrase (low), the embedding can (higher). "
           "The hybrid lands in between - averaging dilutes the embedding's correct credit - and is flagged "
           "because the approaches disagree. Switch Approach to 'embedding' to see the higher score."),
    preset("keyword_salad", "Keyword salad with no explanation", "DB02", extra[("DB02", "keyword_stuffing")],
           "Vocabulary alone should not earn marks. The embedding gives ~0, but the baseline rewards the "
           "vocabulary overlap, so the hybrid is mid-scale - and flagged because the approaches disagree."),
    preset("swapped", "Definitions swapped (LIFO <-> FIFO)", "DS01", extra[("DS01", "confidently_wrong")],
           "Fact-check veto: score 0 although the baseline alone would give about 4.5/5. Flagged for review; "
           "feedback names the factual error."),
    preset("fail_refusal", "KNOWN FAILURE: an honest refusal earns marks", "ST02", extra[("ST02", "refusal")],
           "Should score 0. The embedding credits it (exam-related words match the rubric), the baseline correctly "
           "gives 0, and the hybrid lands in between. Confidence is 0 so it is flagged - the safety net works, but "
           "the score is still wrong. This answer is part of the calibration data and is STILL credited.",
           failure=True),
    preset("fail_terse_rubric", "KNOWN FAILURE: terse rubric flips a correct answer to 0", "DS01",
           "A stack is last in first out.",
           "Approach is 'embedding'. A correct answer scores 0 because the terse rubric wording puts its "
           "similarity just under the credit boundary; rewording the rubric to a fuller sentence flips it. "
           "Fix: write rubric points as full sentences.",
           approach="embedding", failure=True, reference_mark=5.0,
           rubric=[{"id": "p1", "text": "A stack follows a last-in-first-out order", "marks": 5}], max_marks=5),
]
(out_dir / "demo_presets.json").write_text(json.dumps(presets, indent=2))
print(f"wrote {len(presets)} demo presets")
