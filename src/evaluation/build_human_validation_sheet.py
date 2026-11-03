"""
build_human_validation_sheet.py
----------------------------------
Group 2 - AI Exam Answer Grader
Week-5 deliverable: a small human-graded validation subset. This script does NOT
fabricate human judgement - it produces two files for the team to actually use:

  1. week5_human_validation_BLIND_SHEET.csv
     question, rubric, student answer, max marks, and BLANK columns for a human
     grader. Deliberately excludes the synthetic ground-truth score and both
     models' predictions, to avoid anchoring bias.
  2. week5_human_validation_ANSWER_KEY.csv
     the same rows WITH the synthetic ground-truth score, for comparison AFTER grading.

Protocol: each grader independently fills in grader_score (and optional
grader_notes) in their own copy of the BLIND SHEET without looking at the key;
then run compute_human_agreement.py on the completed sheet(s).
"""

import csv
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rubric_as_text(rubric_json):
    return " | ".join(f"({p['marks']} pts) {p['text']}" for p in json.loads(rubric_json))


def main():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r["split"] == "test"]

    blind_fields = ["qid", "student_id", "subject", "question", "reference_answer",
                    "rubric_points", "max_marks", "student_answer", "grader_score", "grader_notes"]
    blind_rows = [{"qid": r["qid"], "student_id": r["student_id"], "subject": r["subject"],
                   "question": r["question"], "reference_answer": r["reference_answer"],
                   "rubric_points": rubric_as_text(r["rubric"]), "max_marks": r["max_marks"],
                   "student_answer": r["student_answer"], "grader_score": "", "grader_notes": ""}
                  for r in test_rows]
    with open(RESULTS_DIR / "week5_human_validation_BLIND_SHEET.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=blind_fields); w.writeheader(); w.writerows(blind_rows)

    key_fields = blind_fields[:-2] + ["synthetic_ground_truth_score", "quality_tier"]
    key_rows = [{"qid": r["qid"], "student_id": r["student_id"], "subject": r["subject"],
                 "question": r["question"], "reference_answer": r["reference_answer"],
                 "rubric_points": rubric_as_text(r["rubric"]), "max_marks": r["max_marks"],
                 "student_answer": r["student_answer"], "synthetic_ground_truth_score": r["human_score"],
                 "quality_tier": r["quality_tier"]} for r in test_rows]
    with open(RESULTS_DIR / "week5_human_validation_ANSWER_KEY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_fields); w.writeheader(); w.writerows(key_rows)

    print(f"Blind sheet: {len(blind_rows)} rows -> results/week5_human_validation_BLIND_SHEET.csv")
    print(f"Answer key:  {len(key_rows)} rows -> results/week5_human_validation_ANSWER_KEY.csv")
    print("\nNext: have each team member fill in grader_score in their own copy of the BLIND SHEET,")
    print("then run compute_human_agreement.py on the completed sheet(s).")


if __name__ == "__main__":
    main()
