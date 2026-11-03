"""
compute_human_agreement.py
-----------------------------
Group 2 - AI Exam Answer Grader
Run AFTER a team member has filled in grader_score in their copy of
week5_human_validation_BLIND_SHEET.csv (see build_human_validation_sheet.py).

Reports (Section 7 metrics: MAE, RMSE, Pearson r, agreement within +/-1 mark):
  - human vs. synthetic ground truth;
  - human vs. baseline model, human vs. embedding model;
  - if 2+ completed sheets are passed in, inter-rater agreement between graders.

Usage:
    python compute_human_agreement.py filled_sheet_grader1.csv [filled_sheet_grader2.csv ...]
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
ANSWER_KEY_PATH = RESULTS_DIR / "week5_human_validation_ANSWER_KEY.csv"
BASELINE_PRED_PATH = RESULTS_DIR / "week2_baseline_predictions.csv"
EMBEDDING_COMPARISON_PATH = RESULTS_DIR / "week4_comparison_rows.csv"


def score_metrics(y_true, y_pred, label):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r, p = pearsonr(y_true, y_pred) if len(set(y_true)) > 1 else (float("nan"), float("nan"))
    within1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    print(f"{label:<32} MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:.3f}  agreement(+/-1)={within1*100:.1f}%  n={len(y_true)}")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if not ANSWER_KEY_PATH.exists():
        print(f"Missing {ANSWER_KEY_PATH} - run build_human_validation_sheet.py first.")
        sys.exit(1)

    answer_key = {(r["qid"], r["student_id"]): r for r in load_csv(ANSWER_KEY_PATH)}
    baseline_preds = {(r["qid"], r["student_id"]): float(r["predicted_score"])
                      for r in load_csv(BASELINE_PRED_PATH)} if BASELINE_PRED_PATH.exists() else {}
    embedding_preds = {(r["qid"], r["student_id"]): float(r["improved_pred"])
                       for r in load_csv(EMBEDDING_COMPARISON_PATH)} if EMBEDDING_COMPARISON_PATH.exists() else {}

    grader_sheets = {}
    for path in sys.argv[1:]:
        rows = load_csv(path)
        scored = {(r["qid"], r["student_id"]): float(r["grader_score"])
                  for r in rows if r.get("grader_score", "").strip() != ""}
        if not scored:
            print(f"WARNING: {path} has no filled-in grader_score values - skipping.")
            continue
        grader_sheets[Path(path).stem] = scored
        print(f"Loaded {len(scored)} graded rows from {Path(path).stem}")
    if not grader_sheets:
        print("No completed grading sheets found. Fill in grader_score and retry.")
        sys.exit(1)

    print()
    for name, scored in grader_sheets.items():
        keys = [k for k in scored if k in answer_key]
        human = [scored[k] for k in keys]
        print(f"--- {name} (n={len(keys)}) ---")
        score_metrics(human, [float(answer_key[k]["synthetic_ground_truth_score"]) for k in keys], "Human vs. synthetic label")
        ks = [k for k in keys if k in baseline_preds]
        if ks:
            score_metrics([scored[k] for k in ks], [baseline_preds[k] for k in ks], "Human vs. baseline model")
        ks = [k for k in keys if k in embedding_preds]
        if ks:
            score_metrics([scored[k] for k in ks], [embedding_preds[k] for k in ks], "Human vs. embedding model")
        print()

    names = list(grader_sheets)
    if len(names) >= 2:
        print("--- Inter-rater agreement between graders ---")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = grader_sheets[names[i]], grader_sheets[names[j]]
                common = [k for k in a if k in b]
                if common:
                    score_metrics([a[k] for k in common], [b[k] for k in common], f"{names[i]} vs {names[j]}")


if __name__ == "__main__":
    main()
