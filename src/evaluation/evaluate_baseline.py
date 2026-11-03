"""
evaluate_baseline.py
---------------------
Group 2 - AI Exam Answer Grader
Week-2 deliverable: "Working baseline + initial evaluation".

Loads data/exam_grading_dataset.csv, fits TFIDFBaselineGrader on the TRAIN
split, scores the TEST split, and reports the metrics mandated by Section 7
of the handout for the "Grading" task: MAE, RMSE, correlation with human
scores, and agreement within +/-1 mark - plus a qualitative error analysis
of the worst cases.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.baseline_grader import TFIDFBaselineGrader  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    grader = TFIDFBaselineGrader().fit(train_rows)
    predictions = grader.grade_batch(test_rows)

    y_true = np.array([float(r["human_score"]) for r in test_rows])
    y_pred = np.array([p.score for p in predictions])

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    corr, p_value = pearsonr(y_true, y_pred)
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))

    per_row_results = []
    for r, p in zip(test_rows, predictions):
        err = abs(float(r["human_score"]) - p.score)
        per_row_results.append({
            "qid": r["qid"], "subject": r["subject"], "student_id": r["student_id"],
            "quality_tier": r["quality_tier"], "max_marks": r["max_marks"],
            "human_score": float(r["human_score"]), "predicted_score": p.score,
            "abs_error": round(err, 2), "tfidf_similarity": p.tfidf_similarity,
            "keyword_overlap": p.keyword_overlap, "confidence": p.confidence,
            "feedback": p.feedback,
        })

    per_row_results.sort(key=lambda x: -x["abs_error"])

    metrics = {
        "n_train": len(train_rows), "n_test": len(test_rows),
        "MAE": round(mae, 3), "RMSE": round(rmse, 3),
        "pearson_r": round(float(corr), 3), "pearson_p_value": round(float(p_value), 4),
        "agreement_within_1_mark": round(within_1, 3),
    }

    with open(RESULTS_DIR / "week2_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS_DIR / "week2_baseline_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_row_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_row_results)

    print("=== Week 2 Baseline Evaluation (TF-IDF + keyword overlap, linear-calibrated) ===")
    print(f"Train examples: {metrics['n_train']}   Test examples: {metrics['n_test']}")
    print(f"MAE:  {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"Pearson correlation with human scores: {metrics['pearson_r']} (p={metrics['pearson_p_value']})")
    print(f"Agreement within +/-1 mark: {metrics['agreement_within_1_mark']*100:.1f}%")
    print()
    print("Top 5 largest errors (for qualitative error analysis):")
    for row in per_row_results[:5]:
        print(f"  [{row['qid']}/{row['student_id']}] tier={row['quality_tier']:<10} "
              f"human={row['human_score']:<5} pred={row['predicted_score']:<5} "
              f"abs_err={row['abs_error']:<5} sim={row['tfidf_similarity']} kw={row['keyword_overlap']}")


if __name__ == "__main__":
    main()
