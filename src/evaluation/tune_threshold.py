"""
tune_threshold.py
-------------------
Group 2 - AI Exam Answer Grader
Week-5 deliverable: both the Week 3 and Week 4 reports promised tuning the
embedding grader's coverage-probability decision threshold on a dedicated
validation split instead of assuming a fixed 0.5 cutoff. This script does
exactly that.

Protocol (no test-set leakage):
  1. Fit EmbeddingSemanticGrader on the TRAIN split only.
  2. Sweep candidate thresholds (0.30 to 0.70) and score each one's
     point-level F1 on the VALIDATION split (never used to fit).
  3. Pick the threshold that maximises validation F1.
  4. Report point-level metrics AND end-to-end score metrics (MAE/RMSE/r/
     agreement) on the held-out TEST split at both the old fixed threshold
     (0.5) and the new tuned threshold, so the improvement is measured on
     data the tuning process never saw.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.embedding_grader import EmbeddingSemanticGrader, get_backend  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.30, 0.71, 0.02)]


def load_rows():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def point_level_pairs(rows, grader):
    """Returns (similarities, ground_truth_labels) for every (rubric point, row) pair."""
    sims, labels = [], []
    for r in rows:
        rubric = json.loads(r["rubric"])
        covered_gt = set(json.loads(r["points_covered"]))
        for p in rubric:
            _, sim = grader._point_best_match(p["text"], r["student_answer"])
            sims.append(sim)
            labels.append(1 if p["id"] in covered_gt else 0)
    return np.array(sims), np.array(labels)


def f1_at_threshold(sims, labels, grader, threshold):
    probs = grader.clf.predict_proba(sims.reshape(-1, 1))[:, 1]
    preds = (probs >= threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return f1


def score_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r, p = pearsonr(y_true, y_pred) if len(set(y_true)) > 1 else (float("nan"), float("nan"))
    within1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "pearson_r": round(float(r), 3) if r == r else None,
            "agreement_within_1_mark": round(within1, 3)}


def main():
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "spacy"
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train_rows)}  validation={len(val_rows)}  test={len(test_rows)}\n")

    grader = EmbeddingSemanticGrader(backend=get_backend(backend_name)).fit(train_rows)
    val_sims, val_labels = point_level_pairs(val_rows, grader)

    sweep_results = []
    for t in THRESHOLD_GRID:
        f1 = f1_at_threshold(val_sims, val_labels, grader, t)
        sweep_results.append({"threshold": t, "validation_f1": round(float(f1), 3)})

    best = max(sweep_results, key=lambda d: d["validation_f1"])
    best_threshold = best["threshold"]

    print("Threshold sweep on validation set:")
    for s in sweep_results:
        marker = "  <-- best" if s["threshold"] == best_threshold else ""
        print(f"  {s['threshold']:.2f}  F1={s['validation_f1']:.3f}{marker}")
    print(f"\nBest threshold on validation: {best_threshold}\n")

    def evaluate_on_test(threshold):
        g = EmbeddingSemanticGrader(backend=grader.backend, threshold=threshold)
        g.clf = grader.clf
        g._fitted = True
        results = g.grade_batch(test_rows)
        y_true = [float(r["human_score"]) for r in test_rows]
        y_pred = [res.score for res in results]
        sims, labels = point_level_pairs(test_rows, g)
        probs = g.clf.predict_proba(sims.reshape(-1, 1))[:, 1]
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0)
        return {"threshold": threshold,
                "point_level": {"precision": round(float(precision), 3),
                                 "recall": round(float(recall), 3), "f1": round(float(f1), 3)},
                "score_level": score_metrics(y_true, y_pred)}

    before = evaluate_on_test(0.5)
    after = evaluate_on_test(best_threshold)

    print("=== Test-set comparison: fixed 0.5 vs. validation-tuned threshold ===")
    print(f"{'Metric':<28}{'threshold=0.5':<18}{'threshold=' + str(best_threshold):<18}")
    print(f"{'Point F1':<28}{before['point_level']['f1']:<18}{after['point_level']['f1']:<18}")
    print(f"{'Score MAE':<28}{before['score_level']['MAE']:<18}{after['score_level']['MAE']:<18}")
    print(f"{'Score RMSE':<28}{before['score_level']['RMSE']:<18}{after['score_level']['RMSE']:<18}")
    print(f"{'Pearson r':<28}{before['score_level']['pearson_r']:<18}{after['score_level']['pearson_r']:<18}")
    print(f"{'Agreement +/-1':<28}{before['score_level']['agreement_within_1_mark']:<18}{after['score_level']['agreement_within_1_mark']:<18}")

    output = {"backend": backend_name, "threshold_sweep": sweep_results, "best_threshold": best_threshold,
              "test_at_threshold_0.5": before, "test_at_tuned_threshold": after}
    suffix = "" if backend_name == "spacy" else f"_{backend_name}"
    with open(RESULTS_DIR / f"week5_threshold_tuning{suffix}.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
