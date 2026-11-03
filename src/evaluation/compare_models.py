"""
compare_models.py
------------------
Group 2 - AI Exam Answer Grader
Week-4 deliverable: "Baseline vs improved model comparison" (Section 6,
Requirement 2 - every group must implement a Baseline + an Improved
Approach and compare them; Section 7 - report quantitative Grading metrics:
MAE, RMSE, correlation, agreement with human graders).

Baseline  = Week 2: TF-IDF cosine similarity + keyword overlap (TFIDFBaselineGrader)
Improved  = Week 3: pretrained-embedding concept-level semantic matching (EmbeddingSemanticGrader)

Both are fit on the identical TRAIN split and scored on the identical,
held-out TEST split (including the 5 adversarial paraphrase/keyword-
stuffing cases), so the comparison is apples-to-apples.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.baseline_grader import TFIDFBaselineGrader        # noqa: E402
from models.embedding_grader import EmbeddingSemanticGrader, get_backend   # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADVERSARIAL_TIERS = {"paraphrase_trap", "keyword_stuffing"}


def load_rows():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        r, p = pearsonr(y_true, y_pred)
    else:
        r, p = float("nan"), float("nan")
    within1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "pearson_r": round(float(r), 3) if r == r else None,
            "agreement_within_1_mark": round(within1, 3), "n": len(y_true)}


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    y_true = [float(r["human_score"]) for r in test_rows]

    scoring_mode = sys.argv[1] if len(sys.argv) > 1 else "hard"
    backend_name = sys.argv[2] if len(sys.argv) > 2 else "spacy"
    print(f"(scoring_mode={scoring_mode}, embedding_backend={backend_name})\n")

    baseline = TFIDFBaselineGrader().fit(train_rows)
    improved = EmbeddingSemanticGrader(
        backend=get_backend(backend_name), scoring_mode=scoring_mode
    ).fit(train_rows)

    baseline_preds = baseline.grade_batch(test_rows)
    improved_preds = improved.grade_batch(test_rows)

    y_pred_baseline = [p.score for p in baseline_preds]
    y_pred_improved = [p.score for p in improved_preds]

    overall = {
        "baseline_tfidf": score_metrics(y_true, y_pred_baseline),
        "improved_embedding": score_metrics(y_true, y_pred_improved),
    }

    def subset_metrics(model_preds):
        normal_t, normal_p, adv_t, adv_p = [], [], [], []
        for r, pred in zip(test_rows, model_preds):
            if r["quality_tier"] in ADVERSARIAL_TIERS:
                adv_t.append(float(r["human_score"])); adv_p.append(pred)
            else:
                normal_t.append(float(r["human_score"])); normal_p.append(pred)
        return {"normal": score_metrics(normal_t, normal_p),
                "adversarial": score_metrics(adv_t, adv_p)}

    by_subset = {
        "baseline_tfidf": subset_metrics(y_pred_baseline),
        "improved_embedding": subset_metrics(y_pred_improved),
    }

    # Row-by-row comparison table
    comparison_rows = []
    for r, bp, ip in zip(test_rows, baseline_preds, improved_preds):
        comparison_rows.append({
            "qid": r["qid"], "student_id": r["student_id"], "quality_tier": r["quality_tier"],
            "human_score": float(r["human_score"]), "max_marks": r["max_marks"],
            "baseline_pred": bp.score, "baseline_abs_err": round(abs(float(r["human_score"]) - bp.score), 2),
            "improved_pred": ip.score, "improved_abs_err": round(abs(float(r["human_score"]) - ip.score), 2),
            "improved_better": abs(float(r["human_score"]) - ip.score) < abs(float(r["human_score"]) - bp.score),
        })

    n_improved_better = sum(1 for c in comparison_rows if c["improved_better"])
    n_baseline_better = sum(1 for c in comparison_rows
                             if c["baseline_abs_err"] < c["improved_abs_err"])
    n_tie = len(comparison_rows) - n_improved_better - n_baseline_better

    summary = {"overall": overall, "by_subset": by_subset,
               "wins": {"improved_better": n_improved_better,
                        "baseline_better": n_baseline_better, "tie": n_tie}}

    suffix = "" if backend_name == "spacy" else f"_{backend_name}"
    with open(RESULTS_DIR / f"week4_comparison_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / f"week4_comparison_rows{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("=== Week 4: Baseline (TF-IDF) vs Improved (Embeddings) ===")
    print(f"n_test = {len(test_rows)}\n")
    print(f"{'Metric':<28}{'Baseline':<15}{'Improved':<15}")
    for k in ["MAE", "RMSE", "pearson_r", "agreement_within_1_mark"]:
        print(f"{k:<28}{overall['baseline_tfidf'][k]!s:<15}{overall['improved_embedding'][k]!s:<15}")
    print()
    print("-- Normal (unseen) test cases --")
    for k in ["MAE", "agreement_within_1_mark"]:
        print(f"{k:<28}{by_subset['baseline_tfidf']['normal'][k]!s:<15}{by_subset['improved_embedding']['normal'][k]!s:<15}")
    print()
    print("-- Adversarial (paraphrase / keyword-stuffing) cases --")
    for k in ["MAE", "agreement_within_1_mark"]:
        print(f"{k:<28}{by_subset['baseline_tfidf']['adversarial'][k]!s:<15}{by_subset['improved_embedding']['adversarial'][k]!s:<15}")
    print()
    print(f"Per-case wins -> improved better: {n_improved_better}, baseline better: {n_baseline_better}, tie: {n_tie}")


if __name__ == "__main__":
    main()
