"""
evaluate_embedding_model.py
-----------------------------
Group 2 - AI Exam Answer Grader
Week-3 deliverable check: confirms "Version 1 of the NLP pipeline" actually
works, via an INTRINSIC evaluation of the core NLP component itself -
concept-level coverage detection - using the metrics Section 7 prescribes
for Information Extraction / Similarity-Matching tasks (Precision, Recall,
F1, cosine similarity), separate from the Week-2 baseline's score-level
MAE/RMSE (that head-to-head "Grading" comparison is Week 4's job).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.embedding_grader import EmbeddingSemanticGrader, get_backend  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "spacy"
    print(f"(using embedding backend: {backend_name})\n")

    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    grader = EmbeddingSemanticGrader(backend=get_backend(backend_name)).fit(train_rows)

    # ---- Intrinsic, point-level evaluation on the held-out test rows ----
    y_true, y_pred, sims = [], [], []
    point_rows = []
    for r in test_rows:
        rubric = json.loads(r["rubric"])
        covered_gt = set(json.loads(r["points_covered"]))
        for p in rubric:
            match = grader.predict_point(p["id"], p["text"], p["marks"], r["student_answer"], point_meta=p)
            gt = 1 if p["id"] in covered_gt else 0
            y_true.append(gt)
            y_pred.append(1 if match.credited else 0)
            sims.append(match.similarity)
            point_rows.append({
                "qid": r["qid"], "student_id": r["student_id"], "quality_tier": r["quality_tier"],
                "point_id": p["id"], "point_text": p["text"], "ground_truth_covered": gt,
                "predicted_covered": int(match.credited), "similarity": match.similarity,
                "coverage_prob": match.coverage_prob, "best_matched_sentence": match.best_sentence,
            })

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "n_test_points": len(y_true),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "confusion_matrix": {"labels": ["not_covered", "covered"], "matrix": cm},
        "mean_best_sentence_similarity": round(float(np.mean(sims)), 3),
    }

    suffix = "" if backend_name == "spacy" else f"_{backend_name}"
    with open(RESULTS_DIR / f"week3_embedding_concept_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS_DIR / f"week3_point_level_predictions{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
        writer.writeheader()
        writer.writerows(point_rows)

    print("=== Week 3 Core NLP Model - Intrinsic Concept-Coverage Evaluation ===")
    print(f"Test rubric-point examples: {metrics['n_test_points']}")
    print(f"Precision: {metrics['precision']}  Recall: {metrics['recall']}  F1: {metrics['f1']}")
    print(f"Confusion matrix [ [TN,FP],[FN,TP] ]: {cm}")
    print(f"Mean best-matching-sentence similarity: {metrics['mean_best_sentence_similarity']}")

    # Also run full grade_batch so grader.py's end-to-end path is exercised
    # and produces a quick sanity check on the adversarial cases from Week 2.
    print()
    print("Sanity check on Week-2's adversarial cases (should now behave better):")
    adversarial_qids_students = {("PY01", "paraphrase_trap"), ("ST01", "paraphrase_trap"),
                                  ("DS02", "paraphrase_trap"), ("ML02", "keyword_stuffing"),
                                  ("DB02", "keyword_stuffing")}
    results = grader.grade_batch(test_rows)
    for r, res in zip(test_rows, results):
        if (r["qid"], r["quality_tier"]) in adversarial_qids_students:
            print(f"  [{r['qid']}/{r['quality_tier']}] human={r['human_score']}  "
                  f"embedding_pred={res.score}  confidence={res.confidence}")


if __name__ == "__main__":
    main()
