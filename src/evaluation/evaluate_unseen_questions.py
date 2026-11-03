"""
evaluate_unseen_questions.py
------------------------------
Group 2 - AI Exam Answer Grader
Week-6: honest generalisation check for a REUSABLE asset.

Why this exists
---------------
Every evaluation in Weeks 2-5 split the data by STUDENT ANSWER within the
same 15 questions: the model was calibrated on some students' answers to
Q1 and tested on other students' answers to the SAME Q1 (same rubric,
same reference answer). That measures "can it grade another student of a
question it has seen" - not "can it grade a question it has never seen",
which is exactly what another group calling POST /evaluate-answer with
their own question would need. This script measures the latter.

Protocol: leave-one-question-out cross-validation. For each of the 15
questions, fit BOTH models on the other 14 questions' rows (all splits),
then grade every row of the held-out question. Predictions from all 15
folds are pooled and scored with the same Section 7 metrics used
everywhere else (MAE, RMSE, Pearson r, agreement within +/-1 mark).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.baseline_grader import TFIDFBaselineGrader                      # noqa: E402
from models.embedding_grader import EmbeddingSemanticGrader, get_backend    # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exam_grading_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADVERSARIAL_TIERS = {"paraphrase_trap", "keyword_stuffing", "blank", "off_topic",
                      "refusal", "confidently_wrong", "mixed_with_error"}


def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    r, _ = pearsonr(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else (float("nan"), 0)
    return {
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 3),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3),
        "pearson_r": round(float(r), 3) if r == r else None,
        "agreement_within_1_mark": round(float(np.mean(np.abs(y_true - y_pred) <= 1.0)), 3),
        "n": int(len(y_true)),
    }


def main():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    qids = sorted(set(r["qid"] for r in rows))
    backend = get_backend("spacy")  # load spaCy once, reuse across folds

    out_rows = []
    pt_true, pt_pred = [], []
    for qid in qids:
        train = [r for r in rows if r["qid"] != qid]
        held = [r for r in rows if r["qid"] == qid]
        base = TFIDFBaselineGrader().fit(train)
        emb = EmbeddingSemanticGrader(backend=backend).fit(train)
        b_res, e_res = base.grade_batch(held), emb.grade_batch(held)
        for r, b, e in zip(held, b_res, e_res):
            out_rows.append({
                "qid": qid, "student_id": r["student_id"], "quality_tier": r["quality_tier"],
                "max_marks": r["max_marks"], "human_score": float(r["human_score"]),
                "baseline_pred": b.score, "embedding_pred": e.score,
                "baseline_conf": b.confidence, "embedding_conf": e.confidence,
                "embedding_contradicted": any(m.fact_check == "contradicted" for m in e.point_matches),
            })
            covered = set(json.loads(r["points_covered"]))
            for m in e.point_matches:
                pt_true.append(1 if m.point_id in covered else 0)
                pt_pred.append(1 if m.credited else 0)
        print(f"  fold {qid}: held out {len(held)} rows", flush=True)

    human = [o["human_score"] for o in out_rows]
    result = {
        "protocol": "leave-one-question-out over 15 questions; models fit on the other 14",
        "overall": {
            "baseline": metrics(human, [o["baseline_pred"] for o in out_rows]),
            "embedding": metrics(human, [o["embedding_pred"] for o in out_rows]),
        },
    }
    normal = [o for o in out_rows if o["quality_tier"] not in ADVERSARIAL_TIERS]
    adv = [o for o in out_rows if o["quality_tier"] in ADVERSARIAL_TIERS]
    for label, subset in (("normal_answers", normal), ("adversarial_and_robustness_answers", adv)):
        h = [o["human_score"] for o in subset]
        result[label] = {
            "baseline": metrics(h, [o["baseline_pred"] for o in subset]),
            "embedding": metrics(h, [o["embedding_pred"] for o in subset]),
        }
    p, rc, f1, _ = precision_recall_fscore_support(pt_true, pt_pred, average="binary", zero_division=0)
    result["embedding_point_level"] = {"precision": round(float(p), 3), "recall": round(float(rc), 3),
                                        "f1": round(float(f1), 3), "n_points": len(pt_true)}

    with open(RESULTS_DIR / "week6_unseen_question_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "week6_unseen_question_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("\n=== Unseen-question generalisation (leave-one-question-out) ===")
    for section in ("overall", "normal_answers", "adversarial_and_robustness_answers"):
        s = result[section]
        print(f"\n[{section}]")
        for name in ("baseline", "embedding"):
            m = s[name]
            print(f"  {name:<10} MAE={m['MAE']:<6} RMSE={m['RMSE']:<6} r={m['pearson_r']!s:<6} "
                  f"agree+-1={m['agreement_within_1_mark']:<6} n={m['n']}")
    print(f"\nEmbedding point-level: {result['embedding_point_level']}")


if __name__ == "__main__":
    main()
