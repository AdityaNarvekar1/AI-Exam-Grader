"""
analyze_hybrid_and_confidence.py
-----------------------------------
Group 2 - AI Exam Answer Grader
Week-6: reproducible backing for three claims made in the API docs and the
Week 6 report. Reads results/week6_unseen_question_rows.csv (written by
evaluate_unseen_questions.py) and reports:

  1. Metrics for baseline, embedding, and the fixed 50/50 'hybrid'
     (the weights were fixed in advance and NOT tuned).
  2. Paired-bootstrap 95% confidence intervals on MAE differences, so
     "better" is only claimed where the interval excludes zero.
  3. Whether each approach's confidence score actually tracks its error
     (Spearman correlation between confidence and -error), and what the
     'needs_human_review' threshold (confidence < 0.7) does in practice.

Honest caveats baked into the output: n=70 synthetic rows (only 10 are
adversarial/robustness cases), the 0.7 threshold was chosen after looking
at these same rows (so its flagged/unflagged MAEs are in-sample), and the
decision to ADD a hybrid was made after seeing baseline/embedding results.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
ROWS_PATH = RESULTS_DIR / "week6_unseen_question_rows.csv"
ADV = {"paraphrase_trap", "keyword_stuffing", "blank", "off_topic",
       "refusal", "confidently_wrong", "mixed_with_error"}
REVIEW_THRESHOLD = 0.7
N_BOOT = 5000


def m(h, p):
    h, p = np.array(h, float), np.array(p, float)
    r = pearsonr(h, p)[0] if len(set(h)) > 1 and len(set(p)) > 1 else float("nan")
    return {"MAE": round(float(np.mean(abs(h - p))), 3),
            "RMSE": round(float(np.sqrt(np.mean((h - p) ** 2))), 3),
            "pearson_r": round(float(r), 3),
            "agreement_within_1_mark": round(float(np.mean(abs(h - p) <= 1)), 3), "n": int(len(h))}


def main():
    rows = list(csv.DictReader(open(ROWS_PATH)))
    h = np.array([float(r["human_score"]) for r in rows])
    mx = np.array([float(r["max_marks"]) for r in rows])
    b = np.array([float(r["baseline_pred"]) for r in rows])
    e = np.array([float(r["embedding_pred"]) for r in rows])
    contradicted = np.array([r["embedding_contradicted"] == "True" for r in rows])
    # Same rule as src/api/main.py: plain average, EXCEPT when the fact-check layer
    # detects a contradiction - then defer to the more conservative score.
    y = np.minimum(np.where(contradicted, np.minimum(b, e), np.round((b + e) / 2 * 2) / 2), mx)
    tiers = np.array([r["quality_tier"] for r in rows])
    adv = np.array([t in ADV for t in tiers])

    out = {"hybrid_rule": "mean(baseline, embedding); if fact-check contradicted a point: min(baseline, embedding)",
           "rows_where_fact_check_fired": int(contradicted.sum()),
           "metrics": {}, "bootstrap_mae_difference": {}, "confidence_informativeness": {}, "review_flag": {}}
    for label, mask in (("overall", np.ones(len(h), bool)), ("normal", ~adv), ("adversarial_and_robustness", adv)):
        out["metrics"][label] = {"baseline": m(h[mask], b[mask]), "embedding": m(h[mask], e[mask]),
                                  "hybrid": m(h[mask], y[mask])}

    rng = np.random.default_rng(0)
    eb, ee, ey = abs(h - b), abs(h - e), abs(h - y)

    def boot(a, c):
        d = [a[i].mean() - c[i].mean() for i in (rng.integers(0, len(h), len(h)) for _ in range(N_BOOT))]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"mean_diff_in_MAE": round(float(a.mean() - c.mean()), 3),
                "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
    out["bootstrap_mae_difference"] = {
        "baseline_minus_hybrid (positive = hybrid better)": boot(eb, ey),
        "embedding_minus_hybrid (positive = hybrid better)": boot(ee, ey),
        "embedding_minus_baseline (positive = baseline better)": boot(ee, eb),
    }

    bc = np.array([float(r["baseline_conf"]) for r in rows])
    ec = np.array([float(r["embedding_conf"]) for r in rows])
    hc = 1 - abs(b - e) / mx
    for name, conf, pred in (("baseline", bc, b), ("embedding", ec, e), ("hybrid_agreement", hc, y)):
        rho, p = spearmanr(conf, -abs(h - pred) / mx)
        out["confidence_informativeness"][name] = {"spearman_conf_vs_neg_error": round(float(rho), 3),
                                                    "p_value": round(float(p), 4)}
    flag = (hc < REVIEW_THRESHOLD) | contradicted
    out["review_flag"] = {
        "threshold": REVIEW_THRESHOLD, "flagged": int(flag.sum()), "of": int(len(h)),
        "fraction_flagged": round(float(flag.mean()), 3),
        "hybrid_MAE_flagged": round(float(ey[flag].mean()), 3),
        "hybrid_MAE_not_flagged": round(float(ey[~flag].mean()), 3),
        "note": "threshold chosen after viewing these rows - these two MAEs are in-sample",
    }
    with open(RESULTS_DIR / "week6_hybrid_and_confidence_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
