"""
analyze_selective_prediction.py  (v2)
---------------------------------------
Replaces "one review threshold (0.7) picked after looking at the rows" with
  * a risk-coverage curve  (error of the answers the system keeps, vs the share it keeps)
  * AURC against a random ordering and an oracle ordering
  * leave-one-question-out split-conformal score intervals (honest +/- band per answer)

Works on any rows CSV with columns: qid, max_marks, human_score, <pred columns>.
Defaults to results/week6_unseen_question_rows.csv (the Week 6 leave-one-question-out output),
so it needs NO models and runs anywhere.

  python src/evaluation/analyze_selective_prediction.py
  python src/evaluation/analyze_selective_prediction.py --rows results/v2/my_rows.csv --pred hybrid_pred --conf hybrid_conf
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_utils import risk_coverage, risk_at_coverage, loqo_conformal, cluster_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROOT / "results" / "week6_unseen_question_rows.csv"))
    ap.add_argument("--pred", default=None, help="prediction column; default: derive the Week-6 hybrid")
    ap.add_argument("--conf", default=None, help="confidence column; default: 1-|baseline-embedding|/max")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.rows, newline="", encoding="utf-8")))
    y = np.array([float(r["human_score"]) for r in rows])
    mx = np.array([float(r["max_marks"]) for r in rows])
    q = [r["qid"] for r in rows]
    if args.pred:
        pred = np.array([float(r[args.pred]) for r in rows])
        conf = np.array([float(r[args.conf]) for r in rows]) if args.conf else np.ones(len(rows))
        label = args.pred
    else:
        b = np.array([float(r["baseline_pred"]) for r in rows]); e = np.array([float(r["embedding_pred"]) for r in rows])
        contra = np.array([r["embedding_contradicted"] == "True" for r in rows])
        pred = np.minimum(np.where(contra, np.minimum(b, e), np.round((b + e) / 2 * 2) / 2), mx)
        conf = 1 - np.abs(b - e) / mx
        conf = np.where(contra, 0.0, conf)
        label = "week6_hybrid"
    err_n = np.abs(y - pred) / mx               # normalised error (fraction of max marks)

    rc = risk_coverage(err_n, conf)
    out = {"predictor": label, "n": len(y), "mean_nMAE": round(float(err_n.mean()), 4),
           "AURC": rc["aurc"], "AURC_random_order": rc["random_aurc"], "AURC_oracle_order": rc["oracle_aurc"],
           "risk_at_coverage": {f"{int(c*100)}%": round(risk_at_coverage(rc, c), 4) for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}}
    # how much of the oracle-vs-random gap does the confidence recover?  (0 = useless, 1 = perfect)
    gap = rc["random_aurc"] - rc["oracle_aurc"]
    out["share_of_oracle_gain_recovered"] = round((rc["random_aurc"] - rc["aurc"]) / gap, 3) if gap > 0 else None
    # the Week-6 operating point, for comparison
    flagged = conf < 0.7
    out["week6_threshold_0.7"] = {"flagged_frac": round(float(flagged.mean()), 3),
                                  "nMAE_kept": round(float(err_n[~flagged].mean()), 4),
                                  "nMAE_flagged": round(float(err_n[flagged].mean()), 4) if flagged.any() else None}
    # conformal
    for alpha in (0.2, 0.1):
        c = loqo_conformal(err_n, q, alpha)
        out[f"conformal_alpha_{alpha}"] = {"target": c["target_coverage"], "empirical_coverage": c["empirical_coverage"],
                                          "mean_half_width_as_pct_of_max": round(c["mean_half_width_frac"] * 100, 1)}
    out["mean_nMAE_cluster_bootstrap"] = cluster_bootstrap(err_n, q)
    print(json.dumps(out, indent=2))
    dst = ROOT / "results" / "v2" / f"selective_{label}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)


if __name__ == "__main__":
    main()
