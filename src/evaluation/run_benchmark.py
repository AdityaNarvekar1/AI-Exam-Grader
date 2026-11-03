"""
run_benchmark.py  (v2)
-----------------------
One benchmark, three protocols, honest statistics.

PROTOCOLS
  A  loqo_original   leave-one-question-out on the original 70 rows        (Week 6 protocol)
  B  train70_test_challenge   fit on the original 70, test on the 85-row challenge set whose
                              length is de-correlated from quality (the shortcut test)
  C  loqo_pooled     leave-one-question-out on original + challenge (155 rows) - the most
                     demanding and the one we recommend quoting

STATISTICS  QWK, nMAE, Pearson/Spearman, within-10%-of-max; cluster-bootstrap (by question) CI
on MAE and on the paired MAE difference versus a reference grader.

USAGE
  python src/evaluation/run_benchmark.py                      # trivial + lexical + tfidf (no downloads)
  python src/evaluation/run_benchmark.py --graders const length tfidf lexical hybrid_nli
  python src/evaluation/run_benchmark.py --protocols B C --ref tfidf
Outputs results/v2/benchmark_<tag>.json and a Markdown table on stdout.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "models"))

from eval_utils import metrics, cluster_bootstrap, paired_cluster_bootstrap, risk_coverage  # noqa: E402
import graders as G  # noqa: E402

ORIG = ROOT / "data" / "exam_grading_dataset.csv"
CHAL = ROOT / "data" / "v2" / "challenge_set.csv"
OUTDIR = ROOT / "results" / "v2"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_predict(name, train, test):
    ad = G.REGISTRY[name]()
    ad.fit(train)
    return ad.predict(test)


def run_protocol(proto, names, orig, chal):
    """Returns (eval_rows, {grader: [GradeOut]})."""
    preds = {n: [] for n in names}
    if proto == "A":
        rows_eval = []
        order = sorted({r["qid"] for r in orig})
        for q in order:
            tr = [r for r in orig if r["qid"] != q]
            te = [r for r in orig if r["qid"] == q]
            rows_eval += te
            for n in names:
                preds[n] += fit_predict(n, tr, te)
        return rows_eval, preds
    if proto == "B":
        for n in names:
            preds[n] = fit_predict(n, orig, chal)
        return chal, preds
    if proto == "C":
        pool = orig + chal
        rows_eval = []
        for q in sorted({r["qid"] for r in pool}):
            tr = [r for r in pool if r["qid"] != q]
            te = [r for r in pool if r["qid"] == q]
            rows_eval += te
            for n in names:
                preds[n] += fit_predict(n, tr, te)
        return rows_eval, preds
    raise ValueError(proto)


CORRECT_TIERS = {"excellent", "good", "terse_correct", "paraphrase_correct", "copied_reference", "all_caps", "typos", "hinglish"}
ERROR_TIERS = {"confidently_wrong", "mixed_with_error", "error_variant", "shotgun_contradictory"}


def point_level(rows, gs):
    """Point-level precision/recall/F1 vs points_covered, for graders that expose point_credit."""
    tp = fp = fn = 0
    for r, g in zip(rows, gs):
        if g.point_credit is None:
            return None
        cov = set(json.loads(r["points_covered"]))
        for pid, c in g.point_credit.items():
            tp += int(c and pid in cov); fp += int(c and pid not in cov); fn += int((not c) and pid in cov)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
            "n_points_evaluated": sum(len(json.loads(r["rubric"])) for r in rows)}


def summarise(rows, preds, ref):
    y = [float(r["human_score"]) for r in rows]
    mx = [float(r["max_marks"]) for r in rows]
    groups = [r["qid"] for r in rows]
    tiers = np.array([r["quality_tier"] for r in rows])
    out = {}
    ref_err = np.abs(np.array(y) - np.array([g.score for g in preds[ref]])) if ref in preds else None
    for n, gs in preds.items():
        p = [g.score for g in gs]
        m = metrics(y, p, mx)
        err = np.abs(np.array(y) - np.array(p))
        m["MAE_ci95_cluster_bootstrap"] = cluster_bootstrap(err, groups)
        if ref_err is not None and n != ref:
            m[f"MAE_diff_vs_{ref}"] = paired_cluster_bootstrap(ref_err, err, groups)  # + => n better
        m["MAE_by_tier"] = {t: round(float(err[tiers == t].mean()), 2) for t in sorted(set(tiers))}
        pl = point_level(rows, gs)
        if pl:
            m["point_level"] = pl
        if any(g.contradicted_points is not None for g in gs) and any(g.contradicted_points for g in gs):
            fired = np.array([bool(g.contradicted_points) for g in gs])
            m["veto"] = {"fires_on_correct_answers": int(fired[np.isin(tiers, list(CORRECT_TIERS))].sum()),
                         "n_correct_answers": int(np.isin(tiers, list(CORRECT_TIERS)).sum()),
                         "fires_on_error_answers": int(fired[np.isin(tiers, list(ERROR_TIERS))].sum()),
                         "n_error_answers": int(np.isin(tiers, list(ERROR_TIERS)).sum())}
        if all(g.confidence is not None for g in gs):
            rc = risk_coverage(err / np.array(mx), [g.confidence for g in gs])
            gap = rc["random_aurc"] - rc["oracle_aurc"]
            m["selective"] = {"AURC": rc["aurc"], "AURC_random": rc["random_aurc"], "AURC_oracle": rc["oracle_aurc"],
                              "share_of_oracle_gain": round((rc["random_aurc"] - rc["aurc"]) / gap, 3) if gap > 0 else None}
        out[n] = m
    return out


def dump_rows(path, rows, preds):
    names = list(preds)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "student_id", "quality_tier", "max_marks", "human_score"] +
                   [f"{n}__{k}" for n in names for k in ("pred", "conf", "n_vetoed")])
        for i, r in enumerate(rows):
            line = [r["qid"], r["student_id"], r["quality_tier"], r["max_marks"], r["human_score"]]
            for n in names:
                g = preds[n][i]
                line += [g.score, "" if g.confidence is None else g.confidence, len(g.contradicted_points or [])]
            w.writerow(line)


def md_table(summary):
    lines = ["| grader | n | MAE | MAE 95% CI (by question) | nMAE % | QWK | Pearson | Spearman |", "|---|---|---|---|---|---|---|---|"]
    for n, m in summary.items():
        ci = m["MAE_ci95_cluster_bootstrap"]
        lines.append(f"| {n} | {m['n']} | {m['MAE']} | [{ci['ci_low']}, {ci['ci_high']}] | {m['nMAE_pct_of_max']} | "
                     f"{m['QWK']} | {m['pearson_r']} | {m['spearman_rho']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graders", nargs="+", default=["const", "length", "tfidf", "lexical"])
    ap.add_argument("--protocols", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--ref", default="tfidf", help="grader the paired differences are computed against")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    G.load_optional_adapters()
    orig, chal = load(ORIG), load(CHAL)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for proto in args.protocols:
        rows, preds = run_protocol(proto, args.graders, orig, chal)
        s = summarise(rows, preds, args.ref)
        result[proto] = s
        dump_rows(OUTDIR / f"rows_{args.tag}_{proto}.csv", rows, preds)
        title = {"A": "A  leave-one-question-out, original 70", "B": "B  train on original 70 -> test on challenge set (85)",
                 "C": "C  leave-one-question-out, pooled 155"}[proto]
        print(f"\n### Protocol {title}\n")
        print(md_table(s))
        extras = [(n, m) for n, m in s.items() if "point_level" in m or "veto" in m or "selective" in m]
        if extras:
            print("\n| grader | point P / R / F1 | veto fires: correct / error answers | AURC (random, oracle) | share of oracle gain |")
            print("|---|---|---|---|---|")
            for n, m in extras:
                pl, ve, se = m.get("point_level"), m.get("veto"), m.get("selective")
                print(f"| {n} | {pl and (pl['precision'], pl['recall'], pl['f1'])} | "
                      f"{ve and (str(ve['fires_on_correct_answers']) + '/' + str(ve['n_correct_answers']) + ' , ' + str(ve['fires_on_error_answers']) + '/' + str(ve['n_error_answers']))} | "
                      f"{se and (se['AURC'], se['AURC_random'], se['AURC_oracle'])} | {se and se['share_of_oracle_gain']} |")
    with open(OUTDIR / f"benchmark_{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nwrote {OUTDIR / f'benchmark_{args.tag}.json'}")


if __name__ == "__main__":
    main()
