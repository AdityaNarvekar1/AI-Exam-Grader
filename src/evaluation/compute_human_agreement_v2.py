"""
compute_human_agreement_v2.py  (v2)
-------------------------------------
Reads 1..N filled-in copies of BLIND_SHEET.csv and reports
  * inter-rater agreement (QWK, MAE, Pearson) for every pair of graders
  * each grader vs the SYNTHETIC label  -> is the synthetic ground truth believable?
  * mean-human vs synthetic, and (optionally) vs any system's predictions (rows_*.csv from run_benchmark)
  * items where graders disagree by >= 25% of max marks (to adjudicate)  -> results/v2/human_validation/disagreements.csv
Interpretation guide:  if human-vs-human QWK is LOWER than system-vs-human QWK, the system is already
within the noise of human grading on this data; if synthetic-vs-human QWK is low, the synthetic labels
should not be used as ground truth.
  python src/evaluation/compute_human_agreement_v2.py --sheets r1.csv r2.csv [--system results/v2/rows_nli_C.csv --col nli_bi__pred]
"""
import argparse, csv, itertools, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_utils import metrics, cluster_bootstrap  # noqa: E402

D = ROOT / "results" / "v2" / "human_validation"


def read_sheet(p):
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        s = (r.get("grader_score") or "").strip()
        if s != "":
            out[r["item_id"]] = float(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--key", default=str(D / "ANSWER_KEY.csv"))
    ap.add_argument("--system", help="rows_*.csv written by run_benchmark.py")
    ap.add_argument("--col", help="prediction column in --system file, e.g. nli_bi__pred")
    ap.add_argument("--out", default=str(D / "agreement_report.json"))
    a = ap.parse_args(argv)
    key = {r["item_id"]: r for r in csv.DictReader(open(a.key, newline="", encoding="utf-8"))}
    sheets = [read_sheet(p) for p in a.sheets]
    rep = {"n_sheets": len(sheets), "items_per_sheet": [len(s) for s in sheets], "pairs": {}, "vs_synthetic": {}}
    mx = lambda ids: [float(key[i]["max_marks"]) for i in ids]

    for (i, si), (j, sj) in itertools.combinations(enumerate(sheets), 2):
        ids = sorted(set(si) & set(sj))
        rep["pairs"][f"rater{i+1}_vs_rater{j+1}"] = metrics([si[x] for x in ids], [sj[x] for x in ids], mx(ids)) if len(ids) > 2 else "too few shared items"
    for i, s in enumerate(sheets):
        ids = sorted(set(s) & set(key))
        rep["vs_synthetic"][f"rater{i+1}"] = metrics([float(key[x]["synthetic_score"]) for x in ids], [s[x] for x in ids], mx(ids))
    common = sorted(set.intersection(*[set(s) for s in sheets]) & set(key)) if sheets else []
    if common:
        mean_h = {x: float(np.mean([s[x] for s in sheets])) for x in common}
        rep["mean_human_vs_synthetic"] = metrics([float(key[x]["synthetic_score"]) for x in common], [mean_h[x] for x in common], mx(common))
        by_tier = {}
        for t in sorted({key[x]["quality_tier"] for x in common}):
            ids = [x for x in common if key[x]["quality_tier"] == t]
            by_tier[t] = round(float(np.mean([abs(mean_h[x] - float(key[x]["synthetic_score"])) for x in ids])), 2)
        rep["mean_abs_gap_human_vs_synthetic_by_tier"] = by_tier
        if a.system and a.col:
            sysrows = {r["student_id"]: float(r[a.col]) for r in csv.DictReader(open(a.system, newline="", encoding="utf-8"))}
            ids = [x for x in common if key[x]["student_id"] in sysrows]
            if len(ids) > 2:
                rep["system_vs_mean_human"] = metrics([mean_h[x] for x in ids], [sysrows[key[x]["student_id"]] for x in ids], mx(ids))
                rep["system_vs_mean_human"]["n_matched"] = len(ids)
        dis = [(x, key[x]["student_id"], key[x]["quality_tier"], [s[x] for s in sheets]) for x in common
               if max(s[x] for s in sheets) - min(s[x] for s in sheets) >= 0.25 * float(key[x]["max_marks"])]
        with open(D / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["item_id", "student_id", "tier", "scores"])
            for x in dis:
                w.writerow([x[0], x[1], x[2], json.dumps(x[3])])
        rep["n_disagreements_ge_25pct"] = len(dis)
    json.dump(rep, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float))
    return rep


if __name__ == "__main__":
    main()
