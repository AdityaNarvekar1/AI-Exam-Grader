"""
run_external_benchmark.py  (v2)
--------------------------------
Validates the new signals on PUBLIC, human-graded data the team did not write.

TASK mohler       answer-level score regression, 5-fold GroupKFold BY QUESTION (no question appears in both
                  train and test), cluster-bootstrap CIs, paired difference vs the reference-similarity baseline.
                  graders: const length refsim lex bi nli nli_bi judge
TASK scientsbank  3-way classification (correct / contradictory / incorrect) trained on `train`, evaluated on the
                  Unseen-Answers / Unseen-Questions / Unseen-Domains test splits; plus zero-shot AUROC of the NLI
                  contradiction probability for detecting "contradictory" answers  <- this is the evidence for
                  replacing hand-written fact-binding rules with an NLI contradiction signal.

  python src/data/export_hf_datasets.py                       # once, on your machine
  python src/evaluation/run_external_benchmark.py --task mohler --data data/external/mohler_open_ended.csv \
         --graders const length refsim lex bi nli nli_bi
  python src/evaluation/run_external_benchmark.py --task scientsbank --max-train 2000 --max-test 1000

Models come from the same env vars as run_benchmark.py (G2_NLI_MODEL, G2_BI_MODEL, G2_LLM_*; "stub" = plumbing test only).
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("src/evaluation", "src/models", "src/data"):
    sys.path.insert(0, str(ROOT / sub))

from eval_utils import metrics, cluster_bootstrap, paired_cluster_bootstrap   # noqa: E402
from load_external import load_mohler, load_scientsbank                        # noqa: E402
import model_adapters as MA                                                    # noqa: E402
from baseline_grader import TFIDFBaselineGrader                                # noqa: E402
from answer_level_grader import AnswerLevelGrader                              # noqa: E402

OUT = ROOT / "results" / "v2"


# ------------------------------------------------------------------------------- mohler
def frac_pred(name, tr, te):
    """-> np.array of predicted score FRACTIONS for the rows in `te`."""
    n = lambda r: len((r["student_answer"] or "").split())
    if name == "const":
        return np.full(len(te), np.mean([r["human_score"] / r["max_marks"] for r in tr]))
    if name == "length":
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(np.log1p([[n(r)] for r in tr]), [r["human_score"] / r["max_marks"] for r in tr])
        return np.clip(m.predict(np.log1p([[n(r)] for r in te])), 0, 1)
    if name == "refsim":
        g = TFIDFBaselineGrader().fit(tr)
        return np.array([g.grade_one(r["question"], r["reference_answer"], None, r["student_answer"], r["max_marks"]).score / r["max_marks"] for r in te])
    if name == "judge":
        j = MA.JudgeAdapter()._judge()
        return np.array([j.grade_one(r["question"], r["reference_answer"], json.loads(r["rubric"]), r["student_answer"], r["max_marks"]).score / r["max_marks"] for r in te])
    nli = MA._nli_scorer() if "nli" in name else None
    bi = MA._bi_scorer() if "bi" in name else None
    g = AnswerLevelGrader(nli=nli, bi=bi, cache=MA._FEATURE_CACHE).fit(tr)
    return g.predict_fraction(te)


def run_mohler(a):
    from sklearn.model_selection import GroupKFold
    rows = load_mohler(a.data)
    if a.max_rows and len(rows) > a.max_rows:
        random.Random(0).shuffle(rows)
        rows = rows[:a.max_rows]
    groups = np.array([r["qid"] for r in rows])
    y = np.array([r["human_score"] for r in rows]); mx = np.array([r["max_marks"] for r in rows])
    preds = {g: np.zeros(len(rows)) for g in a.graders}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=min(a.folds, len(set(groups)))).split(rows, groups=groups)):
        trr, ter = [rows[i] for i in tr], [rows[i] for i in te]
        for g in a.graders:
            preds[g][te] = frac_pred(g, trr, ter) * mx[te]
        print(f"  fold {k + 1}: {len(ter)} test answers, {len(set(groups[te]))} unseen questions", flush=True)
    res = {}
    ref = "refsim" if "refsim" in preds else None
    for g, p in preds.items():
        m = metrics(y, p, mx)
        err = np.abs(y - p)
        m["MAE_ci95_cluster_bootstrap"] = cluster_bootstrap(err, groups)
        if ref and g != ref:
            m["MAE_diff_vs_refsim (+ = better)"] = paired_cluster_bootstrap(np.abs(y - preds[ref]), err, groups)
        res[g] = m
    print(f"\n### Mohler (n={len(rows)} answers, {len(set(groups))} questions), GroupKFold by question\n")
    print("| grader | MAE (0-5) | MAE 95% CI | QWK | Pearson | Spearman | vs refsim (MAE diff, CI) |\n|---|---|---|---|---|---|---|")
    for g, m in res.items():
        ci = m["MAE_ci95_cluster_bootstrap"]; d = m.get("MAE_diff_vs_refsim (+ = better)")
        print(f"| {g} | {m['MAE']} | [{ci['ci_low']}, {ci['ci_high']}] | {m['QWK']} | {m['pearson_r']} | {m['spearman_rho']} | "
              f"{'-' if not d else str(d['estimate']) + ' [' + str(d['ci_low']) + ', ' + str(d['ci_high']) + ']' + (' *' if d['excludes_zero'] else '')} |")
    return res


# ------------------------------------------------------------------------- scientsbank
def featurize(rows, nli, bi):
    from feature_grader import FeatureGrader
    fg = FeatureGrader(nli=nli, bi=bi, use_factcheck=False, cache=MA._FEATURE_CACHE)
    X = []
    for r in rows:
        rub = [{"id": "p1", "text": r["reference_answer"], "marks": 1.0}]
        X.append(fg._answer_features(rub, r["student_answer"] or "")[0][0])
    return np.array(X), fg.feature_names


def run_scientsbank(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = Path(a.dir)
    sub = lambda rows, n: random.Random(0).sample(rows, n) if n and len(rows) > n else rows
    train = sub(load_scientsbank(d / "scientsbank_train.csv", "train"), a.max_train)
    tests = {s: sub(load_scientsbank(d / f"scientsbank_{s}.csv", s), a.max_test) for s in a.tests if (d / f"scientsbank_{s}.csv").exists()}
    labels = ["correct", "contradictory", "incorrect"]
    cfgs = {"majority": None, "lex": (False, False), "bi": (False, True), "nli": (True, False), "nli_bi": (True, True)}
    cfgs = {k: v for k, v in cfgs.items() if k in a.graders or k == "majority"}
    res = {}
    for name, cfg in cfgs.items():
        if cfg is None:
            maj = max(labels, key=lambda l: sum(r["label3"] == l for r in train))
            res[name] = {s: {"macro_F1": round(f1_score([r["label3"] for r in t], [maj] * len(t), average="macro", labels=labels, zero_division=0), 3)} for s, t in tests.items()}
            continue
        nli = MA._nli_scorer() if cfg[0] else None
        bi = MA._bi_scorer() if cfg[1] else None
        Xtr, names = featurize(train, nli, bi)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(Xtr, [r["label3"] for r in train])
        res[name] = {}
        for s, t in tests.items():
            X, _ = featurize(t, nli, bi)
            yt = np.array([r["label3"] for r in t]); yp = clf.predict(X)
            m = {"macro_F1": round(f1_score(yt, yp, average="macro", labels=labels, zero_division=0), 3),
                 "weighted_F1": round(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0), 3),
                 "accuracy": round(accuracy_score(yt, yp), 3), "n": len(t)}
            is_c = (yt == "contradictory").astype(int)
            if is_c.sum() and (1 - is_c).sum():
                m["AUROC_contradictory_learned"] = round(roc_auc_score(is_c, clf.predict_proba(X)[:, list(clf.classes_).index("contradictory")]), 3)
                if "contra_max" in names:
                    m["AUROC_contradictory_zero_shot(contra_max)"] = round(roc_auc_score(is_c, X[:, names.index("contra_max")]), 3)
            res[name][s] = m
    print(f"\n### SciEntsBank 3-way (train n={len(train)}); test splits: {list(tests)}\n")
    for name, per in res.items():
        print(f"- **{name}**: " + "; ".join(f"{s}: {json.dumps(m)}" for s, m in per.items()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mohler", "scientsbank"], required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "external" / "mohler_open_ended.csv"))
    ap.add_argument("--dir", default=str(ROOT / "data" / "external"))
    ap.add_argument("--graders", nargs="+", default=["const", "length", "refsim", "lex", "bi", "nli", "nli_bi"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--tests", nargs="+", default=["test_ua", "test_uq", "test_ud"])
    ap.add_argument("--tag", default="run")
    a = ap.parse_args(argv)
    res = run_mohler(a) if a.task == "mohler" else run_scientsbank(a)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"external_{a.task}_{a.tag}.json", "w"), indent=2, default=float)
    return res


if __name__ == "__main__":
    main()
