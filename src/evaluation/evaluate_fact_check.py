"""
evaluate_fact_check.py  (v2)
------------------------------
Before/after comparison of the rule-based veto layer on original (70) + challenge (85) rows.

Ground truth "should veto" = the answer asserts the WRONG binding for a point that carries
binding/cues metadata (listed explicitly below, from reading the answers - not derived from
the checker).  Every other (answer, metadata-point) pair must NOT be vetoed.

Scope:  'whole answer' = every sentence scanned (v2 apply_fact_check_text); the legacy checker
is run per sentence and a veto on any sentence counts, so the comparison is like-for-like.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import fact_binding_checker as NEW            # noqa: E402
import _legacy_fact_binding_checker as OLD    # noqa: E402

# (qid, quality_tier) -> set of point ids that SHOULD be vetoed
SHOULD_VETO = {
    ("DS01", "confidently_wrong"): {"p1", "p2"},
    ("ML01", "mixed_with_error"): {"p3"},
    ("PY01", "error_variant"): {"p1"},
    ("ST02", "error_variant"): {"p1", "p2"},
    ("DS01", "error_variant"): {"p1", "p2"},
}


def rows():
    out = []
    for p in ("data/exam_grading_dataset.csv", "data/v2/challenge_set.csv"):
        with open(ROOT / p, newline="", encoding="utf-8") as f:
            out += list(csv.DictReader(f))
    return out


def legacy_veto(answer, point):
    for s in NEW.split_sentences(answer) or [answer]:
        r = OLD.apply_fact_check(s, point)
        if r is not None and r.status == "contradicted":
            return True
    return False


def main():
    res = {"legacy": dict(tp=0, fp=0, fn=0, tn=0, fp_cases=[], fn_cases=[]),
           "v2": dict(tp=0, fp=0, fn=0, tn=0, fp_cases=[], fn_cases=[])}
    for r in rows():
        ans = r["student_answer"]
        for p in json.loads(r["rubric"]):
            if not (p.get("binding") or p.get("cues")):
                continue
            truth = p["id"] in SHOULD_VETO.get((r["qid"], r["quality_tier"]), set())
            pred = {"legacy": legacy_veto(ans, p),
                    "v2": (NEW.apply_fact_check_text(ans, p).status == "contradicted")}
            for k, v in pred.items():
                key = ("tp" if v else "fn") if truth else ("fp" if v else "tn")
                res[k][key] += 1
                if key == "fp":
                    res[k]["fp_cases"].append(f"{r['qid']}/{r['quality_tier']}/{p['id']}")
                if key == "fn":
                    res[k]["fn_cases"].append(f"{r['qid']}/{r['quality_tier']}/{p['id']}")
    for k, d in res.items():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        d["precision"] = round(tp / (tp + fp), 3) if tp + fp else None
        d["recall"] = round(tp / (tp + fn), 3) if tp + fn else None
        print(f"{k:7s} TP={tp} FP={fp} FN={fn} TN={d['tn']}  precision={d['precision']} recall={d['recall']}")
        if d["fp_cases"]: print("   false vetoes :", d["fp_cases"])
        if d["fn_cases"]: print("   missed       :", d["fn_cases"])
    out = ROOT / "results" / "v2" / "fact_check_before_after.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
