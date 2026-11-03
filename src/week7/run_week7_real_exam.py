"""
run_week7_real_exam.py  (Week 7)  -  Group 1 -> Group 2 on REAL handwritten scripts
-------------------------------------------------------------------------------------
Input (all in data/group1_real/):
  ocr_output_5_students.csv   Group 1's real OCR export: 5 students x 35 questions (20 MCQ + 15 short)
  answer_key.csv              faculty key (MCQ letters, one-line short-answer keys)
  exam_key_auto.json / exam_key_authored.json   built by build_exam_key.py
  scan_ground_truth.csv       teacher's red-pen marks + manual transcriptions, read off the scans
                              (built by build_scan_ground_truth.py)

What it measures
  1. MCQ: OCR letter grading vs the teacher's MCQ sub-totals.
  2. Short answers, every grader, two rubrics (auto / authored), two texts:
       clean = manual transcription of the handwriting   -> how good is the GRADER on real answers?
       ocr   = exactly what Group 1 delivered             -> how good is the PIPELINE?
     Ground truth = the teacher's mark (0-2).  Graders are trained ONLY on the synthetic data
     (original 70 + challenge 85); nothing is fitted on the teacher's marks ("zero-shot").
  3. "Calibrated to this teacher": a monotone (isotonic) map from each grader's predicted fraction to
     the teacher's mark, fitted leave-one-question-out on the real answers (never sees the held-out
     question).  Answers "is the gap a ranking problem or a strictness/scale problem?"
  4. OCR quality (CER/WER vs transcription, missed / misplaced / truncated segments) and how much
     OCR damage changes each grader's score.
  5. Review routing: do Group 1's confidence / needs_review flags catch the answers OCR damaged?

  python src/week7/run_week7_real_exam.py                     # all graders whose deps are installed
  python src/week7/run_week7_real_exam.py --graders const length tfidf lexical
Outputs: results/week7/real_exam_report.json, real_exam_rows.csv, real_exam_summary.md
"""
import argparse
import csv
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "src/evaluation", "src/models", "src/integration"):
    sys.path.insert(0, str(ROOT / sub))

from eval_utils import cluster_bootstrap, paired_cluster_bootstrap, qwk  # noqa: E402
import graders as G  # noqa: E402
from group1_adapter import load_group1_csv, merge_records, grade_mcq  # noqa: E402

D = ROOT / "data" / "group1_real"
OUT = ROOT / "results" / "week7"
SHORT_Q = [f"Q{i}" for i in range(21, 36)]
MCQ_Q = [f"Q{i}" for i in range(1, 21)]
DEFAULT_GRADERS = ["const", "length", "tfidf", "lexical", "spacy_embedding", "spacy_hybrid",
                   "feat_lex", "bi", "nli", "nli_bi", "nli_bi_exp"]


# ------------------------------------------------------------------------------------ helpers
def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(t):
    return " ".join((t or "").lower().split())


def cer(hyp, ref):
    r = norm(ref)
    return lev(norm(hyp), r) / max(1, len(r))


def wer(hyp, ref):
    r = norm(ref).split()
    return lev(norm(hyp).split(), r) / max(1, len(r))


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a).split(), norm(b).split()).ratio()


def mstats(y, p, groups, mx=2.0):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    pr = np.clip(np.rint(p), 0, mx)
    from scipy.stats import spearmanr
    rho = spearmanr(y, p)[0] if len(set(p)) > 1 and len(set(y)) > 1 else float("nan")
    return {"n": int(len(y)), "MAE": round(float(err.mean()), 3),
            "MAE_ci95": cluster_bootstrap(err, groups),
            "exact_agree_rounded": round(float((pr == y).mean()), 3),
            "within_1_mark": round(float((err <= 1 + 1e-9).mean()), 3),
            "QWK_0_1_2": round(qwk(y, p, [mx] * len(y), levels=int(mx)), 3),
            "spearman": None if np.isnan(rho) else round(float(rho), 3),
            "mean_pred": round(float(p.mean()), 3), "mean_teacher": round(float(y.mean()), 3)}


def loqo_isotonic(frac, y, groups):
    """Leave-one-question-out monotone calibration of predicted fraction -> teacher mark."""
    from sklearn.isotonic import IsotonicRegression
    frac, y, groups = np.asarray(frac, float), np.asarray(y, float), np.asarray(groups)
    out = np.zeros(len(y))
    for q in np.unique(groups):
        tr = groups != q
        m = IsotonicRegression(y_min=0, y_max=2, increasing=True, out_of_bounds="clip").fit(frac[tr], y[tr])
        out[~tr] = m.predict(frac[~tr])
    return out


def loqo_const(y, groups):
    y, groups = np.asarray(y, float), np.asarray(groups)
    return np.array([y[groups != g].mean() for g in groups])


def loqo_length(texts, y, groups):
    from sklearn.linear_model import LinearRegression
    X = np.log1p([[len((t or "").split())] for t in texts])
    y, groups = np.asarray(y, float), np.asarray(groups)
    out = np.zeros(len(y))
    for q in np.unique(groups):
        tr = groups != q
        out[~tr] = np.clip(LinearRegression().fit(X[tr], y[tr]).predict(X[~tr]), 0, 2)
    return out


def load_csv(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8-sig")))


# ---------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--graders", nargs="+", default=DEFAULT_GRADERS)
    ap.add_argument("--keys", nargs="+", default=["authored", "auto"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    G.load_optional_adapters()
    graders = [g for g in a.graders if g in G.REGISTRY]
    skipped = [g for g in a.graders if g not in G.REGISTRY]

    keys = {k: json.load(open(D / f"exam_key_{k}.json")) for k in a.keys}
    raw = load_group1_csv(D / "ocr_output_5_students.csv")
    ocr = merge_records(raw)                                   # (sid, q) -> merged record
    raw_by = {(r["student_id"], r["question"]): r for r in raw}
    gt = {(r["student_id"], r["question"]): r for r in load_csv(D / "scan_ground_truth.csv")}
    students = sorted({k[0] for k in gt})
    report = {"graders_run": graders, "graders_unavailable": skipped}

    # ----------------------------------------------------------------- 1. MCQ
    teacher_tot = {"Student_11": (40, 10), "Student_23": (23, 10), "Student_34": (33, 13),
                   "Student_35": (25, 11), "Student_50": (35, 13)}
    key0 = keys[a.keys[0]]
    mcq = {}
    for s in students:
        auto = sum(grade_mcq(key0[q], ocr[(s, q)]["answer_text"])["score"] for q in MCQ_Q)
        alt = dict(key0["Q1"], correct_option="D")
        auto_q1d = auto - grade_mcq(key0["Q1"], ocr[(s, "Q1")]["answer_text"])["score"] + \
            grade_mcq(alt, ocr[(s, "Q1")]["answer_text"])["score"]
        mcq[s] = {"teacher": teacher_tot[s][1], "auto_csv_key": auto, "auto_if_Q1_is_D": auto_q1d,
                  "ocr_Q1_letter": ocr[(s, "Q1")]["answer_text"]}
    report["mcq"] = {"per_student": mcq,
                     "exact_match_students_csv_key": sum(v["auto_csv_key"] == v["teacher"] for v in mcq.values()),
                     "exact_match_students_Q1_D": sum(v["auto_if_Q1_is_D"] == v["teacher"] for v in mcq.values()),
                     "note": "Teacher ticks D for Q1 on every script; answer_key.csv says B. Student_11 wrote C "
                             "(OCR read D), so with key D the OCR error would award her a mark she did not earn."}

    # ----------------------------------------------------------------- 2. short-answer items
    items = []
    for s in students:
        for q in SHORT_Q:
            g, o, r = gt[(s, q)], ocr[(s, q)], raw_by[(s, q)]
            items.append({"student_id": s, "question": q, "attempted": g["attempted"] == "yes",
                          "teacher": float(g["teacher_mark"]) if g["teacher_mark"] != "" else 0.0,
                          "mark_reading": g["mark_reading"], "clean": g["manual_transcription"],
                          "ocr": o["answer_text"], "ocr_status": o["ocr_status"], "ocr_conf": o["ocr_confidence"],
                          "g1_review": o["group1_needs_review"], "placed_by_content": o["placed_by_content"]})
    ev = [it for it in items if it["attempted"]]
    phantom = [it for it in items if not it["attempted"] and it["ocr_status"] == "found"]
    report["counts"] = {"short_answer_slots": len(items), "attempted_per_scan": len(ev),
                        "not_attempted": len(items) - len(ev), "ocr_found_but_not_attempted": len(phantom),
                        "teacher_mark_distribution": {str(m): sum(it["teacher"] == m for it in ev) for m in (0, 1, 2)},
                        "marks_inferred_from_total": sum(it["mark_reading"] == "inferred_from_total" for it in ev)}

    # ----------------------------------------------------------------- 4. OCR quality (before grading)
    by_student = defaultdict(dict)
    for it in ev:
        by_student[it["student_id"]][it["question"]] = it["clean"]
    for it in ev:
        it["missed"] = it["ocr_status"] == "not_found"
        if it["missed"]:
            it.update(cer=1.0, wer=1.0, len_ratio=0.0, misplaced=False)
            continue
        it["cer"], it["wer"] = round(cer(it["ocr"], it["clean"]), 3), round(wer(it["ocr"], it["clean"]), 3)
        it["len_ratio"] = round(len(norm(it["ocr"]).split()) / max(1, len(norm(it["clean"]).split())), 2)
        sims = {q: sim(it["ocr"], t) for q, t in by_student[it["student_id"]].items()}
        best = max(sims, key=sims.get)
        it["misplaced"] = best != it["question"] and sims[best] > sims[it["question"]] + 0.05
    found = [it for it in ev if not it["missed"]]
    it_bad = lambda it: it["missed"] or it["misplaced"] or it["cer"] > 0.15 or it["len_ratio"] < 0.75
    report["ocr_quality"] = {
        "n_attempted": len(ev), "n_missed_not_found": sum(it["missed"] for it in ev),
        "n_found": len(found),
        "median_CER_found": round(float(np.median([it["cer"] for it in found])), 3),
        "mean_CER_found": round(float(np.mean([it["cer"] for it in found])), 3),
        "median_WER_found": round(float(np.median([it["wer"] for it in found])), 3),
        "mean_WER_found": round(float(np.mean([it["wer"] for it in found])), 3),
        "n_truncated_len_ratio_lt_0.75": sum(it["len_ratio"] < 0.75 for it in found),
        "n_misplaced": sum(it["misplaced"] for it in found),
        "n_bad_any": sum(it_bad(it) for it in ev),
        "missed_items": [f'{it["student_id"]}/{it["question"]} (teacher {it["teacher"]:g})' for it in ev if it["missed"]],
        "misplaced_items": [f'{it["student_id"]}/{it["question"]}' for it in found if it["misplaced"]],
        "truncated_items": [f'{it["student_id"]}/{it["question"]} ({it["len_ratio"]})' for it in found if it["len_ratio"] < 0.75],
        "MCQ_letters_note": "MCQ letters checked against the scans: 99 of 100 match; Student_11 Q1 wrote C, OCR read D."}

    # routing: which flags catch bad OCR?
    def pr(flag):
        tp = sum(flag(it) and it_bad(it) for it in ev); fp = sum(flag(it) and not it_bad(it) for it in ev)
        fn = sum((not flag(it)) and it_bad(it) for it in ev)
        return {"flagged": tp + fp, "caught_bad": tp, "missed_bad": fn,
                "precision": round(tp / (tp + fp), 2) if tp + fp else None, "recall": round(tp / (tp + fn), 2) if tp + fn else None}
    report["ocr_review_routing"] = {
        "definition_of_bad_ocr": "not found, misplaced, CER > 0.15, or < 75% of the words",
        "ocr_conf_lt_0.8_or_not_found": pr(lambda it: it["missed"] or it["ocr_conf"] < 0.8),
        "ocr_conf_lt_0.9_or_not_found": pr(lambda it: it["missed"] or it["ocr_conf"] < 0.9),
        "group1_needs_review_or_not_found": pr(lambda it: it["missed"] or it["g1_review"]),
        "any_group1_signal": pr(lambda it: it["missed"] or it["g1_review"] or it["placed_by_content"] or it["ocr_conf"] < 0.8)}

    # ----------------------------------------------------------------- 3. grade
    pooled = load_csv(ROOT / "data" / "exam_grading_dataset.csv") + load_csv(ROOT / "data" / "v2" / "challenge_set.csv")
    groups = [it["question"] for it in ev]
    y = [it["teacher"] for it in ev]

    def rows_for(keyname, cond):
        k = keys[keyname]
        return [{"qid": it["question"], "question": k[it["question"]]["question"],
                 "reference_answer": k[it["question"]]["reference_answer"],
                 "rubric": json.dumps(k[it["question"]]["rubric"]), "max_marks": 2.0,
                 "student_id": f'{it["student_id"]}_{it["question"]}', "student_answer": it[cond],
                 "points_covered": "[]", "human_score": it["teacher"], "quality_tier": "real", "split": "real"}
                for it in ev]

    preds = {}   # (grader, key, cond) -> np.array
    for gname in graders:
        print(f"[week7] fitting {gname} on {len(pooled)} synthetic rows", flush=True)
        try:
            ad = G.REGISTRY[gname]().fit(pooled)
        except Exception as e:  # pragma: no cover
            print(f"[week7] {gname} failed to fit: {e}")
            report.setdefault("graders_failed", {})[gname] = str(e)
            continue
        for kn in a.keys:
            for cond in ("clean", "ocr"):
                rows = rows_for(kn, cond)
                out = ad.predict(rows)
                p = np.array([o.score for o in out], float)
                p[[not (it[cond] or "").strip() for it in ev]] = 0.0      # empty answer = 0, whatever the grader says
                preds[(gname, kn, cond)] = p
        print(f"[week7] {gname} done", flush=True)

    res = {}
    for (gname, kn, cond), p in list(preds.items()):
        tag = f"{gname}|{kn}|{cond}"
        zs = mstats(y, p, groups)
        cal = loqo_isotonic(p / 2.0, y, groups)
        res[tag] = {"zero_shot": zs, "calibrated_loqo": mstats(y, cal, groups)}
        preds[(gname + "+cal", kn, cond)] = cal
    for cond in ("clean", "ocr"):
        texts = [it[cond] for it in ev]
        c = loqo_const(y, groups)
        L = loqo_length(texts, y, groups)
        L[[not (t or "").strip() for t in texts]] = 0.0
        res[f"teacher_mean_loqo|-|{cond}"] = {"calibrated_loqo": mstats(y, c, groups)}
        res[f"length_loqo|-|{cond}"] = {"calibrated_loqo": mstats(y, L, groups)}
        preds[("teacher_mean_loqo", "-", cond)] = c
        preds[("length_loqo", "-", cond)] = L
    report["short_answer_results"] = res

    # OCR effect: paired difference in absolute error (ocr - clean); + => OCR made it worse
    eff = {}
    yv = np.asarray(y)
    for (gname, kn, cond), p in list(preds.items()):
        if cond != "ocr" or (gname, kn, "clean") not in preds:
            continue
        pc = preds[(gname, kn, "clean")]
        e_ocr, e_clean = np.abs(yv - p), np.abs(yv - pc)
        d = paired_cluster_bootstrap(e_ocr, e_clean, groups)
        eff[f"{gname}|{kn}"] = {"MAE_clean": round(float(e_clean.mean()), 3), "MAE_ocr": round(float(e_ocr.mean()), 3),
                                "MAE_increase_from_OCR": d, "mean_abs_score_shift": round(float(np.abs(p - pc).mean()), 3),
                                "items_shift_ge_1_mark": int((np.abs(p - pc) >= 1).sum())}
    report["ocr_effect_on_scores"] = eff

    # student totals (teacher vs pipeline) for the best calibrated grader on OCR text
    def totals(p):
        t = defaultdict(float)
        for it, v in zip(ev, p):
            t[it["student_id"]] += float(v)
        return t
    tt = totals(y)
    report["student_short_totals"] = {"teacher": {s: tt[s] for s in students}}
    for name in [f"{g}|authored|ocr" for g in graders] + [f"{g}+cal|authored|ocr" for g in graders]:
        g_, k_, c_ = name.split("|")
        if (g_, k_, c_) in preds:
            pt = totals(preds[(g_, k_, c_)])
            report["student_short_totals"][name] = {s: round(pt[s], 1) for s in students}

    # rows
    with open(OUT / "real_exam_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = sorted(preds)
        w.writerow(["student_id", "question", "teacher", "mark_reading", "ocr_status", "ocr_conf", "g1_review",
                    "placed_by_content", "cer", "wer", "len_ratio", "misplaced"] + ["|".join(c) for c in cols])
        for i, it in enumerate(ev):
            w.writerow([it["student_id"], it["question"], it["teacher"], it["mark_reading"], it["ocr_status"], it["ocr_conf"],
                        it["g1_review"], it["placed_by_content"], it["cer"], it["wer"], it["len_ratio"], it["misplaced"]] +
                       [round(float(preds[c][i]), 3) for c in cols])
    json.dump(report, open(OUT / "real_exam_report.json", "w"), indent=2, default=float)
    write_md(report)
    print(f"wrote {OUT}/real_exam_report.json, real_exam_rows.csv, real_exam_summary.md")
    return report


def write_md(rep):
    L = ["# Week 7 - real scripts from Group 1 (5 students, teacher marks from the scans)", ""]
    L.append(f"Graders run: {', '.join(rep['graders_run'])}" + (f"; unavailable: {', '.join(rep['graders_unavailable'])}" if rep['graders_unavailable'] else ""))
    L += ["", "## OCR quality", "```", json.dumps(rep["ocr_quality"], indent=1), "```", "",
          "## Short answers (teacher mark 0-2, n = attempted answers)", "",
          "| grader | rubric | text | MAE zero-shot | QWK | exact | Spearman | MAE calibrated (LOQO) | QWK cal | exact cal |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in rep["short_answer_results"].items():
        g, k, c = tag.split("|")
        z, cal = r.get("zero_shot"), r["calibrated_loqo"]
        L.append(f"| {g} | {k} | {c} | {z['MAE'] if z else '-'} | {z['QWK_0_1_2'] if z else '-'} | {z['exact_agree_rounded'] if z else '-'} | "
                 f"{z['spearman'] if z else cal['spearman']} | {cal['MAE']} | {cal['QWK_0_1_2']} | {cal['exact_agree_rounded']} |")
    L += ["", "## MCQ", "```", json.dumps(rep["mcq"], indent=1), "```"]
    (OUT / "real_exam_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
