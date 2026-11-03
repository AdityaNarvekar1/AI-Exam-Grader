"""
ocr_noise_robustness.py  (Week 7)
----------------------------------
How fast does each grader degrade as OCR gets worse?  The real scripts give only 62 OCR'd answers
(median character error rate ~6%), too few to draw a curve, so this injects SYNTHETIC OCR noise
into the 85-row challenge set at controlled character error rates and re-grades it.

Noise model (shaped on the errors seen in Group 1's real output, e.g. "Cradient", "Kemel",
"SUM" for SVM, "outful", dropped words, lost second page):
  * visually-confusable substitutions (l/1/I, o/0, rn/m, e/c, a/o, u/v, n/h, S/5 ...)
  * random character deletions / insertions
  * whole-word drops
  * optional truncation: the last 40% of the answer is lost (a continuation page that was not merged)

Protocol: fit every grader on the CLEAN original 70 rows, test on the challenge set at each noise
level (protocol B), so the only thing that changes between columns is the noise.  Also reports how often
each grader's own review flag fires, to test the Week-6 claim that grader confidence cannot stand in
for OCR confidence.

  python src/week7/ocr_noise_robustness.py --graders tfidf lexical spacy_hybrid nli_bi
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "src/evaluation", "src/models"):
    sys.path.insert(0, str(ROOT / sub))
from eval_utils import metrics, cluster_bootstrap  # noqa: E402
import graders as G  # noqa: E402

CONFUSE = {"l": "1I", "1": "lI", "I": "l1", "o": "0a", "0": "o", "e": "c", "c": "e", "a": "o", "u": "v",
           "v": "u", "n": "h", "h": "n", "S": "5", "5": "S", "i": "l", "t": "f", "g": "q", "m": "rn", "B": "8"}


def corrupt(text: str, rate: float, rng: random.Random, truncate: bool = False) -> str:
    if not text:
        return text
    words = text.split()
    if truncate and len(words) > 4:
        words = words[:max(2, int(len(words) * 0.6))]
    out_words = []
    for w in words:
        if rate and rng.random() < rate * 0.25:          # word drop
            continue
        chars = []
        for ch in w:
            r = rng.random()
            if r < rate * 0.55 and ch in CONFUSE:
                chars.append(rng.choice(CONFUSE[ch]))
            elif r < rate * 0.75:
                continue                                   # deletion
            elif r < rate * 0.85:
                chars.append(ch + rng.choice("abcdefghijklmnopqrstuvwxyz"))
            else:
                chars.append(ch)
        out_words.append("".join(chars))
    return " ".join(out_words)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--graders", nargs="+", default=["length", "tfidf", "lexical", "spacy_hybrid", "nli_bi"])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    G.load_optional_adapters()
    load = lambda p: list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    orig = load(ROOT / "data" / "exam_grading_dataset.csv")
    chal = load(ROOT / "data" / "v2" / "challenge_set.csv")
    conds = [(f"cer~{r:.2f}", r, False) for r in a.rates] + [("truncated_40pct", 0.0, True), ("cer~0.10+truncated", 0.10, True)]
    sets = {}
    for name, r, tr in conds:
        rng = random.Random(a.seed)
        sets[name] = [dict(row, student_answer=corrupt(row["student_answer"], r, rng, tr)) for row in chal]
    # measured CER of the injected noise
    from run_week7_real_exam import cer
    measured = {n: round(float(np.mean([cer(x["student_answer"], c["student_answer"]) for x, c in zip(s, chal)
                                       if c["student_answer"]])), 3) for n, s in sets.items()}
    y = [float(r["human_score"]) for r in chal]
    mx = [float(r["max_marks"]) for r in chal]
    groups = [r["qid"] for r in chal]
    res = {"measured_CER": measured, "graders": {}}
    for g in a.graders:
        if g not in G.REGISTRY:
            print("skip", g)
            continue
        print(f"[noise] {g}", flush=True)
        ad = G.REGISTRY[g]().fit(orig)
        res["graders"][g] = {}
        for name, s in sets.items():
            out = ad.predict(s)
            p = [o.score for o in out]
            m = metrics(y, p, mx)
            m["MAE_ci95"] = cluster_bootstrap(np.abs(np.array(y) - np.array(p)), groups)
            flags = [bool((o.extra or {}).get("needs_review")) for o in out]
            if any(o.confidence is not None for o in out):
                m["mean_confidence"] = round(float(np.mean([o.confidence for o in out if o.confidence is not None])), 3)
            m["review_flag_rate"] = round(float(np.mean(flags)), 3) if any("needs_review" in (o.extra or {}) for o in out) else None
            res["graders"][g][name] = m
            print(f"   {name:22s} MAE {m['MAE']:.3f}  QWK {m['QWK']:.3f}  flag {m['review_flag_rate']}", flush=True)
    out = ROOT / "results" / "week7"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "ocr_noise_robustness.json", "w"), indent=2, default=float)
    ex = random.Random(1)
    print("example:", corrupt(chal[3]["student_answer"], 0.10, ex)[:200])
    return res


if __name__ == "__main__":
    main()
