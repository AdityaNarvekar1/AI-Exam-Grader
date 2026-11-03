"""
build_blind_sheet_v2.py  (v2)
------------------------------
Builds the HUMAN VALIDATION instrument that the project has promised since Week 2 and never run.

  results/v2/human_validation/BLIND_SHEET.csv   what each grader fills in (NO scores, NO model output,
                                                 shuffled, rubric shown as marks-per-point)
  results/v2/human_validation/ANSWER_KEY.csv    item_id -> synthetic label + tier + student_id (do NOT share)

Contents: the 25 original TEST rows + a stratified sample of the challenge set (default 40; --n-challenge).
Give each grader their OWN copy of BLIND_SHEET.csv; graders must not talk to each other or look at the key.
Minimum useful: 2 graders x ~65 items (~1.5 h each).  Then:
    python src/evaluation/compute_human_agreement_v2.py --sheets r1.csv r2.csv
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rubric_text(r):
    return "\n".join(f"[{p['id']}] ({p['marks']} marks) {p['text']}" for p in json.loads(r["rubric"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-challenge", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results" / "v2" / "human_validation"))
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    orig = [r for r in csv.DictReader(open(ROOT / "data" / "exam_grading_dataset.csv", newline="", encoding="utf-8"))
            if r["split"] == "test"]
    chal = list(csv.DictReader(open(ROOT / "data" / "v2" / "challenge_set.csv", newline="", encoding="utf-8")))
    by_tier = defaultdict(list)
    for r in chal:
        by_tier[r["quality_tier"]].append(r)
    picked, tiers = [], sorted(by_tier)
    for t in tiers:                       # round-robin over tiers => every tier is represented
        rnd.shuffle(by_tier[t])
    i = 0
    while len(picked) < min(a.n_challenge, len(chal)):
        t = tiers[i % len(tiers)]
        if by_tier[t]:
            picked.append(by_tier[t].pop())
        i += 1
    items = orig + picked
    rnd.shuffle(items)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "BLIND_SHEET.csv", "w", newline="", encoding="utf-8") as f, \
         open(out / "ANSWER_KEY.csv", "w", newline="", encoding="utf-8") as g:
        w = csv.writer(f); k = csv.writer(g)
        w.writerow(["item_id", "question", "reference_answer", "rubric", "max_marks", "student_answer",
                    "grader_score", "points_you_credited (e.g. p1;p3)", "notes"])
        k.writerow(["item_id", "student_id", "qid", "quality_tier", "source", "synthetic_score", "max_marks", "points_covered"])
        for n, r in enumerate(items, 1):
            iid = f"H{n:03d}"
            w.writerow([iid, r["question"], r["reference_answer"], rubric_text(r), r["max_marks"], r["student_answer"], "", "", ""])
            k.writerow([iid, r["student_id"], r["qid"], r["quality_tier"], r["split"], r["human_score"], r["max_marks"],
                        ";".join(json.loads(r["points_covered"]))])
    print(f"{len(items)} items ({len(orig)} original test + {len(picked)} challenge) -> {out}")


if __name__ == "__main__":
    main()
