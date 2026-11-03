"""
rubric_lint.py  (v2)
---------------------
The Week-6 report found that rubric WORDING can flip a grading decision, and the API docs told rubric
authors to "write full sentences".  This turns that advice into an automatic check, so a rubric is
validated BEFORE any answer is graded.  No models needed.

  python src/tools/rubric_lint.py --dataset data/exam_grading_dataset.csv
  python src/tools/rubric_lint.py question.json          # {question, reference_answer, rubric, maximum_score}

Severity:  ERROR (grading will be wrong/unsafe)  >  WARN (likely to hurt accuracy)  >  INFO (suggestion)
"""
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from text_utils import is_meta_point, atomic_claims          # noqa: E402
from lexical_rubric_grader import stems                      # noqa: E402

VERBISH = re.compile(r"\b(is|are|was|were|be|has|have|had|can|cannot|means?|returns?|uses?|used|occurs?|requires?|"
                     r"follows?|stores?|allows?|ensures?|reduces?|increases?|adds?|shrinks?|trains?|combines?|"
                     r"produces?|performs?|explains?|states?|holds?|becomes?|persists?|gives?|avoids?|"
                     r"tends?|matters?|approaches|computes?|checks?|halves|executes|moves|do(es)?)\b", re.I)
CONTRAST = re.compile(r"\b(while|whereas|unlike|versus|vs\.?|in contrast|rather than|instead of)\b", re.I)


@dataclass
class Issue:
    level: str
    code: str
    point: str
    message: str


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def lint(question: str, reference: str, rubric: List[dict], max_marks: float) -> List[Issue]:
    out: List[Issue] = []
    total = sum(float(p.get("marks", 0)) for p in rubric)
    if not rubric:
        return [Issue("ERROR", "EMPTY_RUBRIC", "-", "Rubric has no points.")]
    if total > max_marks + 1e-6:
        out.append(Issue("ERROR", "SUM_EXCEEDS_MAX", "-", f"Rubric marks sum to {total} > maximum {max_marks}."))
    elif total < max_marks - 1e-6:
        out.append(Issue("WARN", "UNALLOCATED_MARKS", "-", f"Only {total} of {max_marks} marks are allocated to points."))
    ids = [p["id"] for p in rubric]
    if len(set(ids)) != len(ids):
        out.append(Issue("ERROR", "DUPLICATE_IDS", "-", "Rubric point ids are not unique."))
    ref_stems = set(stems(reference or ""))
    for p in rubric:
        pid, text, marks = p["id"], p.get("text", ""), float(p.get("marks", 0))
        words = text.split()
        if marks <= 0:
            out.append(Issue("ERROR", "NON_POSITIVE_MARKS", pid, "Marks must be > 0."))
        if len(words) < 6 or not VERBISH.search(text):
            out.append(Issue("WARN", "FRAGMENT", pid, "Looks like a fragment. Write a full assertable sentence "
                             "('A list is mutable ...'), not a topic label - fragments made decisions flip in Week 6."))
        if len(words) > 35:
            out.append(Issue("WARN", "TOO_LONG", pid, f"{len(words)} words; split into separate points."))
        if len(rubric) > 1 and marks / max_marks > 0.5:
            out.append(Issue("WARN", "SINGLE_POINT_DOMINATES", pid, f"Carries {marks / max_marks:.0%} of the marks; "
                             "one wrong credit decision swings the score by that much."))
        claims = atomic_claims(text)
        if len(claims) >= 3:
            out.append(Issue("WARN", "COMPOUND_POINT", pid, f"Contains ~{len(claims)} separate claims; split it so partial "
                             "answers get partial credit instead of all-or-nothing."))
        if is_meta_point(text):
            out.append(Issue("WARN", "META_POINT", pid, "Describes the answer ('an example is given') instead of asserting a "
                             "fact; NLI/embedding scorers cannot verify it. Prefer the LLM judge for this point, or state "
                             "the concrete example expected."))
        if CONTRAST.search(text) and not (p.get("binding") or p.get("cues")):
            out.append(Issue("INFO", "CONTRAST_WITHOUT_BINDING", pid, "States an X-vs-Y contrast; an inverted answer "
                             "('X is the wrong one') can look similar. Rely on NLI/LLM contradiction checks, or add "
                             "binding metadata."))
        ps = set(stems(text))
        if ref_stems and ps and len(ps & ref_stems) / len(ps) < 0.5:
            out.append(Issue("WARN", "NOT_SUPPORTED_BY_REFERENCE", pid, "Less than half of this point's content words occur "
                             "in the reference answer; rubric and reference may disagree."))
    for i in range(len(rubric)):
        for j in range(i + 1, len(rubric)):
            if jaccard(stems(rubric[i]["text"]), stems(rubric[j]["text"])) >= 0.6:
                out.append(Issue("WARN", "NEAR_DUPLICATE", f"{rubric[i]['id']}/{rubric[j]['id']}",
                                 "Points are near-duplicates; one answer sentence would be credited for both."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", nargs="?")
    ap.add_argument("--dataset")
    a = ap.parse_args()
    items = []
    if a.dataset:
        seen = set()
        for r in csv.DictReader(open(a.dataset, newline="", encoding="utf-8")):
            if r["qid"] not in seen:
                seen.add(r["qid"])
                items.append((r["qid"], r["question"], r["reference_answer"], json.loads(r["rubric"]), float(r["max_marks"])))
    elif a.json_file:
        d = json.load(open(a.json_file))
        for k, x in enumerate(d if isinstance(d, list) else [d]):
            items.append((x.get("qid", f"item{k}"), x["question"], x["reference_answer"], x["rubric"], float(x["maximum_score"])))
    else:
        ap.error("give a JSON file or --dataset")
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for qid, q, ref, rub, mx in items:
        issues = lint(q, ref, rub, mx)
        print(f"\n[{qid}] {q}")
        if not issues:
            print("   OK")
        for i in issues:
            counts[i.level] += 1
            print(f"   {i.level:5s} {i.code:26s} {i.point:8s} {i.message}")
    print("\nTOTAL:", counts)
    sys.exit(1 if counts["ERROR"] else 0)


if __name__ == "__main__":
    main()
