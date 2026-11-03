"""
load_external.py  (v2)
-----------------------
Loaders that turn PUBLIC, human-graded short-answer datasets into the project's row schema, so the
signals built here (NLI, bi-encoder, LLM judge) can be validated on answers NOBODY on the team wrote.

Datasets (dataset cards checked; column names below are from the cards - confirm with `head` after export):
  Mohler et al. 2011, computer-science short answers     HF: Viswa45/ASAG   (configs raw|cleaned|parsed; splits open_ended 2273 rows, close_ended 169)
        cols: id (E01.Q03.A12), question, instructor_answer, student_answer, score_grader_1, score_grader_2, score_avg (0-5)
  SemEval-2013 Task 7 SciEntsBank                        HF: nkazi/SciEntsBank   splits train 4969 | test_ua 540 | test_uq 733 | test_ud 4562
        cols: id, question, reference_answer, student_answer, label  (0 correct, 1 contradictory, 2 partially_correct_incomplete, 3 irrelevant, 4 non_domain)
        (nkazi/Beetle has the same layout)
  ASAP-SAS (Kaggle)   train.tsv: Id, EssaySet, Score1, Score2, EssayText - NO reference answers/rubrics in the file, so it is
        only usable if you add a prompt/reference per EssaySet yourself (loader provided, flagged).

There are NO per-rubric-point labels in these datasets: ground truth is the ANSWER-LEVEL grade.  Each reference answer is split into
sentences and treated as a pseudo-rubric (equal marks) so the same point-level machinery runs; evaluation is answer-level only.
"""
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

try:
    from models.text_utils import split_sentences, atomic_claims
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from text_utils import split_sentences, atomic_claims

SCI_LABEL5 = {0: "correct", 1: "contradictory", 2: "partially_correct_incomplete", 3: "irrelevant", 4: "non_domain"}
SCI_LABEL3 = {"correct": "correct", "contradictory": "contradictory"}          # everything else -> "incorrect"
SCI_SCORE = {"correct": 1.0, "partially_correct_incomplete": 0.5}             # others 0.0


def pseudo_rubric(reference: str, max_marks: float, min_words: int = 4) -> List[dict]:
    """Reference sentences -> equal-mark rubric points (tiny sentences are merged into the previous one)."""
    sents = split_sentences(reference)
    merged: List[str] = []
    for s in sents:
        if merged and len(s.split()) < min_words:
            merged[-1] += " " + s
        else:
            merged.append(s)
    if len(merged) == 1:
        merged = atomic_claims(merged[0]) or merged
    merged = merged or [reference.strip()]
    each = max_marks / len(merged)
    return [{"id": f"p{i + 1}", "text": t, "marks": each} for i, t in enumerate(merged)]


def _clean(t) -> str:
    """Mohler's export marks sentence ends with '<STOP>'; drop the marker, keep the sentence break."""
    import re
    t = re.sub(r"([^.?!\s])\s*<STOP>", r"\1.", str(t or ""))
    return " ".join(t.replace("<STOP>", " ").split())


def _row(qid, question, ref, answer, sid, score, max_marks, tier, split, **extra) -> Dict[str, str]:
    question, ref, answer = _clean(question), _clean(ref), _clean(answer)
    return {"qid": qid, "subject": "external", "question": question, "reference_answer": ref,
            "rubric": json.dumps(pseudo_rubric(ref, max_marks)), "max_marks": max_marks, "student_id": sid,
            "student_answer": answer or "", "points_covered": "[]", "human_score": score,
            "quality_tier": tier, "split": split, **extra}


def load_mohler(path) -> List[Dict[str, str]]:
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        qid = r["id"].rsplit(".", 1)[0]                       # E01.Q03.A12 -> E01.Q03  (question = grouping unit)
        rows.append(_row(qid, r["question"], r["instructor_answer"], r["student_answer"], r["id"],
                         float(r["score_avg"]), 5.0, "mohler", "external",
                         grader1=r.get("score_grader_1", ""), grader2=r.get("score_grader_2", "")))
    return rows


def load_scientsbank(path, split_name="") -> List[Dict[str, str]]:
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        l5 = SCI_LABEL5[int(r["label"])]
        l3 = SCI_LABEL3.get(l5, "incorrect")
        qid = hashlib.md5(r["question"].encode()).hexdigest()[:8]
        rows.append(_row(qid, r["question"], r["reference_answer"], r["student_answer"], r["id"],
                         2.0 * SCI_SCORE.get(l5, 0.0), 2.0, l5, split_name or "external", label5=l5, label3=l3))
    return rows


def load_asap_sas(path, references: Dict[str, str], max_marks: Dict[str, float]) -> List[Dict[str, str]]:
    """references/max_marks keyed by EssaySet ('1'..'10'); you must supply them (not in the Kaggle file)."""
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8"), delimiter="\t"):
        es = r["EssaySet"]
        if es not in references:
            continue
        rows.append(_row(f"asap{es}", f"ASAP-SAS prompt {es}", references[es], r["EssayText"], r["Id"],
                         float(r["Score1"]), float(max_marks[es]), "asap", "external"))
    return rows
