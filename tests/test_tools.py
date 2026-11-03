import sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tools")); sys.path.insert(0, str(ROOT / "src" / "models"))
import rubric_lint as L  # noqa: E402

def codes(rub, mx=5, ref="A list is mutable and a tuple is immutable."):
    return {i.code for i in L.lint("q", ref, rub, mx)}

def test_clean_rubric_has_no_errors():
    rub = [{"id": "p1", "text": "A list is mutable, meaning it can be changed after creation", "marks": 2.5},
           {"id": "p2", "text": "A tuple is immutable and cannot be modified once created", "marks": 2.5}]
    assert not ({"ERROR"} & {i.level for i in L.lint("q", "A list is mutable and a tuple is immutable. Lists can be changed after creation; tuples cannot be modified once created.", rub, 5)})

def test_sum_exceeds_and_fragment_and_dominates():
    c = codes([{"id": "p1", "text": "mutability", "marks": 8}], mx=5)
    assert {"SUM_EXCEEDS_MAX", "FRAGMENT"} <= c
    c2 = codes([{"id": "p1", "text": "A list is mutable and can be changed freely", "marks": 4},
                {"id": "p2", "text": "Tuples are immutable and cannot be changed", "marks": 1}])
    assert "SINGLE_POINT_DOMINATES" in c2

def test_meta_and_duplicate_and_unsupported():
    rub = [{"id": "p1", "text": "A correct example is given that shows the idea", "marks": 2},
           {"id": "p2", "text": "A stack follows last in first out order", "marks": 1.5},
           {"id": "p3", "text": "A stack follows a last in first out order", "marks": 1.5}]
    c = codes(rub, 5, ref="Stacks are LIFO.")
    assert {"META_POINT", "NEAR_DUPLICATE", "NOT_SUPPORTED_BY_REFERENCE"} <= c

def test_dataset_lint_flags_known_ds01_duplicate_points():
    r = [x for x in csv.DictReader(open(ROOT / "data" / "exam_grading_dataset.csv", encoding="utf-8")) if x["qid"] == "DS01"][0]
    iss = L.lint(r["question"], r["reference_answer"], json.loads(r["rubric"]), float(r["max_marks"]))
    assert any(i.code == "NEAR_DUPLICATE" for i in iss)


def test_blind_sheet_and_agreement_roundtrip(tmp_path, monkeypatch):
    import importlib, random
    sys.path.insert(0, str(ROOT / "src" / "evaluation"))
    import build_blind_sheet_v2 as B
    import compute_human_agreement_v2 as H
    B.main.__globals__  # noqa
    out = tmp_path / "hv"
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--n-challenge", "20"])
    B.main()
    sheet = list(csv.DictReader(open(out / "BLIND_SHEET.csv", encoding="utf-8")))
    key = {r["item_id"]: r for r in csv.DictReader(open(out / "ANSWER_KEY.csv", encoding="utf-8"))}
    assert len(sheet) == 45 and all("grader_score" in r and r["grader_score"] == "" for r in sheet)
    # no leakage of labels/tiers into the blind sheet
    header = " ".join(csv.reader(open(out / "BLIND_SHEET.csv", encoding="utf-8")).__next__()).lower()
    assert "tier" not in header and "synthetic" not in header
    rnd = random.Random(0)
    paths = []
    for k in range(2):
        p = tmp_path / f"r{k}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0].keys())); w.writeheader()
            for r in sheet:
                s = float(key[r["item_id"]]["synthetic_score"]) + rnd.choice([-0.5, 0, 0, 0.5])
                w.writerow({**r, "grader_score": max(0, min(float(r["max_marks"]), s))})
        paths.append(str(p))
    H.D = out
    rep = H.main(["--sheets", *paths, "--key", str(out / "ANSWER_KEY.csv"), "--out", str(out / "rep.json")])
    assert rep["pairs"]["rater1_vs_rater2"]["QWK"] > 0.9 and rep["mean_human_vs_synthetic"]["n"] == 45
