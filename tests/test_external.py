"""External-data plumbing tests on tiny FAKE files (stub NLI/bi-encoder; no numbers here are results)."""
import csv, json, os, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/evaluation", "src/models", "src/data"):
    sys.path.insert(0, str(ROOT / sub))
os.environ["G2_NLI_MODEL"] = "stub"; os.environ["G2_BI_MODEL"] = "stub"
import load_external as L          # noqa: E402
import run_external_benchmark as R  # noqa: E402


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


@pytest.fixture
def mohler_csv(tmp_path):
    ref = "A stack is last in first out. A queue is first in first out."
    rows = []
    for q in range(8):
        for a in range(6):
            good = a % 2 == 0
            ans = ref if good else "Data structures store data."
            rows.append([f"E01.Q{q:02d}.A{a:02d}", f"Question {q}?", ref, ans, 5 if good else 1, 5 if good else 0, 5 if good else 0.5])
    p = tmp_path / "mohler.csv"
    write(p, ["id", "question", "instructor_answer", "student_answer", "score_grader_1", "score_grader_2", "score_avg"], rows)
    return p


def test_pseudo_rubric_marks_sum_to_max():
    pr = L.pseudo_rubric("A stack is LIFO. A queue is FIFO. Both are linear.", 5.0)
    assert abs(sum(p["marks"] for p in pr) - 5.0) < 1e-9 and len(pr) >= 2
    one = L.pseudo_rubric("A list is mutable, while a tuple is immutable and cannot be changed", 4.0)
    assert abs(sum(p["marks"] for p in one) - 4.0) < 1e-9


def test_load_mohler_groups_by_question(mohler_csv):
    rows = L.load_mohler(mohler_csv)
    assert len(rows) == 48 and len({r["qid"] for r in rows}) == 8 and rows[0]["max_marks"] == 5.0
    assert json.loads(rows[0]["rubric"])


def test_load_scientsbank_label_mapping(tmp_path):
    p = tmp_path / "s.csv"
    write(p, ["id", "question", "reference_answer", "student_answer", "label"],
          [["a", "q1", "ref one.", "ans", 0], ["b", "q1", "ref one.", "ans", 1], ["c", "q1", "ref one.", "ans", 2],
           ["d", "q2", "ref two.", "ans", 3], ["e", "q2", "ref two.", "ans", 4]])
    rows = L.load_scientsbank(p, "train")
    assert [r["label3"] for r in rows] == ["correct", "contradictory", "incorrect", "incorrect", "incorrect"]
    assert [r["human_score"] for r in rows] == [2.0, 0.0, 1.0, 0.0, 0.0]


def test_mohler_benchmark_runs_end_to_end_with_stubs(mohler_csv, tmp_path, capsys):
    res = R.main(["--task", "mohler", "--data", str(mohler_csv), "--graders", "const", "length", "refsim", "lex", "bi", "nli", "nli_bi",
                  "--folds", "4", "--tag", "pytest"])
    assert set(res) == {"const", "length", "refsim", "lex", "bi", "nli", "nli_bi"}
    assert res["lex"]["pearson_r"] > 0.5            # the fake task is trivially separable
    (ROOT / "results" / "v2" / "external_mohler_pytest.json").unlink()


def test_scientsbank_benchmark_runs_with_stubs(tmp_path):
    ref = "The bulb lights because the circuit is closed."
    def mk(n, off):
        return [[f"id{off + i}", f"q{i % 5}", ref, ref if i % 3 == 0 else ("The circuit is not closed so the bulb is lit." if i % 3 == 1 else "Batteries are cells."), i % 3 if i % 3 != 2 else 3] for i in range(n)]
    write(tmp_path / "scientsbank_train.csv", ["id", "question", "reference_answer", "student_answer", "label"], mk(60, 0))
    write(tmp_path / "scientsbank_test_ua.csv", ["id", "question", "reference_answer", "student_answer", "label"], mk(30, 100))
    res = R.main(["--task", "scientsbank", "--dir", str(tmp_path), "--graders", "lex", "nli", "--tests", "test_ua", "--tag", "pytest"])
    assert "macro_F1" in res["nli"]["test_ua"] and "AUROC_contradictory_zero_shot(contra_max)" in res["nli"]["test_ua"]
    (ROOT / "results" / "v2" / "external_scientsbank_pytest.json").unlink()
