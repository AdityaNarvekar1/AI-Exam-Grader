"""Runs src/tools/rubric_lint.lint over every short-answer entry of an exam key and writes a text report."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tools"))
from rubric_lint import lint  # noqa: E402


def lint_key(key: dict, out_path) -> dict:
    counts, lines = {"ERROR": 0, "WARN": 0, "INFO": 0}, []
    for q, e in key.items():
        if e.get("question_type") == "mcq":
            continue
        issues = lint(e["question"], e["reference_answer"], e["rubric"], e["maximum_score"])
        lines.append(f"[{q}] {e['question']}" + ("  OK" if not issues else ""))
        for i in issues:
            counts[i.level] += 1
            lines.append(f"   {i.level:5s} {i.code:26s} {i.point:8s} {i.message}")
    lines.append(f"TOTAL: {counts}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rubric lint:", counts, "->", out_path)
    return counts
