"""
build_exam_key.py  (Week 7)
----------------------------
Turns the faculty answer key that came with Group 1's scans (data/group1_real/answer_key.csv: 20 MCQ
letters + 15 one-line short-answer keys) into the EXAM KEY our API needs (question number ->
question, reference answer, rubric, maximum score).  The adapter's docstring already said this was
"the one new artefact the two components need from faculty"; the CSV gives us most of it, but it has
no marks and no rubric.

Two exam keys are written, so the effect of rubric authoring can be measured instead of assumed:

  exam_key_auto.json      rubric = the reference answer split mechanically into claims
                          (load_external.pseudo_rubric, equal marks).  No human input at all.
  exam_key_authored.json  rubric = 2 one-mark points per question, written by Group 2 FROM THE ANSWER KEY
                          TEXT ONLY, in the full-sentence style rubric_lint asks for.
                          CAVEAT: the author had already seen the student answers and teacher marks
                          when writing them; points were not edited after any model was run, but this is
                          not a blind rubric.  Faculty should replace it with their own.

Marks: MCQ 1 each; short answers 2 each (read off the teacher's marking on the scans: no short answer
gets more than 2; 20 + 15 x 2 = 50).
MCQ Q1: the CSV says B, but the teacher's ticks on all five scripts accept D (and reject A and C).
The key is kept as the CSV says (it is the faculty artefact) and the conflict is reported by
run_week7_real_exam.py rather than silently "fixed".
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from load_external import pseudo_rubric  # noqa: E402

D = ROOT / "data" / "group1_real"
SHORT_MAX = 2.0

AUTHORED = {
    "Q21": ["Supervised learning is a type of machine learning in which the model is trained on labeled data.",
            "The labeled training data contains inputs together with their known correct outputs."],
    "Q22": ["Model M1 is overfitting, because it scores 99% on training data but only 30% on test data.",
            "The model has memorized the training data and fails to generalize to unseen data."],
    "Q23": ["Regression analysis is a statistical process for estimating the relationship between variables.",
            "It relates a dependent variable to one or more independent variables."],
    "Q24": ["Poor data quality, missing values or difficulty in collecting data is a major challenge in machine learning.",
            "Overfitting, underfitting or the long time needed to train models is a major challenge in machine learning."],
    "Q25": ["Healthcare, such as disease prediction, is a major real-world application of data science.",
            "Finance fraud detection or e-commerce recommendation systems are major applications of data science."],
    "Q26": ["Structured data is organized in rows and columns, as in SQL tables or Excel sheets.",
            "Semi-structured data has some tags or markers but is not in a grid, as in JSON, XML or HTML."],
    "Q27": ["Outliers can be detected with the Z-score, the IQR rule, scatter plots or box plots.",
            "Outliers can be handled by trimming them, capping them or imputing new values."],
    "Q28": ["Data pre-processing is the set of steps that clean the raw data.",
            "Pre-processing transforms raw data into a format suitable for a machine learning model."],
    "Q29": ["K-NN is a distance-based algorithm that compares data points by their distance.",
            "Without normalization, features with larger scales dominate the distance calculation."],
    "Q30": ["Continuous attributes are discretized by grouping their values into bins or intervals.",
            "For example, ages from 1 to 100 can be grouped into child, adult and senior bins."],
    "Q31": ["K is the number of clusters that the data is divided into by K-Means.",
            "The number of clusters K is predefined, chosen before the algorithm runs."],
    "Q32": ["K in K-NN, the maximum tree depth or the number of trees are hyperparameters of machine learning algorithms.",
            "The learning rate and the batch size are hyperparameters of machine learning algorithms."],
    "Q33": ["Gradient descent is an optimization algorithm that minimizes a cost function.",
            "Gradient descent iteratively updates the parameters by moving in the direction of steepest descent."],
    "Q34": ["The best SVM classifier is the one with the widest margin between the boundaries.",
            "A good classifier has no data points falling inside the margin and no misclassified points."],
    "Q35": ["The RBF kernel maps the data into a higher-dimensional space.",
            "This lets the SVM find a non-linear decision boundary."],
}


def main():
    rows = list(csv.DictReader(open(D / "answer_key.csv", newline="", encoding="utf-8-sig")))
    auto, authored = {}, {}
    for r in rows:
        q = r["question"]
        if r["question_type"] == "mcq":
            e = {"question": r["question_text"], "question_type": "mcq", "correct_option": r["correct_answer"].strip(),
                 "maximum_score": 1.0}
            auto[q] = authored[q] = e
            continue
        ref = r["correct_answer"].strip()
        auto[q] = {"question": r["question_text"], "question_type": "short", "reference_answer": ref,
                   "rubric": pseudo_rubric(ref, SHORT_MAX), "maximum_score": SHORT_MAX}
        authored[q] = {"question": r["question_text"], "question_type": "short", "reference_answer": ref,
                       "rubric": [{"id": f"p{i + 1}", "text": t, "marks": 1.0} for i, t in enumerate(AUTHORED[q])],
                       "maximum_score": SHORT_MAX}
    for name, key in (("exam_key_auto.json", auto), ("exam_key_authored.json", authored)):
        json.dump(key, open(D / name, "w"), indent=2)
        print("wrote", D / name, len(key), "questions")
    # rubric lint on the authored key (same checks rubric authors are told to run)
    from rubric_lint_shim import lint_key
    lint_key(authored, D / "exam_key_authored_lint.txt")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
