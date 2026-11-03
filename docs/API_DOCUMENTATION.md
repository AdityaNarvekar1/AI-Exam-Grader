# AI Exam Answer Grader — API documentation (Group 2)

The grader is a **recommendation tool**. It suggests a mark, says which rubric points it found, and
flags answers a person should look at. A human grader keeps the final say on every mark.

Start the service:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn src.api.main:app --port 8000
```

Interactive docs: <http://localhost:8000/docs> (OpenAPI schema at `/openapi.json`).

| Method | Path | What it does |
|---|---|---|
| POST | `/evaluate-answer` | Grade one answer against a reference answer and rubric |
| POST | `/evaluate-answer/batch` | Up to 200 answers in one call |
| POST | `/grade-digitized-answers` | Grade a whole exam as delivered by Group 1 (OCR records + exam key) |
| GET | `/health` | Liveness check |
| GET | `/demo` | Small browser demo page |

---

## 1. `POST /evaluate-answer`

### Request

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | Question text |
| `reference_answer` | string | yes | Examiner's model answer |
| `rubric` | list of `{id, text, marks}` | yes, ≥ 1 point | Marks must sum to ≤ `maximum_score` |
| `student_answer` | string | yes | May be empty (scores 0, flagged) |
| `maximum_score` | number > 0 | yes | |
| `approach` | `hybrid` \| `embedding` \| `baseline` | no, default `hybrid` | See section 4 |

Optional per-point `binding` / `cues` metadata switches on the rule-based fact-check for that point
(see `src/models/fact_binding_checker.py`).

### Response

The first five fields are exactly the course handout's contract; the rest are extra.

| Field | Meaning |
|---|---|
| `question`, `score`, `maximum_score`, `feedback`, `confidence` | Handout contract |
| `approach_used` | Which grader produced the score |
| `needs_human_review` | `true` if confidence < 0.7 or a rubric point was contradicted |
| `component_scores` | Hybrid only: the two scores that were averaged |
| `point_breakdown` | Per rubric point: credited?, similarity, best-matching sentence, fact-check result |

### Status codes

`200` graded · `422` invalid input (missing field, rubric marks above maximum, empty rubric) ·
`400` batch too large · `500` unexpected error (returned as JSON, never a stack trace).

---

## 2. `POST /grade-digitized-answers` (Group 1 → Group 2, Week 7)

Accepts Group 1's records **in either format**:

* the schema published in the handout: `{student_id, question, answer_text, page, confidence}`
* Group 1's **real export** (`data/group1_real/ocr_output_5_students.csv`):
  `{student_id, question, question_type, ocr_answer, status, confidence, needs_review, page, placed_by_content}`

plus an **exam key** (question number → how to grade it). Short-answer entries need a reference answer
and rubric; MCQ entries need only the correct letter:

```json
{
  "exam_id": "ML-ENDSEM",
  "exam_key": {
    "Q1":  {"question": "Which of the following is an assumption of Linear Regression?",
            "question_type": "mcq", "correct_option": "B", "maximum_score": 1},
    "Q21": {"question": "Define Supervised Learning.", "question_type": "short",
            "reference_answer": "A type of ML where the model is trained on labeled data with known inputs and outputs.",
            "rubric": [{"id": "p1", "marks": 1, "text": "Supervised learning is a type of machine learning in which the model is trained on labeled data."},
                       {"id": "p2", "marks": 1, "text": "The labeled training data contains inputs together with their known correct outputs."}],
            "maximum_score": 2}
  },
  "answers": [
    {"student_id": "Student_11", "question": "Q1", "question_type": "mcq", "ocr_answer": "D",
     "status": "found", "confidence": 0.729, "needs_review": "no", "page": 1, "placed_by_content": "no"},
    {"student_id": "Student_11", "question": "Q21", "question_type": "short",
     "ocr_answer": "Supervised Learning is a type of machine - - that are used to train and predict data that are labeled.",
     "status": "found", "confidence": 0.884, "needs_review": "no", "page": 2, "placed_by_content": "no"}
  ],
  "approach": "hybrid",
  "ocr_review_threshold": 0.8
}
```

Ready-made exam keys for the real exam: `data/group1_real/exam_key_authored.json` and `exam_key_auto.json`.

What the adapter does (`src/integration/group1_adapter.py`):

1. Merges multi-page answers (same student + question) in page order; the merged OCR confidence is the
   lowest of the parts.
2. MCQ → letter match against `correct_option`. Short answers → the selected NLP grader.
3. Adds review reasons. Any reason sets `needs_human_review = true`:

| Reason | Trigger |
|---|---|
| `ocr_answer_not_found` | Group 1 found no text for this question (`status = not_found`) |
| `low_ocr_confidence` | OCR confidence below `ocr_review_threshold` (default 0.8) |
| `group1_needs_review` | Group 1 flagged the record itself |
| `group1_placed_by_content` | Group 1 assigned the text to the question by its content, not the printed number |
| `empty_ocr_text_possible_recognition_failure` | Text is empty although the record says found |
| `fact_check_contradiction` | A rubric point was contradicted |
| `low_grader_agreement` | Grader confidence below 0.7 |
| `no_exam_key_for_question` | The exam key has no entry; the answer is skipped, not crashed |

**Choosing the OCR threshold.** On the 65 real short answers, `confidence < 0.8 or not found` caught
11 of the 15 answers the OCR had damaged, with 6 false alarms (precision 0.65, recall 0.73). Raising it to
0.9 catches 13 of 15 but flags 44 of 65 answers. Group 1's own `needs_review` flag caught only 4 of 15.
See `results/week7/real_exam_report.json → ocr_review_routing`.

---

## 3. How to read `confidence`

It is a heuristic agreement score, not a probability. For `hybrid` it is
`1 − |baseline − embedding| / maximum_score`; on unseen questions it tracked lower error (Spearman +0.59).
For `embedding` alone it did not (−0.15, not significant). It says nothing about OCR quality: under
injected OCR noise the grader's confidence and review flag barely move
(`results/week7/ocr_noise_robustness.json`), so OCR confidence must be checked separately, which the
digitised-answer endpoint does.

## 4. Writing rubric points

Run `python src/tools/rubric_lint.py your_question.json` before grading. In short:

* Write each point as a full sentence that states a fact ("A list is mutable…"), not a topic label.
* Give each point its own marks; don't let one point carry most of the marks.
* Split compound points so partial answers can earn partial credit.
* "An example is given" style points cannot be checked by similarity models; state the expected example.
* The one-line keys in a typical faculty answer key are too terse to grade from directly. On the real
  exam the mechanically split key (`exam_key_auto.json`) did worse than a two-point authored rubric for
  every rubric-aware grader.

## 5. Known limitations

* On the real exam the zero-shot graders were far stricter than the teacher (who gave 2/2 to 41 of 65
  answers), and no grader beat a predict-the-teacher's-average baseline on mean error once calibrated.
  Treat scores as a ranking aid plus feedback, not a mark.
* Trained on synthetic answers (155) written by the team; real data so far is 5 students, one teacher.
* Rule-based fact-check only fires where binding/cue metadata was authored.
* No authentication or rate limiting.
