# Week 7 - real scripts from Group 1 (5 students, teacher marks from the scans)

Graders run: const, length, tfidf, lexical

## OCR quality
```
{
 "n_attempted": 65,
 "n_missed_not_found": 3,
 "n_found": 62,
 "median_CER_found": 0.056,
 "mean_CER_found": 0.11,
 "median_WER_found": 0.133,
 "mean_WER_found": 0.195,
 "n_truncated_len_ratio_lt_0.75": 2,
 "n_misplaced": 0,
 "n_bad_any": 15,
 "missed_items": [
  "Student_11/Q34 (teacher 2)",
  "Student_34/Q32 (teacher 2)",
  "Student_35/Q32 (teacher 1)"
 ],
 "misplaced_items": [],
 "truncated_items": [
  "Student_11/Q24 (0.51)",
  "Student_35/Q24 (0.56)"
 ],
 "MCQ_letters_note": "MCQ letters checked against the scans: 99 of 100 match; Student_11 Q1 wrote C, OCR read D."
}
```

## Short answers (teacher mark 0-2, n = attempted answers)

| grader | rubric | text | MAE zero-shot | QWK | exact | Spearman | MAE calibrated (LOQO) | QWK cal | exact cal |
|---|---|---|---|---|---|---|---|---|---|
| const | authored | clean | 0.769 | 0.0 | 0.231 | None | 0.66 | -0.528 | 0.231 |
| const | authored | ocr | 0.815 | -0.019 | 0.215 | -0.036 | 0.66 | -0.528 | 0.231 |
| const | auto | clean | 0.769 | 0.0 | 0.231 | None | 0.66 | -0.528 | 0.231 |
| const | auto | ocr | 0.815 | -0.019 | 0.215 | -0.036 | 0.66 | -0.528 | 0.231 |
| length | authored | clean | 0.869 | 0.073 | 0.2 | 0.198 | 0.657 | 0.033 | 0.585 |
| length | authored | ocr | 0.9 | 0.093 | 0.2 | 0.206 | 0.635 | 0.191 | 0.6 |
| length | auto | clean | 0.869 | 0.073 | 0.2 | 0.198 | 0.657 | 0.033 | 0.585 |
| length | auto | ocr | 0.9 | 0.093 | 0.2 | 0.206 | 0.635 | 0.191 | 0.6 |
| tfidf | authored | clean | 1.054 | 0.026 | 0.185 | 0.143 | 0.672 | -0.269 | 0.338 |
| tfidf | authored | ocr | 1.123 | 0.016 | 0.185 | 0.123 | 0.678 | -0.189 | 0.354 |
| tfidf | auto | clean | 1.054 | 0.026 | 0.185 | 0.143 | 0.672 | -0.269 | 0.338 |
| tfidf | auto | ocr | 1.123 | 0.016 | 0.185 | 0.123 | 0.678 | -0.189 | 0.354 |
| lexical | authored | clean | 0.785 | 0.16 | 0.431 | 0.203 | 0.64 | 0.107 | 0.446 |
| lexical | authored | ocr | 0.862 | 0.169 | 0.385 | 0.245 | 0.638 | 0.229 | 0.538 |
| lexical | auto | clean | 1.246 | 0.03 | 0.231 | 0.077 | 0.668 | -0.249 | 0.354 |
| lexical | auto | ocr | 1.308 | 0.002 | 0.215 | 0.003 | 0.676 | -0.527 | 0.231 |
| teacher_mean_loqo | - | clean | - | - | - | -0.599 | 0.66 | -0.528 | 0.231 |
| length_loqo | - | clean | - | - | - | 0.153 | 0.61 | 0.128 | 0.508 |
| teacher_mean_loqo | - | ocr | - | - | - | -0.599 | 0.66 | -0.528 | 0.231 |
| length_loqo | - | ocr | - | - | - | -0.189 | 0.701 | -0.17 | 0.4 |

## MCQ
```
{
 "per_student": {
  "Student_11": {
   "teacher": 10,
   "auto_csv_key": 10.0,
   "auto_if_Q1_is_D": 11.0,
   "ocr_Q1_letter": "D"
  },
  "Student_23": {
   "teacher": 10,
   "auto_csv_key": 10.0,
   "auto_if_Q1_is_D": 10.0,
   "ocr_Q1_letter": "A"
  },
  "Student_34": {
   "teacher": 13,
   "auto_csv_key": 12.0,
   "auto_if_Q1_is_D": 13.0,
   "ocr_Q1_letter": "D"
  },
  "Student_35": {
   "teacher": 11,
   "auto_csv_key": 10.0,
   "auto_if_Q1_is_D": 10.0,
   "ocr_Q1_letter": "A"
  },
  "Student_50": {
   "teacher": 13,
   "auto_csv_key": 12.0,
   "auto_if_Q1_is_D": 13.0,
   "ocr_Q1_letter": "D"
  }
 },
 "exact_match_students_csv_key": 2,
 "exact_match_students_Q1_D": 3,
 "note": "Teacher ticks D for Q1 on every script; answer_key.csv says B. Student_11 wrote C (OCR read D), so with key D the OCR error would award her a mark she did not earn."
}
```
