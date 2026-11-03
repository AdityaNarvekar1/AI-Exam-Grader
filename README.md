# Group 2 — AI Exam Answer Grader

**Course:** NLP Course Project 2026 — "AI University Platform"
**Team (Roll No.):** S010, S011, S012, S013, S014, S015
**Status:** Final (Week 8) — API, Group 1 integration on real scripts, v2 evaluation

## What it is

A reusable NLP component that compares a student's answer to a reference
answer and a mark-weighted rubric, and returns a **recommended** score,
rubric-grounded feedback, a confidence value and a "needs human review" flag.
**It is a triage aid — a human grader keeps final authority.**

**Component:** AI Exam Answer Grader · **Platform area:** Academic Services → Assessment
**Interface:** `POST /evaluate-answer` (full contract: [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md))
**Upstream:** consumes Group 1's (Handwritten Examination Digitizer) real OCR export — see `data/group1_real/` and the Week 7 report.

## Reusable Asset Test (Section 21): "what would another group need?"

1. `pip install -r requirements.txt` and `python -m spacy download en_core_web_md`
2. `uvicorn src.api.main:app --port 8000`
3. Open <http://localhost:8000/docs>, or run `python examples/call_api.py`

Everything else — request/response fields, status codes, how to read
`confidence`, how to write rubric points, known limitations — is in
`docs/API_DOCUMENTATION.md`.

## Repository structure

```
├── README.md
├── requirements.txt
├── data/exam_grading_dataset.csv        # 70 synthetic, rubric-anchored answers (34 train / 11 val / 25 test)
├── src/
│   ├── preprocessing/generate_dataset.py
│   ├── models/
│   │   ├── baseline_grader.py           # TF-IDF + keyword overlap (Week 2)
│   │   ├── embedding_grader.py          # per-concept embedding matching (Week 3)
│   │   └── fact_binding_checker.py      # factual-inversion veto layer (Week 5)
│   ├── api/                             # FastAPI service (Week 6)
│   │   └── main.py  schemas.py  model_loader.py
│   └── evaluation/                      # every reported number is reproducible from here
│       ├── evaluate_baseline.py  evaluate_embedding_model.py  compare_models.py
│       ├── tune_threshold.py  evaluate_unseen_questions.py  analyze_hybrid_and_confidence.py
│       ├── build_human_validation_sheet.py  compute_human_agreement.py
│       └── verify_sentence_transformer_backend.py  diagnose_sentence_transformer_gap.py
├── tests/test_api.py                    # 18 tests: contract, validation, fact-check, edge cases
├── examples/call_api.py                 # runnable client
├── results/                             # JSON/CSV outputs of the scripts above
└── docs/                                # API docs + Week 1-6 reports
```

## Results at a glance (and how much to trust them)

Two measurements, deliberately kept separate:

| Setting | What it measures | Baseline | Embedding | Hybrid |
|---|---|---|---|---|
| Fixed split, n=25 (Weeks 4–5) | Grading *new students* on questions the model was calibrated on | MAE 1.42 | MAE 1.04 | — |
| **Leave-one-question-out, n=70 (Week 6)** | Grading a **question the model has never seen** — the real API use case | MAE 1.27 | MAE 1.53 | **MAE 1.09** |

On unseen questions the Weeks 4–5 advantage of the embedding model
**does not hold**: baseline and embedding are not statistically distinguishable
(paired bootstrap), and embedding is *worse* on ordinary answers. The hybrid
is reliably better than embedding alone and not distinguishable from baseline
overall (hence it is the default for its feedback, review flag and adversarial
robustness, not because it was proven superior). The Weeks 4–5 comparisons
were not significance-tested and used n=20–25. All labels are synthetic;
human validation (`results/v2/human_validation/BLIND_SHEET.csv`) is pending.

## Reproduce

```bash
python src/preprocessing/generate_dataset.py
python src/evaluation/evaluate_baseline.py
python src/evaluation/evaluate_embedding_model.py spacy
python src/evaluation/compare_models.py hard spacy
python src/evaluation/tune_threshold.py spacy
python src/evaluation/evaluate_unseen_questions.py          # ~1 min
python src/evaluation/analyze_hybrid_and_confidence.py
pytest tests/test_api.py -v
```

## Weeks 7–8 and v2 (final)

| What | Where |
|---|---|
| Real Group 1 output, answer key, scans, teacher marks + transcriptions | `data/group1_real/` |
| Adapter for Group 1's real CSV (ocr_answer, status, MCQ) | `src/integration/group1_adapter.py` |
| Real-exam evaluation (OCR quality, routing, grading vs teacher) | `src/week7/run_week7_real_exam.py` → `results/week7/` |
| Synthetic OCR-noise sweep | `src/week7/ocr_noise_robustness.py` |
| Transformer graders (NLI, bi-encoder), LLM judge, stacking | `src/models/`, `src/evaluation/model_adapters.py` |
| Length-decorrelated challenge set | `data/v2/challenge_set.csv` |
| Benchmarks A/B/C with clustered CIs | `src/evaluation/run_benchmark.py` → `results/v2/` |
| Public human-graded validation (Mohler) | `src/evaluation/run_external_benchmark.py` |
| Reports | `docs/Week7_Integration_Responsible_AI_Group2.docx`, `docs/Week8_Final_Report_Group2.docx`, `docs/Group2_Final_Presentation.pptx` |

Headline: NLI features give the lowest error on unseen synthetic questions (MAE 0.90 vs 1.09 for the
Week 6 hybrid). On 65 real answers marked by the teacher, the graders are much stricter than the teacher
and do not beat "predict the teacher's average" once calibrated, so the component is a feedback and
triage aid. OCR confidence < 0.8 catches 11 of 15 OCR-damaged answers.

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_md
python src/evaluation/run_benchmark.py --graders const length tfidf lexical nli   # synthetic A/B/C
python src/week7/build_scan_ground_truth.py && python src/week7/build_exam_key.py
python src/week7/run_week7_real_exam.py                                           # real scripts
python src/week7/ocr_noise_robustness.py
python -m pytest -q
```

## Roadmap

- **Done:** Week 7 integration on real scripts, Responsible AI, Week 8 report and deck.
- **Open:** more real teacher-marked scripts; faculty rubrics; run the LLM judge on real answers; fill the blind human-validation sheet (`results/v2/human_validation/BLIND_SHEET.csv`).
- **Future (needs an LLM API key, unavailable so far):** an LLM/NLI-based
  fact-check — the general fix for factual inversions that the rule-based
  layer only partly covers.
