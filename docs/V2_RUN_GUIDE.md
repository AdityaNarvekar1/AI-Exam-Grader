# Run guide (v2, Weeks 7–8)

All commands from the project root. CPU is enough; the NLI and bi-encoder models download from
Hugging Face on first use (~1 GB).

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m pytest -q                                   # all tests
```

## Synthetic benchmark (protocols A, B, C)
```bash
python src/evaluation/run_benchmark.py --graders const length tfidf lexical spacy_embedding spacy_hybrid --tag classic
python src/evaluation/run_benchmark.py --graders tfidf lexical bi nli nli_bi nli_bi_exp --tag transformers
```
Model choice via env vars: `G2_NLI_MODEL`, `G2_BI_MODEL`, `G2_DEVICE`, `G2_NLI_VETO`.

## Week 7: Group 1's real scripts
```bash
python src/week7/build_scan_ground_truth.py   # teacher marks + transcriptions from the scans
python src/week7/build_exam_key.py            # exam keys (auto / authored) + rubric lint
python src/week7/run_week7_real_exam.py       # OCR quality, routing, grading vs teacher
python src/week7/ocr_noise_robustness.py      # synthetic OCR-noise sweep
python src/week7/write_reports.py             # rebuilds the Week 7 and Week 8 .docx reports
```

## Public human-graded data
```bash
python src/data/export_hf_datasets.py --only mohler
python src/evaluation/run_external_benchmark.py --task mohler --graders const length refsim lex
```

## LLM judge (not yet run on real answers)
```bash
ollama pull qwen2.5:7b-instruct
G2_LLM_BACKEND=ollama G2_LLM_MODEL=qwen2.5:7b-instruct python src/week7/run_week7_real_exam.py --graders judge lexical
```

## API
```bash
uvicorn src.api.main:app --port 8000      # docs at /docs; contract in docs/API_DOCUMENTATION.md
```
