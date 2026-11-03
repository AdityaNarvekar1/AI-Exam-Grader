"""Builds docs/Week7_*.docx and docs/Week8_Final_Report_Group2.docx from the result files."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_style import Report

wk = json.load(open(ROOT / "results/week7/real_exam_report.json"))
oq, rt, mcq = wk["ocr_quality"], wk["ocr_review_routing"], wk["mcq"]
res = wk["short_answer_results"]
noise = json.load(open(ROOT / "results/week7/ocr_noise_robustness.json"))
TEAM = ("Team (Roll No.)", "S010, S011, S012, S013, S014, S015")

def g(tag, kind="zero_shot"):
    r = res.get(tag, {}).get(kind)
    return r

# ------------------------------------------------------------------ Week 7
R = Report("AI Exam Answer Grader", "Week 7 Report — Integration with Group 1 and Responsible AI — Group 2")
R.kv([TEAM, ("Week", "Week 7 — Integration & Responsible AI (Target: 17–30 October 2026)"),
      ("Deliverable", "Group 1 → Group 2 integration on real scripts; OCR-error study; Responsible-AI statement")])
R.h("1. Objective")
R.p("Week 6 left three open items: run the grader on Group 1's digitised answers and measure what recognition errors do to the scores, replace synthetic labels with real grader judgement, and write the Responsible-AI statement. Group 1 has now delivered real output for five students, and the scanned scripts carry the course teacher's red-pen marks. That gave us, for the first time, real human grades to test against.")
R.h("2. Data received from Group 1")
R.bullets([f"**OCR export** (data/group1_real/ocr_output_5_students.csv): 5 students × 35 questions (20 MCQ, 15 short answers), one row per question, with OCR confidence, Group 1's own review flag and a 'placed_by_content' flag.",
           "**Faculty answer key** (answer_key.csv): MCQ letters and one-line keys for the short answers. No marks, no rubric.",
           "**Scans** (scans/Student_*.pdf): the handwritten scripts as marked by the teacher."])
R.p("From the scans we recorded each short answer's teacher mark and a manual transcription (src/week7/build_scan_ground_truth.py → scan_ground_truth.csv). Every short answer is marked out of 2. For all five students, MCQ marks plus short-answer marks reproduce the total the teacher circled on page 1 exactly (40, 23, 33, 25, 35). Five marks used a glyph that is not a clean numeral; they were set to the only value that makes the total add up and are flagged in the file. Transcriptions were made by reading the scans and should be spot-checked by a team member.")
d = wk["counts"]["teacher_mark_distribution"]
R.p(f"Result: **{wk['counts']['attempted_per_scan']} attempted short answers with a real teacher mark** (10 slots were left blank). The teacher is lenient: {d['2']} answers got 2/2, {d['1']} got 1 and {d['0']} got 0.")
R.h("3. Interface changes needed for the real data")
R.p("Group 1's real export does not match the schema in the handout, so src/integration/group1_adapter.py and the API schema were extended (old format still accepted; all 29 API/integration tests pass):")
R.table(["Real data", "What we changed"], [
    ["Text field is ocr_answer, not answer_text", "Both accepted"],
    ["status = not_found, empty confidence", "Empty text, confidence 0, review reason ocr_answer_not_found"],
    ["question_type = mcq", "MCQs graded by letter match against correct_option; never reach the NLP grader"],
    ["needs_review, placed_by_content flags", "Passed through as review reasons"],
    ["CSV starts with a UTF-8 byte-order mark", "Reader uses utf-8-sig (the first column name was otherwise unreadable)"],
    ["Answer key has no marks or rubric", "build_exam_key.py makes two exam keys: auto (reference split into claims) and authored (2 one-mark points per question)"]],
    widths=[2.6, 4.2])
R.h("4. MCQ results")
rows = [[s, v["teacher"], int(v["auto_csv_key"]), int(v["auto_if_Q1_is_D"]), v["ocr_Q1_letter"]] for s, v in mcq["per_student"].items()]
R.table(["Student", "Teacher MCQ total", "Auto (answer_key.csv)", "Auto if Q1 = D", "OCR letter for Q1"], rows, widths=[1.3, 1.3, 1.5, 1.3, 1.4])
R.p("OCR letters match the scans in 99 of 100 MCQs. The mismatches with the teacher come from the key, not the OCR: answer_key.csv says Q1 = B, but the teacher's ticks accept D on every script. With D, three of five totals match exactly; the one remaining OCR error (Student_11 wrote C, OCR read D) would then wrongly gain a mark. Student_35's remaining one-mark gap is an ambiguous tick on Q12. We report the key conflict to faculty rather than change it ourselves.")
R.h("5. OCR quality on the short answers")
R.table(["Measure", "Value"], [
    ["Attempted answers / found by OCR", f"{oq['n_attempted']} / {oq['n_found']}"],
    ["Median character error rate (found)", f"{oq['median_CER_found']:.1%} (mean {oq['mean_CER_found']:.1%})"],
    ["Median word error rate (found)", f"{oq['median_WER_found']:.1%} (mean {oq['mean_WER_found']:.1%})"],
    ["Missed entirely", "; ".join(oq["missed_items"])],
    ["Truncated (second page lost)", "; ".join(oq["truncated_items"])],
    ["Answers damaged in any way", f"{oq['n_bad_any']} of {oq['n_attempted']}"]], widths=[2.6, 4.2])
R.p("The costly errors are structural, not character-level: a whole answer that runs over two pages is either missed (Student_11 Q34) or loses its continuation (Student_11 Q24, Student_35 Q24). Student_34's Q32 text was appended to Q31 and Q32 came back as not_found. Character errors are mostly visual confusions ('Kemel' for Kernel, 'SUM' for SVM, 'outful' for output).")
R.h("6. Which flag catches damaged OCR?")
R.table(["Routing rule", "Flagged", "Damaged caught", "Precision", "Recall"], [
    ["OCR confidence < 0.8 or not found", rt["ocr_conf_lt_0.8_or_not_found"]["flagged"], f"{rt['ocr_conf_lt_0.8_or_not_found']['caught_bad']} / 15", rt["ocr_conf_lt_0.8_or_not_found"]["precision"], rt["ocr_conf_lt_0.8_or_not_found"]["recall"]],
    ["OCR confidence < 0.9 or not found", rt["ocr_conf_lt_0.9_or_not_found"]["flagged"], f"{rt['ocr_conf_lt_0.9_or_not_found']['caught_bad']} / 15", rt["ocr_conf_lt_0.9_or_not_found"]["precision"], rt["ocr_conf_lt_0.9_or_not_found"]["recall"]],
    ["Group 1 needs_review or not found", rt["group1_needs_review_or_not_found"]["flagged"], f"{rt['group1_needs_review_or_not_found']['caught_bad']} / 15", rt["group1_needs_review_or_not_found"]["precision"], rt["group1_needs_review_or_not_found"]["recall"]],
    ["Any Group 1 signal", rt["any_group1_signal"]["flagged"], f"{rt['any_group1_signal']['caught_bad']} / 15", rt["any_group1_signal"]["precision"], rt["any_group1_signal"]["recall"]]],
    widths=[2.6, 0.9, 1.2, 0.9, 0.9], note="'Damaged' = not found, misplaced, character error rate above 15%, or fewer than 75% of the words.")
R.p("We keep the API default of 0.8: it catches 11 of 15 damaged answers with 6 false alarms. Group 1's own flag adds little. Recommendation to Group 1: flag answers that continue onto another page, since those caused every serious failure.")
R.h("7. Grading the real answers")
R.p("Graders were trained only on our synthetic data (155 rows) and applied unchanged to the real answers ('zero-shot'). We also tested a 'calibrated' version: a monotone map from predicted score to teacher mark, fitted leave-one-question-out on the real answers, so no question's own marks are used to grade it.")
rows = []
order = [("const", "authored", "Predict synthetic average"), ("length", "authored", "Word count only"), ("tfidf", "authored", "TF-IDF baseline (Week 2)"),
         ("spacy_hybrid", "authored", "spaCy hybrid (Week 6 default)"), ("lexical", "auto", "Lexical rubric, auto key"), ("lexical", "authored", "Lexical rubric, authored key"),
         ("feat_lex", "authored", "Feature grader, lexical"), ("bi", "authored", "Feature + bi-encoder"), ("nli", "authored", "Feature + NLI (DeBERTa)"), ("nli_bi", "authored", "Feature + NLI + bi-encoder")]
for gn, k, label in order:
    zc, zo = g(f"{gn}|{k}|clean"), g(f"{gn}|{k}|ocr")
    cc = g(f"{gn}|{k}|clean", "calibrated_loqo")
    if zc:
        rows.append([label, zc["MAE"], zc["QWK_0_1_2"], f"{zc['exact_agree_rounded']:.0%}", zo["MAE"], cc["MAE"]])
tm = res["teacher_mean_loqo|-|clean"]["calibrated_loqo"]
rows.append(["Predict teacher's average (LOQO)", "-", "-", "-", "-", tm["MAE"]])
R.table(["Grader", "MAE clean", "QWK", "Exact", "MAE on OCR text", "MAE calibrated"], rows, widths=[2.3, 0.8, 0.7, 0.7, 1.1, 1.1],
        note="Teacher mark 0–2, n = 65. MAE in marks. QWK on 0/1/2. Exact = rounded prediction equals teacher mark.")
R.p("Three findings, stated plainly. First, zero-shot, every grader is far stricter than this teacher, who gives full marks to partly correct answers; our synthetic labels reward rubric coverage. Second, the authored two-point rubric clearly beats the mechanically split answer key for the rubric-aware graders, so faculty rubrics matter more than model choice here. Third, once calibrated to the teacher, no grader beats simply predicting the teacher's average mark on mean error, and agreement (QWK) stays low. With 65 answers from one lenient teacher, our system is not ready to propose marks on this exam; it is useful for feedback, ranking and routing.")
R.h("8. Effect of OCR errors on scores")
eff = wk["ocr_effect_on_scores"]
rows = []
for gn in ["tfidf", "lexical", "spacy_hybrid", "nli", "nli_bi"]:
    e = eff.get(f"{gn}|authored")
    if e:
        d_ = e["MAE_increase_from_OCR"]
        rows.append([gn, e["MAE_clean"], e["MAE_ocr"], f"{d_['estimate']:+.3f} [{d_['ci_low']}, {d_['ci_high']}]", e["mean_abs_score_shift"], e["items_shift_ge_1_mark"]])
R.table(["Grader", "MAE clean", "MAE OCR", "Increase [95% CI]", "Mean score shift", "Items moved ≥ 1 mark"], rows, widths=[1.3, 0.9, 0.9, 1.7, 1.1, 1.1])
R.p("On real scripts OCR adds a small amount of error on average, driven by the few missed or truncated answers. To see the trend beyond five students we injected synthetic OCR noise into the 85-row challenge set (src/week7/ocr_noise_robustness.py):")
rows = []
for gn, per in noise["graders"].items():
    rows.append([gn] + [per[c]["QWK"] for c in ["cer~0.00", "cer~0.05", "cer~0.10", "cer~0.20", "truncated_40pct"]])
R.table(["Grader", "QWK clean", "5% CER", "10% CER", "20% CER", "40% truncated"], rows, widths=[1.5, 1, 1, 1, 1, 1.2])
R.p("String-matching graders lose most of their signal by 10–20% character error. The real OCR sits around 6% median, so it is inside the tolerable range for most answers; the outliers are what must go to a human.")
R.h("9. Responsible-AI statement")
R.bullets(["**Advisory only.** The service returns a recommended mark; it never writes a grade. The teacher keeps final authority.",
           "**Two separate review triggers.** OCR confidence and grader confidence measure different things; either one low sends the answer to a person.",
           "**Known bias.** The grader is stricter than the one teacher we measured, so it would under-mark lenient courses if used unreviewed. Calibrate per course before any use.",
           "**Narrow evidence.** Trained on synthetic answers written by the team; real validation is 5 students, one exam, one teacher.",
           "**Language.** Hinglish and non-native English score lower with lexical signals; these answers should be reviewed.",
           "**Prompt injection.** Answers are treated as data; the LLM-judge design requires a verbatim quote for any credit.",
           "**Privacy.** Real scripts are stored under anonymous IDs only; names visible on the scans are not transcribed."])
R.h("10. Next steps (Week 8)")
R.bullets(["Final report and presentation.", "Ask faculty to confirm the Q1 key and to write rubrics for the short answers.", "Collect more scripts and a second teacher's marks; then re-fit the calibration.", "Run the LLM judge (code ready, src/models/llm_judge.py) on the 65 real answers."])
R.save(ROOT / "docs/Week7_Integration_Responsible_AI_Group2.docx")

# ------------------------------------------------------------------ Week 8
bench = json.load(open(ROOT / "results/v2/benchmark_classic.json"))
tr = ROOT / "results/v2/benchmark_transformers.json"
tb = json.load(open(tr)) if tr.exists() else {}
F = Report("AI Exam Answer Grader", "Week 8 — Final Project Report — Group 2")
F.kv([TEAM, ("Week", "Week 8 — Final report, presentation and demo (31 October 2026)"),
      ("Component", "Academic Services → Assessment: AI Exam Answer Grader (consumes Group 1's digitised scripts)")])
F.h("1. Summary")
F.p("We built a reusable grading component that compares a student's answer with a reference answer and a mark-weighted rubric, returns a recommended score with per-point feedback, and flags answers that need a person. In the last two weeks we connected it to Group 1's real output, measured OCR quality, and tested the grader against a teacher's real marks for the first time. The pipeline works end to end. Our models beat earlier versions on held-out synthetic questions, but they do not yet agree well with the real teacher, so the component should be used for feedback and triage, not to set marks.")
F.h("2. Problem and approach")
F.p("Faculty spend a lot of time marking short descriptive answers and give little explanation. Our component takes the question, reference answer, rubric, student answer and maximum marks, and returns a score, feedback, a confidence value and a review flag. Two families of approach were built and compared, as the course requires: (A) similarity-based graders, from TF-IDF (Week 2) through spaCy embeddings (Week 3) to transformer NLI and bi-encoder features, and (B) a rubric-grounded LLM judge that must quote evidence for each point (code complete, not yet run at scale).")
F.h("3. Timeline")
F.table(["Week", "Delivered"], [
    ["1", "Proposal; synthetic, rubric-anchored dataset plan"],
    ["2", "70-answer dataset; TF-IDF + keyword baseline"],
    ["3", "Per-rubric-point embedding grader (spaCy)"],
    ["4–5", "Comparison, error analysis, rule-based fact-check veto"],
    ["6", "REST API, hybrid default, leave-one-question-out evaluation"],
    ["v2 audit", "Length shortcut found; challenge set; fixed fact-check; NLI/bi-encoder/LLM-judge graders; clustered statistics"],
    ["7", "Group 1 integration on real scripts; teacher marks from scans; OCR study; Responsible AI"],
    ["8", "This report, presentation, demo"]], widths=[1.0, 5.8])
F.h("4. What the audit changed")
F.p("Our synthetic labels were strongly tied to answer length (Spearman 0.82): a model that only counts words scored MAE 1.25 against 1.27 for the TF-IDF baseline. We added an 85-answer challenge set where length says nothing about quality (correlation −0.09), fixed a substring bug in the fact-check ('mutable' inside 'immutable'), and switched to confidence intervals that resample whole questions.")
F.h("5. Results on synthetic data")
rows = []
for gn, lab in [("length", "Word count only"), ("tfidf", "TF-IDF (Week 2)"), ("spacy_embedding", "spaCy embedding (Week 3)"), ("spacy_hybrid", "spaCy hybrid (Week 6)"), ("lexical", "Lexical rubric (v2)")]:
    A, B = bench["A"][gn], bench["B"][gn]
    C = bench.get("C", {}).get(gn)
    rows.append([lab, A["MAE"], A["QWK"], B["MAE"], B["QWK"], C["MAE"] if C else "-"])
for gn, lab in [("bi", "Feature + bi-encoder"), ("nli", "Feature + NLI (DeBERTa)"), ("nli_bi", "Feature + NLI + bi-encoder")]:
    A = tb.get("A", {}).get(gn); B = tb.get("B", {}).get(gn); C = tb.get("C", {}).get(gn)
    if A:
        rows.append([lab, A["MAE"], A["QWK"], B["MAE"] if B else "-", B["QWK"] if B else "-", C["MAE"] if C else "-"])
F.table(["Grader", "A: MAE", "A: QWK", "B: MAE", "B: QWK", "C: MAE"], rows, widths=[2.3, 0.8, 0.8, 0.8, 0.8, 0.8],
        note="A = leave-one-question-out on original 70; B = train on 70, test on challenge 85; C = leave-one-question-out on pooled 155. MAE in marks (5/10-mark questions). Transformer rows: protocol A only; the B/C run was stopped for lack of CPU time and is one command to rerun.")
F.p("On unseen synthetic questions the NLI feature grader has the lowest error (0.90 vs 1.09 for the Week 6 hybrid). On the challenge set the original spaCy model collapses (QWK 0.13), confirming that its Week 4–6 advantage was partly the length shortcut.")
F.h("6. Real scripts (Week 7)")
F.p(f"Five students, 65 attempted short answers with teacher marks read off the scans. OCR: median character error {oq['median_CER_found']:.1%}, three answers missed and two truncated, all from answers running over two pages. 'OCR confidence < 0.8 or not found' catches 11 of 15 damaged answers. MCQ letters are 99/100 correct; the faculty key disagrees with the teacher on Q1. The teacher gave full marks to 41 of 65 answers. Zero-shot, our graders are much stricter; after calibrating to the teacher, none beats predicting the teacher's average mark (MAE {tm['MAE']}). Details: Week 7 report.")
F.h("7. Outside validation (public human-graded data)")
F.p("On the Mohler et al. computer-science dataset (2,273 answers, 81 questions, two human graders, 5-fold split by question), word count has no predictive value (QWK 0.02), TF-IDF reference similarity reaches QWK 0.37, and the two human graders agree with each other at QWK 0.44. The length shortcut we found is a property of our synthetic data, not of real marking.")
F.h("8. Limitations")
F.bullets(["Real validation is small: one exam, five students, one teacher; five marks were inferred from the circled totals.",
           "Synthetic training data written by the team; real-teacher agreement is low.",
           "Rubrics for the real exam were written by us after seeing the answers; faculty rubrics are needed.",
           "The LLM judge is implemented and unit-tested but has not been run on the real answers.",
           "No authentication or rate limiting on the API."])
F.h("9. Recommendations")
F.bullets(["Use the component for per-point feedback, ranking and review routing; keep marks with the teacher.",
           "Route answers with OCR confidence below 0.8, not_found status, or a multi-page answer to a person.",
           "Collect 200+ teacher-marked real answers from at least two teachers, then calibrate per course.",
           "Have faculty write two-to-four point rubrics in full sentences; run rubric_lint before grading."])
F.h("10. Reproducing")
F.p("pip install -r requirements.txt; python -m spacy download en_core_web_md; python src/evaluation/run_benchmark.py --graders tfidf lexical nli; python src/week7/run_week7_real_exam.py; python src/week7/ocr_noise_robustness.py; pytest. See README.md and docs/V2_RUN_GUIDE.md.")
F.save(ROOT / "docs/Week8_Final_Report_Group2.docx")
print("reports written")
