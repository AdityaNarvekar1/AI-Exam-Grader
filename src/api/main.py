"""
main.py
-------
Group 2 - AI Exam Answer Grader
Reusable NLP asset exposed as a REST API (handout Section 9).

Run from the project root:
    uvicorn src.api.main:app --port 8000

Then open
    http://localhost:8000/docs   interactive API documentation (auto-generated)
    http://localhost:8000/demo/  the demonstration user interface

Endpoints
---------
POST /evaluate-answer         grade one student answer (core endpoint)
POST /evaluate-answer/batch   grade a list of answers in one call
POST /grade-digitized-answers grade Group 1's digitised answer records (Week 7)
GET  /health                  liveness + which approaches are loaded
GET  /                        basic info and links
GET  /demo/                   demo UI (static page calling this API)

This is a triage aid: a human grader keeps final authority over every mark.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))          # allow `uvicorn src.api.main:app` from the project root
sys.path.append(str(HERE.parent))   # src/ (for the integration package)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from model_loader import registry
from schemas import (EvaluateAnswerRequest, EvaluateAnswerResponse, PointBreakdown,
                      HealthResponse, ErrorResponse, GradeDigitizedRequest,
                      GradeDigitizedResponse)
from integration.group1_adapter import grade_digitized

DEMO_DIR = HERE.parents[1] / "demo"
EXAMPLES_DIR = HERE.parents[1] / "examples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    yield


app = FastAPI(
    title="AI Exam Answer Grader API",
    description=(
        "Group 2's reusable NLP asset for the AI University Platform "
        "(Academic Services -> Assessment). Compares a student's answer against a "
        "reference answer and a mark-weighted rubric and returns a recommended score, "
        "confidence, rubric-grounded feedback and a human-review flag. Approaches: "
        "'baseline' (TF-IDF + keyword overlap), 'embedding' (pretrained-embedding "
        "per-concept matching with a fact-binding veto layer) and 'hybrid' (default). "
        "This is a triage aid: a human grader keeps final authority."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Lets another group's browser front-end call this API during the class
# demonstration. "*" is for the demo only - restrict origins in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _breakdown(point_matches):
    return [PointBreakdown(point_id=m.point_id, point_text=m.point_text, marks=m.marks,
                            credited=m.credited, similarity=m.similarity,
                            best_matched_sentence=m.best_sentence, fact_check=m.fact_check)
            for m in point_matches]


def _rubric(req: EvaluateAnswerRequest):
    return [p.model_dump(exclude_none=True) for p in req.rubric]


def _grade_with_baseline(req: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
    r = registry.baseline.grade_one(question=req.question, reference_answer=req.reference_answer,
                                    rubric=_rubric(req), student_answer=req.student_answer,
                                    max_marks=req.maximum_score)
    return EvaluateAnswerResponse(question=req.question, score=r.score,
                                  maximum_score=req.maximum_score, feedback=r.feedback,
                                  confidence=r.confidence, approach_used="baseline")


def _grade_with_embedding(req: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
    r = registry.embedding.grade_one(reference_answer=req.reference_answer, rubric=_rubric(req),
                                     student_answer=req.student_answer, max_marks=req.maximum_score)
    return EvaluateAnswerResponse(question=req.question, score=r.score,
                                  maximum_score=req.maximum_score, feedback=r.feedback,
                                  confidence=r.confidence, approach_used="embedding",
                                  point_breakdown=_breakdown(r.point_matches))


def _grade_with_hybrid(req: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
    """Rule lives in src/models/hybrid_grader.py (shared with the evaluation scripts)."""
    r = registry.hybrid.grade_one(req.question, req.reference_answer, _rubric(req),
                                  req.student_answer, req.maximum_score)
    return EvaluateAnswerResponse(
        question=req.question, score=r.score, maximum_score=req.maximum_score,
        feedback=r.feedback, confidence=r.confidence, approach_used="hybrid",
        needs_human_review=r.needs_human_review,
        component_scores={"baseline": r.baseline_score, "embedding": r.embedding_score},
        point_breakdown=_breakdown(r.point_matches))


def _grade(req: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
    if registry.hybrid is None:
        raise HTTPException(status_code=503, detail="Models are still loading; retry shortly.")
    try:
        if req.approach == "baseline":
            return _grade_with_baseline(req)
        if req.approach == "embedding":
            return _grade_with_embedding(req)
        return _grade_with_hybrid(req)
    except HTTPException:
        raise
    except Exception as e:  # never leak a raw stack trace
        raise HTTPException(status_code=400, detail=f"Could not grade this answer: {e}")


@app.post("/evaluate-answer", response_model=EvaluateAnswerResponse,
          responses={400: {"model": ErrorResponse}},
          summary="Grade one student answer against a rubric")
def evaluate_answer(req: EvaluateAnswerRequest):
    """
    Core endpoint. The first five response fields (question, score, maximum_score,
    feedback, confidence) match the handout's Section 5 example for this group;
    the rest are additive (approach_used, needs_human_review, component_scores,
    point_breakdown).
    """
    return _grade(req)


@app.post("/evaluate-answer/batch", response_model=list[EvaluateAnswerResponse],
          summary="Grade a list of student answers in one call")
def evaluate_answer_batch(reqs: list[EvaluateAnswerRequest]):
    if len(reqs) > 200:
        raise HTTPException(status_code=400, detail="Batch size limited to 200 per call.")
    return [_grade(r) for r in reqs]


@app.post("/grade-digitized-answers", response_model=GradeDigitizedResponse,
          summary="Grade answer records produced by Group 1 (Handwritten Examination Digitizer)")
def grade_digitized_answers(req: GradeDigitizedRequest):
    """
    Accepts records in Group 1's published schema ({student_id, question: "Q3",
    answer_text, page, confidence}) plus an exam key (question number -> question,
    reference answer, rubric, max marks). Multi-page answers are merged; answers
    with low OCR confidence, an empty OCR text, or a grader flag are marked for
    human review. See docs/API.md section 7.
    """
    if len(req.answers) > 500:
        raise HTTPException(status_code=400, detail="At most 500 answer records per call.")

    def grade_fn(spec: dict, text: str) -> dict:
        r = _grade(EvaluateAnswerRequest(
            question=spec["question"], reference_answer=spec["reference_answer"],
            rubric=spec["rubric"], student_answer=text,
            maximum_score=spec["maximum_score"], approach=req.approach))
        return r.model_dump()

    out = grade_digitized([a.model_dump() for a in req.answers], req.exam_key, grade_fn,
                          ocr_review_threshold=req.ocr_review_threshold)
    return GradeDigitizedResponse(exam_id=req.exam_id, **out)


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
def health():
    loaded = registry.hybrid is not None
    return HealthResponse(
        status="ok" if loaded else "loading",
        approaches_available=["hybrid", "baseline", "embedding"] if loaded else [],
        dataset_rows_used_for_calibration=registry.n_calibration_rows)


@app.get("/", summary="Basic info")
def root():
    return {"name": "AI Exam Answer Grader API", "group": "Group 2 - NLP Course Project 2026",
            "docs": "/docs", "demo": "/demo/", "core_endpoint": "POST /evaluate-answer",
            "integration_endpoint": "POST /grade-digitized-answers (Group 1 -> Group 2)",
            "default_approach": "hybrid"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")
if EXAMPLES_DIR.exists():
    app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")
