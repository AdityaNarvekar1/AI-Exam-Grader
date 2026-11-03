"""
schemas.py
----------
Group 2 - AI Exam Answer Grader
Pydantic request/response models for POST /evaluate-answer.

The response schema matches, field-for-field, the exact example given in
the course handout's Section 5 for this group:

    {
      "question": "Q4",
      "score": 7,
      "maximum_score": 10,
      "feedback": "The answer correctly explains...",
      "confidence": 0.87
    }

Extra fields (approach_used, point_breakdown) are additive, not a
replacement - any caller that only reads the five fields above still gets
exactly the documented contract; callers who want the richer, explainable
per-rubric-point breakdown can read point_breakdown too.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class RubricPoint(BaseModel):
    id: str = Field(..., description="Short unique identifier for this rubric point, e.g. 'p1'.")
    text: str = Field(..., description="The concept this point checks for, in a full sentence "
                                        "(not a fragment) - see Week 3 report for why phrasing matters.")
    marks: float = Field(..., gt=0, description="Marks awarded if this point is credited.")
    binding: Optional[dict] = Field(
        None, description="Advanced/optional: for rubric points with a clear two-way "
                           "entity<->attribute distinction (e.g. 'stack is LIFO' vs 'queue is "
                           "FIFO'), enables the fact-binding veto layer that catches factual "
                           "inversions embedding similarity alone misses. See "
                           "src/models/fact_binding_checker.py. Safe to omit.")
    cues: Optional[dict] = Field(
        None, description="Advanced/optional: for relationship-type rubric points (e.g. a "
                           "tradeoff), a confirming/contradicting phrase list for the same "
                           "veto layer. Safe to omit.")


class EvaluateAnswerRequest(BaseModel):
    question: str = Field(..., description="The exam question text.")
    reference_answer: str = Field(..., description="The examiner's model/reference answer.")
    rubric: List[RubricPoint] = Field(..., min_length=1,
                                       description="Mark-weighted list of concepts the answer should cover.")
    student_answer: str = Field(..., description="The student's answer to grade.")
    maximum_score: float = Field(..., gt=0, description="Maximum marks available for this question.")
    approach: Literal["hybrid", "embedding", "baseline"] = Field(
        "hybrid", description="'hybrid' (default): plain average of the two approaches below; "
                               "on questions the model had never seen it was reliably better than "
                               "'embedding' alone and not distinguishable from 'baseline' overall, "
                               "while adding per-concept feedback, a fact-check veto signal, and a "
                               "confidence that empirically tracks error (Week 6 report). "
                               "'embedding': pretrained-embedding per-concept matching - its own "
                               "confidence was NOT informative about error in testing. "
                               "'baseline': TF-IDF + keyword overlap.")

    @model_validator(mode="after")
    def rubric_marks_must_not_exceed_maximum(self):
        # model_validator(mode="after") runs once ALL fields are populated.
        # (A field_validator on `rubric` ran before `maximum_score` existed,
        # because Pydantic validates in field-definition order, so the check
        # was silently skipped - caught by an API test, fixed here.)
        total = sum(p.marks for p in self.rubric)
        if total > self.maximum_score + 1e-6:
            raise ValueError(f"Sum of rubric marks ({total}) exceeds maximum_score ({self.maximum_score}).")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Explain the difference between a list and a tuple in Python.",
                "reference_answer": "A list is mutable... a tuple is immutable...",
                "rubric": [
                    {"id": "p1", "text": "A list is mutable, meaning it can be changed after "
                                          "creation, while a tuple is immutable and cannot be changed",
                     "marks": 2.5},
                    {"id": "p2", "text": "Tuples are faster and hashable, so they can be used "
                                          "as dictionary keys, unlike lists", "marks": 2.5},
                ],
                "student_answer": "A list can be changed after it's made but a tuple can't.",
                "maximum_score": 5,
                "approach": "hybrid",
            }
        }
    }


class PointBreakdown(BaseModel):
    point_id: str
    point_text: str
    marks: float
    credited: bool
    similarity: float
    best_matched_sentence: str
    fact_check: str = Field(..., description="'confirmed' | 'contradicted' | 'neutral' | 'not_applicable'.")


class EvaluateAnswerResponse(BaseModel):
    # --- core contract fields (match the handout's Section 5 example exactly) ---
    question: str
    score: float
    maximum_score: float
    feedback: str
    confidence: float = Field(
        ..., ge=0, le=1,
        description="For approach='hybrid': agreement between the two approaches, "
                    "1 - |baseline - embedding| / maximum_score. A heuristic, NOT a calibrated "
                    "probability; on unseen questions it correlated with lower error "
                    "(Spearman +0.61). For 'baseline' it was moderately informative (+0.44). "
                    "For 'embedding' alone it was NOT informative (-0.15, n.s.) - treat that "
                    "number as unvalidated.")

    # --- additive, optional-to-use fields ---
    approach_used: str
    needs_human_review: bool = Field(
        False, description="True when confidence < 0.7 (approaches disagree by >30% of the "
                            "marks) OR the fact-check layer detected a contradicted rubric point. "
                            "Only computed for approach='hybrid'; always False otherwise.")
    component_scores: Optional[dict] = Field(
        None, description="For approach='hybrid': the two underlying scores that were averaged.")
    point_breakdown: Optional[List[PointBreakdown]] = None


class HealthResponse(BaseModel):
    status: str
    approaches_available: List[str]
    dataset_rows_used_for_calibration: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ---------------------------------------------------------------------------
# Week 7: integration with Group 1 (Handwritten Examination Digitizer)
# ---------------------------------------------------------------------------
class ExamKeyEntry(BaseModel):
    question: str
    question_type: Literal["short", "mcq"] = "short"
    reference_answer: Optional[str] = None
    rubric: Optional[List[RubricPoint]] = None
    correct_option: Optional[str] = Field(None, description="MCQ only: the correct letter, e.g. 'B'.")
    maximum_score: float = Field(..., gt=0)

    @model_validator(mode="after")
    def marks_within_maximum(self):
        if self.question_type == "mcq":
            if not self.correct_option:
                raise ValueError("An MCQ exam-key entry needs correct_option.")
            return self
        if not self.reference_answer or not self.rubric:
            raise ValueError("A short-answer exam-key entry needs reference_answer and a non-empty rubric.")
        if sum(p.marks for p in self.rubric) > self.maximum_score + 1e-6:
            raise ValueError("Sum of rubric marks exceeds maximum_score.")
        return self


class Group1Record(BaseModel):
    """One answer segment from Group 1. Accepts the published schema (answer_text, confidence) and the
    real export (ocr_answer, status, question_type, needs_review, placed_by_content)."""
    student_id: str
    question: str = Field(..., description="Question NUMBER, e.g. 'Q3' (not the question text).")
    answer_text: Optional[str] = None
    ocr_answer: Optional[str] = None
    page: Optional[int] = None
    confidence: Optional[float] = Field(1.0, ge=0, le=1, description="Group 1's OCR confidence.")
    status: Optional[Literal["found", "not_found"]] = None
    question_type: Optional[Literal["short", "mcq"]] = None
    needs_review: Optional[str] = None
    placed_by_content: Optional[str] = None

    @field_validator("page", "confidence", mode="before")
    @classmethod
    def empty_string_is_none(cls, v):
        return None if v == "" else v

    @model_validator(mode="after")
    def has_text_field(self):
        if self.answer_text is None and self.ocr_answer is None and self.status != "not_found":
            raise ValueError("Record needs answer_text (published schema) or ocr_answer (Group 1 export).")
        return self


class GradeDigitizedRequest(BaseModel):
    exam_id: Optional[str] = None
    exam_key: dict = Field(..., description="Map of question number -> {question, reference_answer, "
                                             "rubric, maximum_score}. Neither group produces this today; "
                                             "faculty supply it.")
    answers: List[Group1Record]
    approach: Literal["hybrid", "embedding", "baseline"] = "hybrid"
    ocr_review_threshold: float = Field(0.8, ge=0, le=1, description="OCR confidence below this "
                                                                      "sends the answer to a human.")

    @field_validator("exam_key")
    @classmethod
    def validate_exam_key(cls, v):
        return {qid: ExamKeyEntry(**entry).model_dump(exclude_none=True) for qid, entry in v.items()}

    model_config = {"json_schema_extra": {"example": {
        "exam_id": "DS-MIDSEM-2026",
        "exam_key": {"Q1": {
            "question": "Explain the difference between a stack and a queue.",
            "reference_answer": "A stack is LIFO; a queue is FIFO.",
            "rubric": [{"id": "p1", "marks": 2.5, "text": "A stack follows a last-in-first-out order, and a correct real-world example is given"},
                       {"id": "p2", "marks": 2.5, "text": "A queue follows a first-in-first-out order, and a correct real-world example is given"}],
            "maximum_score": 5}},
        "answers": [{"student_id": "23D001", "question": "Q1", "page": 2, "confidence": 0.91,
                     "answer_text": "A stack is last in first out like plates, a queue is first in first out like a line."}],
        "approach": "hybrid", "ocr_review_threshold": 0.8}}}


class McqGrade(BaseModel):
    question: str
    score: float
    maximum_score: float
    feedback: str
    confidence: float
    needs_human_review: bool
    approach_used: Literal["mcq_exact_match"]
    selected_option: str
    correct_option: str


class DigitizedItemResult(BaseModel):
    student_id: str
    question: str
    pages: List[Optional[int]]
    ocr_confidence: float
    answer_text: str
    status: str
    grade: Optional[Union[EvaluateAnswerResponse, McqGrade]] = None
    needs_human_review: bool
    review_reasons: List[str]


class GradeDigitizedResponse(BaseModel):
    exam_id: Optional[str] = None
    results: List[DigitizedItemResult]
    summary: dict
