"""
call_api.py
-----------
Group 2 - AI Exam Answer Grader
Minimal example of calling POST /evaluate-answer from another group's code.
Standard library only (no `requests` needed).

1. Start the server (from the project root):
       uvicorn src.api.main:app --port 8000
   Wait for "Application startup complete" (8-20 s: it loads spaCy and fits
   the models once).
2. Run:
       python examples/call_api.py
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def post(path, body):
    req = urllib.request.Request(
        BASE_URL + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


request_body = {
    "question": "Explain the difference between a stack and a queue.",
    "reference_answer": "A stack is last-in-first-out; a queue is first-in-first-out.",
    # Write each rubric point as a COMPLETE SENTENCE (see docs/API_DOCUMENTATION.md,
    # "Writing good rubric points" - phrasing measurably changes the result).
    "rubric": [
        {"id": "p1", "marks": 2.5,
         "text": "A stack follows a last-in-first-out order, and a correct real-world example is given",
         # Optional: lets the fact-check layer catch swapped definitions.
         "binding": {"subject_terms": ["stack"],
                     "correct_value_terms": ["last-in-first-out", "lifo"],
                     "contradicting_value_terms": ["first-in-first-out", "fifo"]}},
        {"id": "p2", "marks": 2.5,
         "text": "A queue follows a first-in-first-out order, and a correct real-world example is given",
         "binding": {"subject_terms": ["queue"],
                     "correct_value_terms": ["first-in-first-out", "fifo"],
                     "contradicting_value_terms": ["last-in-first-out", "lifo"]}},
    ],
    "maximum_score": 5,
}

answers = {
    "correct": "A stack is last-in-first-out, like a pile of plates. A queue is "
               "first-in-first-out, like people waiting in a line.",
    "definitions swapped": "A stack is first-in-first-out, like people in a line, "
                           "while a queue is last-in-first-out, like a pile of plates.",
    "off-topic": "Our college cricket team won the tournament last week.",
}

for label, text in answers.items():
    result = post("/evaluate-answer", {**request_body, "student_answer": text})
    print(f"--- {label} ---")
    print(f"score: {result['score']} / {result['maximum_score']}   "
          f"confidence: {result['confidence']}   needs_human_review: {result['needs_human_review']}")
    print(f"feedback: {result['feedback']}\n")
