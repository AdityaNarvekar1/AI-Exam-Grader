"""
group1_to_group2_demo.py
------------------------
Group 1 (digitiser) -> Group 2 (grader) interaction demo.

  python examples/group1_to_group2_demo.py                       # uses the MOCK records
  python examples/group1_to_group2_demo.py path/to/real_group1_output.json

The input file is a JSON list of Group 1 records
({student_id, question, answer_text, page, confidence}). The default file is
MOCK data written by Group 2 in Group 1's published schema - replace it with
Group 1's real output once their service is available. Start the server first:
    uvicorn src.api.main:app --port 8000
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent

records_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "group1_sample_output.json"
body = {"exam_id": "DEMO-EXAM",
        "exam_key": json.loads((HERE / "exam_key_sample.json").read_text()),
        "answers": json.loads(records_path.read_text()),
        "approach": "hybrid", "ocr_review_threshold": 0.8}

req = urllib.request.Request(BASE_URL + "/grade-digitized-answers", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as resp:
    out = json.loads(resp.read())

print(f"{'student':<8} {'q':<3} {'pages':<7} {'OCR':<5} {'score':<9} {'grader conf':<12} review?  reasons")
for r in out["results"]:
    g = r["grade"]
    score = f"{g['score']}/{g['maximum_score']}" if g else "-"
    conf = f"{g['confidence']}" if g else "-"
    print(f"{r['student_id']:<8} {r['question']:<3} {str(r['pages']):<7} {r['ocr_confidence']:<5} "
          f"{score:<9} {conf:<12} {'YES' if r['needs_human_review'] else 'no':<8} {', '.join(r['review_reasons'])}")
print("\nsummary:", out["summary"])
