"""
model_loader.py
----------------
Group 2 - AI Exam Answer Grader
Loads and fits both models ONCE at API startup (spaCy loading and model
fitting are too slow to redo on every request). For the deployed asset we
calibrate on ALL labelled historical data (train + validation + test
splits combined) rather than holding out a test set - the held-out test
split was only needed for the Week 2-5 evaluation reports; a deployed
model should learn from everything available.
"""

import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.baseline_grader import TFIDFBaselineGrader          # noqa: E402
from models.embedding_grader import EmbeddingSemanticGrader, get_backend  # noqa: E402
from models.hybrid_grader import HybridGrader  # noqa: E402

DATA_PATH = SRC_DIR.parent / "data" / "exam_grading_dataset.csv"


class ModelRegistry:
    """Holds both fitted graders; instantiated once by the FastAPI app at startup."""

    def __init__(self):
        self.baseline = None
        self.embedding = None
        self.hybrid = None
        self.n_calibration_rows = 0

    def load(self):
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        self.n_calibration_rows = len(all_rows)

        self.baseline = TFIDFBaselineGrader().fit(all_rows)
        self.embedding = EmbeddingSemanticGrader(backend=get_backend("spacy")).fit(all_rows)
        self.hybrid = HybridGrader(self.baseline, self.embedding)
        return self


registry = ModelRegistry()
