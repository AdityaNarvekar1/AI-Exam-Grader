"""
verify_sentence_transformer_backend.py
----------------------------------------
Group 2 - AI Exam Answer Grader
Run this ONCE, locally, after installing sentence-transformers and
downloading all-MiniLM-L6-v2, to confirm the swap works before re-running
the full Week 3/4 evaluations. This could not be run in the sandbox this
project was built in (huggingface.co was unreachable there).

Usage:
    pip install sentence-transformers
    python src/evaluation/verify_sentence_transformer_backend.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.embedding_grader import SentenceTransformerBackend, EmbeddingSemanticGrader  # noqa: E402


def main():
    print("Loading all-MiniLM-L6-v2 (downloads ~90MB on first run)...")
    backend = SentenceTransformerBackend()
    print("Model loaded OK.\n")

    # Sanity check 1: sentence splitting
    sents = backend._split_sentences("A stack is LIFO. A queue is FIFO!")
    assert len(sents) == 2, f"Expected 2 sentences, got {sents}"
    print(f"Sentence splitting OK: {sents}")

    # Sanity check 2: the exact paraphrase case that stumped the spaCy
    # backend in Week 3 (PY01) - a real sentence-transformer should score
    # this noticeably higher than the 0.697 spaCy similarity we measured.
    point_text = "List is mutable, tuple is immutable"
    paraphrase = ("One of these two collection types can have its contents "
                  "rearranged after you build it, whereas the other is "
                  "locked in place the moment you create it.")
    point_vec = backend.vector(point_text)
    for sent, vec in backend.sentences_with_vectors(paraphrase):
        import numpy as np
        sim = float(np.dot(point_vec, vec))
        print(f"Similarity to '{sent[:60]}...': {sim:.3f}  "
              f"(spaCy backend scored 0.697 on this exact pair)")

    # Sanity check 3: run it through the full grader
    print("\nRunning full EmbeddingSemanticGrader with the new backend...")
    train_rows = [{
        "rubric": '[{"id": "p1", "text": "List is mutable, tuple is immutable", "marks": 2.0}]',
        "student_answer": "A list can be changed after creation, a tuple cannot.",
        "points_covered": '["p1"]',
    }, {
        "rubric": '[{"id": "p1", "text": "List is mutable, tuple is immutable", "marks": 2.0}]',
        "student_answer": "Python has many data types used for programming.",
        "points_covered": '[]',
    }]
    grader = EmbeddingSemanticGrader(backend=backend).fit(train_rows)
    import json
    result = grader.grade_one("", json.loads(train_rows[0]["rubric"]), paraphrase, 2.0, "PY01", "demo")
    print(f"Score: {result.score}/2.0  Confidence: {result.confidence}  Feedback: {result.feedback}")
    print("\nIf the paraphrase now scores > 0 (ideally close to 2.0), the swap is working "
          "and closing the gap the spaCy backend had on this exact case.")
    print("Next: re-run 'python src/evaluation/compare_models.py hard sentence_transformer' "
          "for the full Week 4 comparison with the new backend.")


if __name__ == "__main__":
    main()
