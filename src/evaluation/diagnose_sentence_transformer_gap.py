"""
diagnose_sentence_transformer_gap.py
--------------------------------------
Group 2 - AI Exam Answer Grader
Diagnostic for an unexpected finding: all-MiniLM-L6-v2 (via
SentenceTransformerBackend) scored WORSE than the spaCy static-embedding
backend on our rubric-concept-matching task (F1 0.703 vs 0.765; the two
paraphrase cases spaCy fixed regressed to 0 credit).

This script isolates ONE hypothesis: rubric points are terse, fragment-like
phrases ("List is mutable, tuple is immutable"), not full natural
sentences - and MiniLM, trained mostly on natural sentence pairs
(STS/NLI), may embed that style of text less reliably than simple
word-vector averaging does. It runs three checks:

  1. Sanity check - a known-easy, well-formed sentence pair. If MiniLM
     scores this low too, something is wrong with the setup itself, not
     the fragment-vs-sentence hypothesis.
  2. The exact failing case from Week 3/4 (rubric fragment vs. paraphrase).
  3. The SAME rubric concept, rephrased as a full grammatical sentence,
     against the same paraphrase - to see if that recovers similarity.

Run this after src/evaluation/verify_sentence_transformer_backend.py has
already confirmed the model loads correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.embedding_grader import SentenceTransformerBackend  # noqa: E402


def cos(a, b):
    import numpy as np
    return float(np.dot(a, b))


def main():
    backend = SentenceTransformerBackend()

    print("=== 1. Sanity check: is MiniLM working normally at all? ===")
    pairs = [
        ("A cat sat on the mat.", "A cat was sitting on the mat."),
        ("The stock market fell sharply today.", "Equities dropped significantly this afternoon."),
    ]
    for a, b in pairs:
        sim = cos(backend.vector(a), backend.vector(b))
        print(f"  {sim:.3f}  |  '{a}'  vs  '{b}'")
    print("  (Expect > 0.7 for these well-formed paraphrase pairs. If not, "
          "something more basic is wrong - stop here and report back.)\n")

    print("=== 2. The exact Week 3/4 failing case (fragment vs. paraphrase) ===")
    fragment = "List is mutable, tuple is immutable"
    paraphrase = (
        "One of these two collection types can have its contents "
        "rearranged after you build it, whereas the other is locked in "
        "place the moment you create it, so its items can never later "
        "be reordered or changed."
    )
    sim_fragment = cos(backend.vector(fragment), backend.vector(paraphrase))
    print(f"  {sim_fragment:.3f}  |  fragment: '{fragment}'")
    print(f"          vs paraphrase: '{paraphrase[:70]}...'\n")

    print("=== 3. Same concept, rephrased as a full grammatical sentence ===")
    full_sentence = "A list is mutable, meaning it can be changed after it is created, while a tuple is immutable and cannot be changed once it is made."
    sim_full = cos(backend.vector(full_sentence), backend.vector(paraphrase))
    print(f"  {sim_full:.3f}  |  full sentence: '{full_sentence}'")
    print(f"          vs paraphrase: '{paraphrase[:70]}...'\n")

    print("=== Verdict ===")
    if sim_full > sim_fragment + 0.15:
        print(f"  HYPOTHESIS CONFIRMED: rephrasing the rubric point as a full sentence "
              f"raised similarity from {sim_fragment:.3f} to {sim_full:.3f}.")
        print("  Fix: rewrite rubric point texts in data as full sentences, not fragments, "
              "before re-running the sentence_transformer backend.")
    elif sim_full > sim_fragment:
        print(f"  PARTIAL SUPPORT: rephrasing helped a little ({sim_fragment:.3f} -> {sim_full:.3f}) "
              "but doesn't fully explain the gap. Other factors likely contribute too "
              "(e.g. MiniLM's general anisotropy/similarity compression, or this model "
              "simply isn't well-suited to short-concept-vs-long-explanation matching "
              "without fine-tuning).")
    else:
        print(f"  HYPOTHESIS NOT SUPPORTED: rephrasing did not help ({sim_fragment:.3f} -> {sim_full:.3f}). "
              "The fragment-vs-sentence style mismatch is not the (main) explanation - "
              "worth reporting as an open question in the Week 4 report rather than "
              "asserting a specific cause.")


if __name__ == "__main__":
    main()
