"""End-to-end demo: score a small set of (reference, hypothesis) MT pairs
with BLEU, METEOR, and the semantic-similarity proxy, flag low-quality pairs
by ensemble agreement, and contrast that against flagging on a single metric
alone.

The pairs below are hand-constructed to exercise each metric's known failure
mode:

- pair 2 is a valid paraphrase (correct meaning, different wording) -- BLEU
  alone would likely flag it as low quality; METEOR and the semantic proxy
  should recognize it's fine, so the ensemble should NOT flag it.
- pair 3 is fluent but says something different from the reference (a
  believable mistranslation) -- this should get flagged by all three.
- pair 4 is a truncated/incomplete translation -- BLEU's brevity penalty and
  the other metrics should catch this.
- pairs 1 and 5 are straightforwardly correct, near-identical translations.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import TranslationPair, compare_to_single_metric, evaluate_corpus

PAIRS = [
    TranslationPair(
        "p1",
        source="(src omitted for demo brevity)",
        reference="The technician will arrive between 9 and 11 tomorrow morning.",
        hypothesis="The technician will arrive between 9 and 11 tomorrow morning.",
    ),
    TranslationPair(
        "p2_paraphrase",
        source="(src omitted for demo brevity)",
        reference="Your order number 48213 has shipped and should arrive within three to five business days.",
        hypothesis="Package 48213 is now on the way to you, delivery expected in three to five business days.",
    ),
    TranslationPair(
        "p3_mistranslation",
        source="(src omitted for demo brevity)",
        reference="The technician will arrive between 9 and 11 tomorrow morning.",
        hypothesis="Your refund has already been processed and should appear within five days.",
    ),
    TranslationPair(
        "p4_truncated",
        source="(src omitted for demo brevity)",
        reference="The technician will arrive between 9 and 11 tomorrow morning, please make sure someone is home.",
        hypothesis="The technician will arrive.",
    ),
    TranslationPair(
        "p5",
        source="(src omitted for demo brevity)",
        reference="Please restart the router and wait two minutes before reconnecting.",
        hypothesis="Please restart the router and wait two minutes before reconnecting.",
    ),
]


def main() -> None:
    results = evaluate_corpus(PAIRS)

    print(f"{'pair_id':<20} {'BLEU':>7} {'METEOR':>8} {'sem_sim':>8} {'flagged':>9}  votes")
    for r in results:
        votes_str = ",".join(m for m, v in r.metric_votes.items() if v) or "-"
        print(
            f"{r.pair_id:<20} {r.scores.bleu:>7.1f} {r.scores.meteor:>8.2f} "
            f"{r.scores.semantic_similarity:>8.2f} {str(r.flagged):>9}  {votes_str}"
        )

    print(
        "\nNote pair 2 (a valid paraphrase, low n-gram overlap): BLEU alone "
        "falls below its floor here because the wording barely overlaps "
        "with the reference, but METEOR and the semantic-similarity proxy "
        "both recognize the meaning is preserved and stay above their "
        "floors -- only 1 of 3 metrics votes to flag it, so the ensemble "
        "correctly does not."
    )

    comparison = compare_to_single_metric(results, single_metric="bleu")
    print("\nEnsemble vs. BLEU-only flagging:")
    print(f"  Flagged by ensemble:      {sorted(comparison['ensemble_flagged'])}")
    print(f"  Flagged by BLEU alone:    {sorted(comparison['single_metric_flagged'])}")
    print(f"  Caught only by ensemble:  {sorted(comparison['caught_only_by_ensemble'])}")
    print(f"  Caught only by BLEU:      {sorted(comparison['caught_only_by_single_metric'])}")
    print(
        "  -> BLEU-only flagging catches pair 2 as a false positive (a "
        "valid paraphrase) that the ensemble correctly lets through; this "
        "is the concrete version of 'ensemble flags vs. a single metric, "
        "compared for agreement' used to justify the ensemble approach "
        "over relying on one metric."
    )


if __name__ == "__main__":
    main()
