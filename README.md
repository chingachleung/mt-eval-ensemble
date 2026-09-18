# mt-eval-ensemble

[![tests](https://github.com/chingachleung/mt-eval-ensemble/actions/workflows/tests.yml/badge.svg)](https://github.com/chingachleung/mt-eval-ensemble/actions/workflows/tests.yml)

A small, runnable machine-translation quality pipeline: score (reference,
hypothesis) pairs with three metrics that fail in different ways (BLEU,
METEOR, and a semantic-similarity proxy), flag low-quality pairs by ensemble
agreement instead of a single metric, and compare that against flagging on
one metric alone.

This is an original, from-scratch demo — not production code — built to show
the same idea behind an ensembled multi-metric MT-quality flagging system I
built professionally: no single automatic metric is trustworthy enough alone
to flag low-quality translation pairs, but agreement across metrics with
different blind spots is a much stronger signal.

## Why this design

- **Real BLEU and METEOR, not reimplementations.** `bleu_score` calls
  `sacrebleu`; `meteor_metric` calls `nltk`'s METEOR implementation
  (WordNet-based synonym/stem matching). Both are standard, correct
  implementations of well-known metrics.
- **An honest, labeled stand-in for COMET.** A neural quality-estimation
  metric like COMET is a pretrained cross-lingual regression model, and
  downloading pretrained models isn't available in this environment (no
  Hugging Face Hub access here). Rather than skip that leg of the ensemble
  or silently fake it, `semantic_similarity_proxy` is real, working
  TF-IDF-cosine code that plays the same *role* (a non-n-gram signal of
  meaning overlap) while being explicit in its docstring and this README
  that it is a materially weaker signal than actual COMET. See "Swapping in
  real COMET" below for what activating the real thing would look like.
- **Ensemble by vote, not by averaging.** `evaluate_pair` flags a pair only
  when at least 2 of the 3 metrics independently fall below their floor —
  not when a blended average score is low. This means a pair that fools one
  metric (e.g. BLEU penalizing a valid paraphrase for low n-gram overlap)
  doesn't get wrongly flagged just because that one metric was pessimistic.
  `examples/run_demo.py` shows this concretely: a valid paraphrase pair gets
  flagged by BLEU alone but correctly waved through by the ensemble, while a
  real mistranslation gets flagged by all three.
- **Ensemble-vs-single-metric comparison, not just an aggregate score.**
  `compare_to_single_metric` reports which pairs each approach flags, so the
  value of ensembling is demonstrated on specific examples rather than
  asserted.

## What's real vs. mocked

| Component | Status |
|---|---|
| `bleu_score` (sacrebleu) | Real, standard implementation |
| `meteor_metric` (nltk, WordNet-based) | Real, standard implementation |
| `semantic_similarity_proxy` (TF-IDF cosine) | Real code, but an **explicit stand-in** for a neural QE metric like COMET — see above |
| Ensemble voting / flagging logic | Real, fully functional |
| The 5-pair demo corpus in `examples/run_demo.py` | Fictional, hand-constructed to exercise each metric's known failure mode |

### Swapping in real COMET

`semantic_similarity_proxy` and `MetricScores.semantic_similarity` are the
only places a real COMET call would plug in — the `Thresholds`,
`evaluate_pair`, and voting logic don't need to change. Swapping it in looks
like: `pip install unbabel-comet`, load a pretrained checkpoint (e.g.
`Unbabel/wmt22-comet-da`) with `comet.load_from_checkpoint(comet.download_model(...))`,
and replace the TF-IDF cosine computation with `model.predict([{"src":
source, "mt": hypothesis, "ref": reference}])`. Note COMET also uses the
source sentence (not just reference vs. hypothesis), which this proxy
ignores since TF-IDF has no notion of cross-lingual alignment.

## Run it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python examples/run_demo.py
```

### Example output

```
pair_id                 BLEU   METEOR  sem_sim   flagged  votes
p1                     100.0     1.00     1.00     False  -
p2_paraphrase            25.9     0.39     0.21     False  bleu
p3_mistranslation        3.7     0.05     0.03      True  bleu,meteor,semantic_similarity
p4_truncated             5.3     0.20     0.37      True  bleu,meteor
p5                     100.0     1.00     1.00     False  -

Ensemble vs. BLEU-only flagging:
  Flagged by ensemble:      ['p3_mistranslation', 'p4_truncated']
  Flagged by BLEU alone:    ['p2_paraphrase', 'p3_mistranslation', 'p4_truncated']
  Caught only by BLEU:      ['p2_paraphrase']
```

`p2_paraphrase` is the case that matters here: it's a valid, correctly
reworded translation, but BLEU's n-gram matching drops it below the floor
anyway. METEOR and the semantic-similarity proxy both stay above their
floors, so the 2-of-3 ensemble vote correctly does not flag it — while a
BLEU-only pipeline would file it as a false-positive data-quality issue.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Layout

```
src/
  metrics.py      # bleu_score, meteor_metric, semantic_similarity_proxy
  ensemble.py       # TranslationPair, Thresholds, evaluate_pair/corpus, compare_to_single_metric
examples/run_demo.py # 5-pair demo: scoring, flagging, ensemble-vs-BLEU-only comparison
tests/                # pytest suite
```
