"""Three MT-quality metrics with genuinely different failure modes, meant to
be combined rather than relied on individually.

- BLEU (via sacrebleu): n-gram precision against a reference, with a
  brevity penalty. Real, standard implementation -- not reimplemented here.
  Weak on valid paraphrases (penalizes correct translations that just don't
  share BLEU's n-grams with the reference) and does not model synonymy or
  word order flexibly.
- METEOR (via nltk): aligns hypothesis and reference words allowing
  synonym/stem matches (via WordNet) and rewards word-order match via a
  fragmentation penalty. Real, standard implementation. Better than BLEU at
  catching valid paraphrases; weaker on adequacy (can reward a fluent
  hypothesis that doesn't actually preserve the source's meaning).
- Semantic similarity proxy (TF-IDF cosine): a stand-in for a trained
  neural quality-estimation metric like COMET. This is an HONEST
  simplification, not a hidden one: COMET is a pretrained cross-lingual
  regression model, and downloading pretrained models is not available in
  this environment (no Hugging Face Hub access). TF-IDF cosine over the
  hypothesis and reference captures lexical/semantic overlap using only
  local computation -- real code, but a materially weaker signal than an
  actual embedding-based metric. See README for what swapping in real COMET
  would look like.
"""
from __future__ import annotations

from dataclasses import dataclass

import sacrebleu
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MetricScores:
    bleu: float  # sacrebleu corpus/sentence BLEU, 0-100
    meteor: float  # 0-1
    semantic_similarity: float  # 0-1, TF-IDF cosine proxy for a neural QE metric


def bleu_score(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU via sacrebleu (0-100)."""
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score


def meteor_metric(hypothesis: str, reference: str) -> float:
    """METEOR via nltk (0-1). Requires the 'wordnet' nltk corpus."""
    return meteor_score([reference.split()], hypothesis.split())


def semantic_similarity_proxy(hypothesis: str, reference: str) -> float:
    """TF-IDF cosine similarity between hypothesis and reference (0-1). A
    local, zero-dependency stand-in for a neural quality-estimation metric
    -- see module docstring for why, and the README for the real-COMET swap
    path."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([reference, hypothesis])
    except ValueError:
        # Happens if both strings are empty or entirely stopword-free of
        # shared vocabulary after tokenization; treat as no similarity.
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def score_pair(hypothesis: str, reference: str) -> MetricScores:
    return MetricScores(
        bleu=bleu_score(hypothesis, reference),
        meteor=meteor_metric(hypothesis, reference),
        semantic_similarity=semantic_similarity_proxy(hypothesis, reference),
    )
