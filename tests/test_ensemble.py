import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import (
    Thresholds,
    TranslationPair,
    compare_to_single_metric,
    evaluate_corpus,
    evaluate_pair,
)
from src.metrics import bleu_score, meteor_metric, semantic_similarity_proxy


def test_bleu_identical_sentences_is_max_score():
    score = bleu_score("the cat sat on the mat", "the cat sat on the mat")
    assert score == pytest.approx(100.0)


def test_bleu_unrelated_sentences_is_low():
    score = bleu_score("purple elephants dance quietly", "the invoice was paid on time")
    assert score < 10.0


def test_meteor_identical_sentences_near_one():
    score = meteor_metric("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.95


def test_meteor_unrelated_sentences_is_low():
    score = meteor_metric("purple elephants dance quietly", "the invoice was paid on time")
    assert score < 0.2


def test_semantic_similarity_identical_sentences_is_one():
    score = semantic_similarity_proxy("the cat sat on the mat", "the cat sat on the mat")
    assert score == pytest.approx(1.0)


def test_semantic_similarity_unrelated_sentences_is_low():
    score = semantic_similarity_proxy("purple elephants dance quietly", "the invoice was paid on time")
    assert score < 0.1


def test_evaluate_pair_identical_not_flagged():
    pair = TranslationPair("t1", source="", reference="the router will restart", hypothesis="the router will restart")
    result = evaluate_pair(pair)
    assert result.flagged is False
    assert result.reasons == []


def test_evaluate_pair_unrelated_hypothesis_flagged_by_all_three():
    pair = TranslationPair(
        "t2",
        source="",
        reference="the technician will arrive tomorrow morning",
        hypothesis="your refund has already been processed",
    )
    result = evaluate_pair(pair)
    assert result.flagged is True
    assert all(result.metric_votes.values())


def test_evaluate_pair_respects_min_votes_threshold():
    pair = TranslationPair(
        "t3",
        source="",
        reference="the technician will arrive tomorrow morning",
        hypothesis="the technician will arrive tomorrow morning",
    )
    # Force every metric to "vote" flagged by setting floors above the max.
    # possible score, but require all 3 votes to flag.
    thresholds = Thresholds(bleu_floor=101.0, meteor_floor=1.1, semantic_floor=1.1, min_votes_to_flag=3)
    result = evaluate_pair(pair, thresholds)
    assert result.flagged is True
    assert len(result.reasons) == 3


def test_evaluate_corpus_returns_one_result_per_pair():
    pairs = [
        TranslationPair("a", source="", reference="hello world", hypothesis="hello world"),
        TranslationPair("b", source="", reference="hello world", hypothesis="goodbye moon"),
    ]
    results = evaluate_corpus(pairs)
    assert len(results) == 2
    assert {r.pair_id for r in results} == {"a", "b"}


def test_compare_to_single_metric_identifies_ensemble_only_catches():
    pairs = [
        TranslationPair("a", source="", reference="hello world", hypothesis="hello world"),
        TranslationPair("b", source="", reference="hello world", hypothesis="goodbye moon entirely different"),
    ]
    results = evaluate_corpus(pairs)
    comparison = compare_to_single_metric(results, single_metric="bleu")
    assert "b" in comparison["ensemble_flagged"]
    assert comparison["ensemble_flagged"] | comparison["caught_only_by_single_metric"] >= comparison["ensemble_flagged"]
