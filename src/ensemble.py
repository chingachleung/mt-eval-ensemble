"""Ensemble low-quality-pair flagging: combine BLEU, METEOR, and the
semantic-similarity proxy so that no single metric's blind spot determines
the outcome. Hence ensemble metric.

Each metric has a different failure mode (see metrics.py), so a pair that
fools one metric usually doesn't fool the others the same way. Flagging on
*majority agreement* (>=2 of 3 metrics below their floor) rather than any
single metric is the same "agreement between independent signals is more
trustworthy than any one signal" idea behind flagging noisy/low-quality
training pairs with an ensemble instead of a single scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import MetricScores, score_pair


@dataclass
class TranslationPair:
    pair_id: str
    source: str
    reference: str
    hypothesis: str


@dataclass
class FlagResult:
    pair_id: str
    scores: MetricScores
    metric_votes: dict[str, bool]  # which individual metrics flagged this pair as low quality
    flagged: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Thresholds:
    bleu_floor: float = 30.0  # sacrebleu 0-100 scale
    meteor_floor: float = 0.30  # 0-1 scale
    semantic_floor: float = 0.20  # 0-1 scale
    min_votes_to_flag: int = 2  # out of 3 metrics


def evaluate_pair(pair: TranslationPair, thresholds: Thresholds = Thresholds()) -> FlagResult:
    scores = score_pair(pair.hypothesis, pair.reference)

    votes = {
        "bleu": scores.bleu < thresholds.bleu_floor,
        "meteor": scores.meteor < thresholds.meteor_floor,
        "semantic_similarity": scores.semantic_similarity < thresholds.semantic_floor,
    }
    n_votes = sum(votes.values())
    flagged = n_votes >= thresholds.min_votes_to_flag

    reasons = [f"{metric} below floor" for metric, voted in votes.items() if voted]

    return FlagResult(pair_id=pair.pair_id, scores=scores, metric_votes=votes, flagged=flagged, reasons=reasons)


def evaluate_corpus(pairs: list[TranslationPair], thresholds: Thresholds = Thresholds()) -> list[FlagResult]:
    return [evaluate_pair(pair, thresholds) for pair in pairs]


def compare_to_single_metric(results: list[FlagResult], single_metric: str = "semantic_similarity") -> dict:
    """Contrast the ensemble's flagged set against what flagging on
    `single_metric` alone would have caught -- makes the "why ensemble, not
    just one metric" case concrete rather than asserted."""
    ensemble_flagged = {r.pair_id for r in results if r.flagged}
    single_flagged = {r.pair_id for r in results if r.metric_votes.get(single_metric, False)}

    return {
        "ensemble_flagged": ensemble_flagged,
        "single_metric_flagged": single_flagged,
        "caught_only_by_ensemble": ensemble_flagged - single_flagged,
        "caught_only_by_single_metric": single_flagged - ensemble_flagged,
        "agreement": ensemble_flagged & single_flagged,
    }
