"""Validation-tuned reranking for FELIX++ edit candidates."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

from ..metrics import compute_metrics
from .plus_decode import EditCandidate


@dataclass(frozen=True)
class RerankWeights:
    values: Dict[str, float]


def score_candidate(candidate: EditCandidate, weights: RerankWeights) -> float:
    return sum(weights.values.get(name, 0.0) * value for name, value in candidate.features.items())


def choose_candidate(candidates: Sequence[EditCandidate], weights: RerankWeights) -> EditCandidate:
    if not candidates:
        raise ValueError("choose_candidate requires at least one candidate")
    return max(candidates, key=lambda c: (score_candidate(c, weights), c.text))


def _metric_value(preds: List[str], refs: List[str], metric: str) -> float:
    m = compute_metrics(preds, refs)
    if metric == "wer":
        return -m["wer"]
    if metric == "exact_match":
        return m["exact_match"]
    return m.get(metric, m["bleu"])


def select_predictions(candidate_groups: Sequence[Sequence[EditCandidate]], weights: RerankWeights) -> List[str]:
    return [choose_candidate(group, weights).text for group in candidate_groups]


def tune_weights(
    candidate_groups: Sequence[Sequence[EditCandidate]],
    refs: Sequence[str],
    feature_names: Sequence[str],
    grid: Sequence[float],
    metric: str = "wer",
) -> RerankWeights:
    if len(candidate_groups) != len(refs):
        raise ValueError("candidate_groups and refs must have the same length")

    best = RerankWeights({name: 0.0 for name in feature_names})
    best_value = float("-inf")
    for values in product(grid, repeat=len(feature_names)):
        weights = RerankWeights(dict(zip(feature_names, values)))
        preds = [choose_candidate(group, weights).text for group in candidate_groups]
        value = _metric_value(preds, list(refs), metric)
        if value > best_value:
            best = weights
            best_value = value
    return best
