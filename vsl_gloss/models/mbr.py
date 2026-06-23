"""Minimum Bayes Risk (MBR) selection over a pool of candidate hypotheses.

Given an unbiased sample of hypotheses drawn from one or more models, MBR picks
the single hypothesis that maximises expected utility against the rest of the
pool (Eikema & Aziz, *Is MAP Decoding All You Need?*, EMNLP 2020)::

    h* = argmax_h  E_{y~pool}[ utility(h, y) ]

The pool is a Monte-Carlo estimate of the model distribution, so the selected
hypothesis is the *consensus* one — the output the sample agrees is most
central. Pooling samples from several models turns this into a principled
ensemble that needs no shared vocabulary (we never average logits).

This module is deliberately framework-free (pure Python + an optional sacreBLEU
utility) so the selection logic is unit-testable without a GPU.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Optional, Sequence

try:  # sentence-level utilities match the corpus-level metrics when available
    import sacrebleu  # type: ignore

    _HAS_SACREBLEU = True
except ImportError:  # pragma: no cover
    _HAS_SACREBLEU = False


# --------------------------------------------------------------------------- utilities
def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, char_order: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-beta (chrF). Pure-Python fall-back matching sacreBLEU's
    defaults closely enough for ranking candidates (char_order=6, beta=2)."""
    if hyp == ref:                       # identical (incl. both empty) -> perfect
        return 100.0
    if not hyp or not ref:               # exactly one side empty -> no overlap
        return 0.0
    if _HAS_SACREBLEU:
        try:
            return sacrebleu.sentence_chrf(hyp, [ref]).score
        except Exception:                # never let a metric edge-case crash MBR
            pass
    precs, recs = [], []
    for n in range(1, char_order + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        overlap = sum(min(c, r[g]) for g, c in h.items())
        hn, rn = sum(h.values()), sum(r.values())
        if hn:
            precs.append(overlap / hn)
        if rn:
            recs.append(overlap / rn)
    if not precs or not recs:
        return 0.0
    p = sum(precs) / len(precs)
    rc = sum(recs) / len(recs)
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * rc / (b2 * p + rc)


def bleu(hyp: str, ref: str) -> float:
    if hyp == ref:
        return 100.0
    if not hyp or not ref:
        return 0.0
    if _HAS_SACREBLEU:
        try:
            return sacrebleu.sentence_bleu(hyp, [ref]).score
        except Exception:
            pass
    # token-level F1 fall-back (sentence BLEU is unstable without smoothing libs)
    h, r = Counter(hyp.split()), Counter(ref.split())
    overlap = sum(min(c, r[g]) for g, c in h.items())
    hn, rn = sum(h.values()), sum(r.values())
    if not hn or not rn:
        return 0.0
    p, rc = overlap / hn, overlap / rn
    return 0.0 if p + rc == 0 else 100.0 * 2 * p * rc / (p + rc)


def get_utility(name: str) -> Callable[[str, str], float]:
    return {"chrf": chrf, "bleu": bleu}.get(name, chrf)


# --------------------------------------------------------------------------- selection
def mbr_select(
    samples: Sequence[str],
    extra_hyps: Optional[Sequence[str]] = None,
    utility: Callable[[str, str], float] = chrf,
) -> str:
    """Return the MBR-optimal hypothesis.

    ``samples`` is the Monte-Carlo pool (used as pseudo-references; duplicates are
    *kept* so a candidate's frequency increases its consensus weight).
    ``extra_hyps`` are additional candidates allowed to *win* but excluded from
    the reference distribution (e.g. each model's beam-1-best). Ties break toward
    the more frequent (more agreed-upon) candidate, then lexicographically for
    determinism.
    """
    pool = [s for s in samples if s is not None]
    if not pool:
        return next((h for h in (extra_hyps or []) if h is not None), "")

    ref_counts = Counter(pool)            # full distribution, empties included
    total = sum(ref_counts.values())
    candidates = set(ref_counts) | {h for h in (extra_hyps or []) if h is not None}

    # An empty string is never a desirable output: drop it from the *winnable*
    # set whenever any non-empty candidate exists (it still counts as ref mass,
    # so it correctly lowers everyone's consensus score).
    non_empty = {h for h in candidates if h.strip()}
    if non_empty:
        candidates = non_empty

    best_hyp, best_score, best_freq = "", -1.0, -1
    for h in sorted(candidates):  # sorted -> deterministic tie-break
        eu = sum(w * utility(h, r) for r, w in ref_counts.items()) / total
        freq = ref_counts.get(h, 0)
        if (eu, freq) > (best_score, best_freq):
            best_hyp, best_score, best_freq = h, eu, freq
    return best_hyp


def mbr_select_batch(
    pools: Sequence[Sequence[str]],
    extra_per_item: Optional[Sequence[Sequence[str]]] = None,
    utility_name: str = "chrf",
) -> List[str]:
    """Vector-of-pools convenience wrapper around :func:`mbr_select`."""
    util = get_utility(utility_name)
    extra_per_item = extra_per_item or [[] for _ in pools]
    return [
        mbr_select(pool, extra, util)
        for pool, extra in zip(pools, extra_per_item)
    ]
