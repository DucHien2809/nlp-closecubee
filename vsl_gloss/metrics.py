"""Evaluation metrics for Vie -> VSL gloss.

Headline metrics:

* **BLEU**  -- corpus BLEU (sacreBLEU; Post, 2018) on whitespace tokens.
* **chrF**  -- character n-gram F-score, robust to the small target vocab.
* **WER**   -- word error rate (the project's primary metric; lower is better).
* **TER**   -- translation edit rate.
* **EM**    -- exact-match sentence accuracy.

Two design choices keep the numbers honest:

1. Every string is whitespace-tokenised exactly as the corpus is, and BLEU is
   computed with ``tokenize="none"`` so sacreBLEU does not re-tokenise Vietnamese
   diacritics in surprising ways.
2. Metrics are reported **overall, per transformation category, and on a
   case-folded variant**. Because ~24% of the test pairs are verbatim copies, an
   overall BLEU/WER alone is misleading; the per-category table is what shows
   whether a system actually learns the re-ordering/deletion transformation.

sacreBLEU / jiwer are used when installed; otherwise correct pure-Python
fall-backs are used so the module runs anywhere (results match on this corpus).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

try:  # primary path
    import sacrebleu  # type: ignore

    _HAS_SACREBLEU = True
except ImportError:  # pragma: no cover
    _HAS_SACREBLEU = False

try:
    import jiwer  # type: ignore

    _HAS_JIWER = True
except ImportError:  # pragma: no cover
    _HAS_JIWER = False


# --------------------------------------------------------------------------- helpers
def _toks(s: str) -> List[str]:
    return s.split()


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences (DP, O(len(a)*len(b)))."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- metrics
def exact_match(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(p.strip() == r.strip() for p, r in zip(preds, refs))
    return 100.0 * hits / len(preds)


def wer(preds: List[str], refs: List[str]) -> float:
    """Corpus WER (%) = total token edits / total reference tokens."""
    if _HAS_JIWER:
        # jiwer aggregates correctly across the corpus.
        return 100.0 * jiwer.wer(refs, preds)
    edits = total = 0
    for p, r in zip(preds, refs):
        rt = _toks(r)
        edits += _edit_distance(_toks(p), rt)
        total += len(rt)
    return 100.0 * edits / max(1, total)


def ter(preds: List[str], refs: List[str]) -> Optional[float]:
    if _HAS_SACREBLEU:
        return sacrebleu.corpus_ter(preds, [refs]).score
    return None


def _bleu_pure(preds: List[str], refs: List[str], max_n: int = 4) -> float:
    """Corpus BLEU-4 with epsilon smoothing; fall-back when sacreBLEU absent."""
    p_num = [0] * max_n
    p_den = [0] * max_n
    pred_len = ref_len = 0
    for p, r in zip(preds, refs):
        pt, rt = _toks(p), _toks(r)
        pred_len += len(pt)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            pred_ng = Counter(tuple(pt[i : i + n]) for i in range(len(pt) - n + 1))
            ref_ng = Counter(tuple(rt[i : i + n]) for i in range(len(rt) - n + 1))
            overlap = sum(min(c, ref_ng[g]) for g, c in pred_ng.items())
            p_num[n - 1] += overlap
            p_den[n - 1] += max(0, len(pt) - n + 1)
    precisions = []
    for num, den in zip(p_num, p_den):
        if den == 0:
            precisions.append(0.0)
        elif num == 0:
            precisions.append(1e-9)  # smoothing to avoid log(0)
        else:
            precisions.append(num / den)
    if min(precisions) <= 0:
        return 0.0
    geo = math.exp(sum(math.log(x) for x in precisions) / max_n)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(1, pred_len))
    return 100.0 * bp * geo


def corpus_bleu(preds: List[str], refs: List[str]) -> float:
    if _HAS_SACREBLEU:
        return sacrebleu.corpus_bleu(preds, [refs], tokenize="none").score
    return _bleu_pure(preds, refs)


def chrf(preds: List[str], refs: List[str]) -> Optional[float]:
    if _HAS_SACREBLEU:
        return sacrebleu.corpus_chrf(preds, [refs]).score
    return None


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """All headline metrics for one (preds, refs) list pair."""
    out = {
        "n": len(preds),
        "bleu": round(corpus_bleu(preds, refs), 3),
        "wer": round(wer(preds, refs), 3),
        "exact_match": round(exact_match(preds, refs), 3),
    }
    c = chrf(preds, refs)
    if c is not None:
        out["chrf"] = round(c, 3)
    t = ter(preds, refs)
    if t is not None:
        out["ter"] = round(t, 3)
    return out


def compute_report(
    preds: List[str],
    refs: List[str],
    categories: Optional[List[str]] = None,
    normalized: bool = True,
) -> Dict:
    """Full evaluation report: overall + per-category (+ case-folded variants)."""
    assert len(preds) == len(refs), "preds/refs length mismatch"
    report: Dict = {"overall": compute_metrics(preds, refs)}

    if normalized:
        lp = [p.lower() for p in preds]
        lr = [r.lower() for r in refs]
        report["overall_lowercased"] = compute_metrics(lp, lr)

    if categories is not None:
        assert len(categories) == len(preds), "categories length mismatch"
        by_cat: Dict[str, Dict[str, float]] = {}
        cats = sorted(set(categories))
        for cat in cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            by_cat[cat] = compute_metrics([preds[i] for i in idx], [refs[i] for i in idx])
        report["by_category"] = by_cat
    return report


def metrics_backend() -> Dict[str, bool]:
    """Which optional backends are active (logged once at eval time)."""
    return {"sacrebleu": _HAS_SACREBLEU, "jiwer": _HAS_JIWER}
