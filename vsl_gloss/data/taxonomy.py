"""Transformation taxonomy: how does a VSL gloss differ from its Vietnamese
source?

Empirically (10k corpus) the Vie -> VSL mapping is dominated by two operations:
function-word **deletion** and constituent **re-ordering** (SVO -> SOV, numeral
after noun, fronted wh-word moved to the end). Only ~1-2% of targets introduce
a token absent from the source. Labelling every pair lets us (a) build a
stratified, leakage-free split and (b) report metrics per category, which is
the only honest way to read the headline numbers given ~24% of pairs are
verbatim copies.

Categories (computed on case-folded whitespace tokens):

    IDENTICAL          source == target
    REORDER_ONLY       same multiset of tokens, different order
    DELETION_ONLY      target is a sub-multiset of source, order preserved
    DELETION_REORDER   target is a sub-multiset of source, order changed
    LEXICAL            target contains a token / count not in the source
                       (substitution or insertion -- the genuinely hard cases)
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import List


class Category(str, Enum):
    IDENTICAL = "identical"
    REORDER_ONLY = "reorder_only"
    DELETION_ONLY = "deletion_only"
    DELETION_REORDER = "deletion_reorder"
    LEXICAL = "lexical"


ALL_CATEGORIES = [c.value for c in Category]


def _is_subsequence(sub: List[str], seq: List[str]) -> bool:
    """True if ``sub`` appears in ``seq`` in order (not necessarily contiguous)."""
    it = iter(seq)
    return all(tok in it for tok in sub)


def _is_sub_multiset(a: Counter, b: Counter) -> bool:
    """True if multiset ``a`` is contained in multiset ``b``."""
    return all(b[t] >= c for t, c in a.items())


def classify(src_tokens: List[str], tgt_tokens: List[str]) -> str:
    """Return the :class:`Category` value for one source/target token pair.

    Comparison is case-insensitive: sentence-initial capitalisation that arises
    purely from re-ordering (e.g. ``mèo`` -> ``Mèo``) is an artefact, not a
    lexical change.
    """
    src = [t.lower() for t in src_tokens]
    tgt = [t.lower() for t in tgt_tokens]

    if src == tgt:
        return Category.IDENTICAL.value

    cs, ct = Counter(src), Counter(tgt)
    if cs == ct:
        return Category.REORDER_ONLY.value

    if _is_sub_multiset(ct, cs):
        # Pure deletion vs deletion + re-ordering.
        return (
            Category.DELETION_ONLY.value
            if _is_subsequence(tgt, src)
            else Category.DELETION_REORDER.value
        )

    return Category.LEXICAL.value
