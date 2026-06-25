"""FELIX++ label extraction: tag, pointer, insertion, and format labels.

Insertion labels capture unmatched lexical and punctuation tokens needed for
edit reconstruction. Format labels describe deterministic post-edit repair:
terminal punctuation is a normalization target that should replace/normalize
the final punctuation later, not append a second mark blindly.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

KEEP = "KEEP"
DELETE = "DELETE"

NONE_PHRASE = "NONE"

FINAL_PUNCT_NONE = "NONE"
FINAL_PUNCT_PERIOD = "."
FINAL_PUNCT_QUESTION = "?"
FINAL_PUNCT_EXCLAM = "!"

FIRST_CASE_PRESERVE = "preserve"
FIRST_CASE_LOWER = "lower"
FIRST_CASE_UPPER = "upper"

PUNCT = {".", "?", "!", ",", ";", ":"}


@dataclass(frozen=True)
class FormatLabels:
    """Post-edit normalization labels.

    ``punct_spacing=True`` means the decoder should apply deterministic
    punctuation spacing normalization after edit reconstruction. It is not a
    learned label for raw whitespace in the tokenized input.
    """

    final_punct: str = FINAL_PUNCT_NONE
    first_case: str = FIRST_CASE_PRESERVE
    punct_spacing: bool = True


@dataclass(frozen=True)
class PlusLabels:
    source_tokens: List[str]
    target_tokens: List[str]
    tags: List[str]
    order: List[int]
    insertions: List[Tuple[int, Tuple[str, ...]]] = field(default_factory=list)
    format: FormatLabels = field(default_factory=FormatLabels)


def _first_alpha_token(tokens: Sequence[str]) -> str:
    for tok in tokens:
        if any(ch.isalpha() for ch in tok):
            return tok
    return ""


def _align_key(token: str) -> str:
    return token.casefold()


def _first_case(rendered_tokens: Sequence[str], tgt_tokens: Sequence[str]) -> str:
    src = _first_alpha_token(rendered_tokens)
    tgt = _first_alpha_token(tgt_tokens)
    if not src or not tgt:
        return FIRST_CASE_PRESERVE
    if tgt == src:
        return FIRST_CASE_PRESERVE
    if _align_key(tgt) != _align_key(src):
        return FIRST_CASE_PRESERVE
    if src[:1] != src[:1].lower() and tgt[:1] == tgt[:1].lower():
        return FIRST_CASE_LOWER
    if src[:1] == src[:1].lower() and tgt[:1] == tgt[:1].upper():
        return FIRST_CASE_UPPER
    return FIRST_CASE_PRESERVE


def _final_punct(tokens: Sequence[str]) -> str:
    if not tokens:
        return FINAL_PUNCT_NONE
    last = tokens[-1]
    if last in {FINAL_PUNCT_PERIOD, FINAL_PUNCT_QUESTION, FINAL_PUNCT_EXCLAM}:
        return last
    return FINAL_PUNCT_NONE


def _oracle_reconstruct(
    src_tokens: Sequence[str],
    order: Sequence[int],
    insertions: Sequence[Tuple[int, Tuple[str, ...]]],
) -> List[str]:
    insertions_by_slot: Dict[int, List[Tuple[str, ...]]] = defaultdict(list)
    for slot, phrase in insertions:
        insertions_by_slot[slot].append(phrase)

    rendered: List[str] = []
    for slot, src_i in enumerate(order):
        for phrase in insertions_by_slot.get(slot, []):
            rendered.extend(phrase)
        rendered.append(src_tokens[src_i])
    for phrase in insertions_by_slot.get(len(order), []):
        rendered.extend(phrase)
    return rendered


def extract_plus_labels(src_tokens: List[str], tgt_tokens: List[str]) -> PlusLabels:
    buckets: Dict[str, deque[int]] = defaultdict(deque)
    for i, tok in enumerate(src_tokens):
        buckets[_align_key(tok)].append(i)

    tags = [DELETE] * len(src_tokens)
    order: List[int] = []
    raw_insertions: List[Tuple[int, str]] = []

    for tok in tgt_tokens:
        q = buckets.get(_align_key(tok))
        if q:
            src_i = q.popleft()
            tags[src_i] = KEEP
            order.append(src_i)
        else:
            raw_insertions.append((len(order), tok))

    grouped: List[Tuple[int, Tuple[str, ...]]] = []
    current_slot = None
    current_tokens: List[str] = []
    for slot, tok in raw_insertions:
        if current_slot is None or slot == current_slot:
            current_slot = slot
            current_tokens.append(tok)
        else:
            grouped.append((current_slot, tuple(current_tokens)))
            current_slot = slot
            current_tokens = [tok]
    if current_slot is not None:
        grouped.append((current_slot, tuple(current_tokens)))

    rendered_tokens = _oracle_reconstruct(src_tokens, order, grouped)
    # Insertions can include final punctuation for oracle edit reconstruction;
    # final_punct separately teaches later format repair to replace/normalize
    # the terminal punctuation instead of appending another mark.
    fmt = FormatLabels(
        final_punct=_final_punct(tgt_tokens),
        first_case=_first_case(rendered_tokens, tgt_tokens),
        punct_spacing=True,
    )
    return PlusLabels(src_tokens, tgt_tokens, tags, order, grouped, fmt)


def _phrase_key(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def build_insertion_vocab(
    records: Iterable[Dict],
    min_count: int = 1,
    max_phrase_len: int = 4,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for rec in records:
        labels = extract_plus_labels(rec["vie"].split(), rec["vsl"].split())
        for _, phrase in labels.insertions:
            if 0 < len(phrase) <= max_phrase_len:
                counts[_phrase_key(phrase)] += 1

    vocab = {NONE_PHRASE: 0}
    for phrase, count in sorted(counts.items()):
        if count >= min_count:
            vocab[phrase] = len(vocab)
    return vocab
