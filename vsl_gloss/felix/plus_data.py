"""Dataset + collator for FELIX++ tag, pointer, insertion, and format targets."""
from __future__ import annotations

from typing import Dict, List, Optional

from .labels import tokenize
from .model import DELETE_ID, IGNORE, KEEP_ID
from .plus_labels import (
    FINAL_PUNCT_EXCLAM,
    FINAL_PUNCT_NONE,
    FINAL_PUNCT_PERIOD,
    FINAL_PUNCT_QUESTION,
    FIRST_CASE_LOWER,
    FIRST_CASE_PRESERVE,
    FIRST_CASE_UPPER,
    KEEP,
    NONE_PHRASE,
    FormatLabels,
    build_insertion_vocab,
    extract_plus_labels,
)

DEFAULT_FORMAT_KEY = "final=NONE|case=preserve|spacing=1"


def encode_source(tokenizer, src_tokens: List[str], max_length: int):
    """Tokenize pre-split words; return input ids, first sub-word positions, coverage."""
    enc = tokenizer(src_tokens, is_split_into_words=True, truncation=True,
                    max_length=max_length)
    word_ids = enc.word_ids()
    first: Dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in first:
            first[wid] = pos
    n_words = len(src_tokens)
    n_covered = sum(1 for w in range(n_words) if w in first)
    first_idx = [first.get(w, 0) for w in range(n_words)]
    return enc["input_ids"], first_idx, n_covered


def format_key(fmt: FormatLabels) -> str:
    spacing = 1 if fmt.punct_spacing else 0
    return f"final={fmt.final_punct}|case={fmt.first_case}|spacing={spacing}"


def default_format_vocab() -> Dict[str, int]:
    finals = [
        FINAL_PUNCT_NONE,
        FINAL_PUNCT_PERIOD,
        FINAL_PUNCT_QUESTION,
        FINAL_PUNCT_EXCLAM,
    ]
    cases = [
        FIRST_CASE_PRESERVE,
        FIRST_CASE_LOWER,
        FIRST_CASE_UPPER,
    ]
    return {
        format_key(FormatLabels(final_punct=final, first_case=case, punct_spacing=True)): i
        for i, (final, case) in enumerate((final, case) for final in finals for case in cases)
    }


class FelixPlusExample(dict):
    pass


def _phrase_key(tokens) -> str:
    return " ".join(tokens)


def build_plus_example(
    rec: Dict,
    tokenizer,
    insertion_vocab: Dict[str, int],
    format_vocab: Dict[str, int],
    max_length: int,
) -> Optional[FelixPlusExample]:
    if NONE_PHRASE not in insertion_vocab:
        raise ValueError(f"insertion_vocab must contain {NONE_PHRASE!r}")
    if DEFAULT_FORMAT_KEY not in format_vocab:
        raise ValueError(
            f"format_vocab must contain DEFAULT_FORMAT_KEY ({DEFAULT_FORMAT_KEY!r})"
        )

    src = tokenize(rec["vie"])
    if not src:
        return None
    labels = extract_plus_labels(src, tokenize(rec["vsl"]))
    input_ids, first_idx, n_covered = encode_source(tokenizer, src, max_length)
    if n_covered != len(src):
        return None

    tags = [KEEP_ID if tag == KEEP else DELETE_ID for tag in labels.tags]
    none_id = insertion_vocab[NONE_PHRASE]
    insertion_labels = [none_id] * (len(labels.order) + 1)
    unknown_insertions = 0
    for slot, phrase in labels.insertions:
        phrase_id = insertion_vocab.get(_phrase_key(phrase))
        if phrase_id is None:
            unknown_insertions += 1
        elif 0 <= slot < len(insertion_labels):
            insertion_labels[slot] = phrase_id

    fmt_key = format_key(labels.format)
    format_label = format_vocab.get(fmt_key, format_vocab[DEFAULT_FORMAT_KEY])

    return FelixPlusExample(
        id=rec.get("id"),
        vie=rec["vie"],
        vsl=rec["vsl"],
        category=rec.get("category"),
        input_ids=input_ids,
        first_idx=first_idx,
        tags=tags,
        order=labels.order,
        insertion_labels=insertion_labels,
        unknown_insertions=unknown_insertions,
        format_label=format_label,
        src_tokens=src,
    )


class FelixPlusDataset:
    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        insertion_vocab: Dict[str, int],
        format_vocab: Dict[str, int],
        max_length: int,
    ):
        self.examples: List[FelixPlusExample] = []
        self.skipped = 0
        for rec in records:
            ex = build_plus_example(rec, tokenizer, insertion_vocab, format_vocab, max_length)
            if ex is None:
                self.skipped += 1
            else:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class FelixPlusCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: List[FelixPlusExample]):
        import torch

        t_max = max(len(ex["input_ids"]) for ex in batch)
        w_max = max(len(ex["src_tokens"]) for ex in batch)
        order_max = max(len(ex["order"]) for ex in batch)
        ins_max = max(len(ex["insertion_labels"]) for ex in batch)
        eos_idx = w_max
        b = len(batch)

        input_ids = torch.full((b, t_max), self.pad, dtype=torch.long)
        attn = torch.zeros((b, t_max), dtype=torch.long)
        first_idx = torch.zeros((b, w_max), dtype=torch.long)
        word_mask = torch.zeros((b, w_max), dtype=torch.long)
        tag_labels = torch.full((b, w_max), IGNORE, dtype=torch.long)
        key_keep = torch.zeros((b, w_max), dtype=torch.long)
        succ = torch.full((b, w_max + 1), IGNORE, dtype=torch.long)
        order_target = torch.full((b, order_max), IGNORE, dtype=torch.long)
        insertion_labels = torch.full((b, ins_max), IGNORE, dtype=torch.long)
        insertion_mask = torch.zeros((b, ins_max), dtype=torch.long)
        format_labels = torch.zeros((b,), dtype=torch.long)

        for i, ex in enumerate(batch):
            ids = ex["input_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1
            n = len(ex["src_tokens"])
            first_idx[i, :n] = torch.tensor(ex["first_idx"], dtype=torch.long)
            word_mask[i, :n] = 1
            tag_labels[i, :n] = torch.tensor(ex["tags"], dtype=torch.long)
            for j, tag in enumerate(ex["tags"]):
                if tag == KEEP_ID:
                    key_keep[i, j] = 1

            order = ex["order"]
            order_target[i, : len(order)] = torch.tensor(order, dtype=torch.long)
            pos = {src_j: t for t, src_j in enumerate(order)}
            succ[i, 0] = order[0] if order else eos_idx
            for j in range(n):
                if j in pos:
                    t = pos[j]
                    succ[i, 1 + j] = order[t + 1] if t + 1 < len(order) else eos_idx

            ins = ex["insertion_labels"]
            insertion_labels[i, : len(ins)] = torch.tensor(ins, dtype=torch.long)
            insertion_mask[i, : len(ins)] = 1
            format_labels[i] = ex["format_label"]

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "first_subword_idx": first_idx,
            "word_mask": word_mask,
            "tag_labels": tag_labels,
            "succ_target": succ,
            "key_keep_mask": key_keep,
            "order_target": order_target,
            "insertion_labels": insertion_labels,
            "insertion_mask": insertion_mask,
            "format_labels": format_labels,
            "meta": [
                {
                    "id": ex["id"],
                    "vie": ex["vie"],
                    "vsl": ex["vsl"],
                    "category": ex["category"],
                    "src_tokens": ex["src_tokens"],
                }
                for ex in batch
            ],
        }
