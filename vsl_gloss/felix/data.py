"""Dataset + collator turning ``(vie, vsl)`` pairs into FELIX training tensors.

Each example is encoded once into: sub-word ``input_ids``; the ``first_subword``
position of every source word (for gathering word reps); the gold ``tags``
(KEEP/DELETE); and the gold ``order`` (kept word indices in target order). The
collator pads a batch and, knowing the batch's ``W_max``, builds the pointer
``succ_target`` whose EOS key index is ``W_max`` (the column the model appends).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .labels import KEEP, extract_labels, tokenize
from .model import IGNORE

KEEP_ID, DELETE_ID = 0, 1


def encode_source(tokenizer, src_tokens: List[str], max_length: int):
    """Tokenize pre-split words; return (input_ids, first_subword_idx, n_covered).

    ``first_subword_idx[w]`` is the position of word ``w``'s first sub-word in the
    sub-word sequence. ``n_covered`` is how many leading words survived truncation.
    """
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


class FelixExample(dict):
    pass


def build_example(rec: Dict, tokenizer, max_length: int) -> Optional[FelixExample]:
    src = tokenize(rec["vie"])
    if not src:
        return None
    lab = extract_labels(src, tokenize(rec["vsl"]))
    input_ids, first_idx, n_covered = encode_source(tokenizer, src, max_length)
    if n_covered != len(src):
        # A word fell outside the sub-word budget; its tag/pointer would be ill-defined.
        return None
    tags = [KEEP_ID if t == KEEP else DELETE_ID for t in lab.tags]
    return FelixExample(
        id=rec.get("id"), vie=rec["vie"], vsl=rec["vsl"], category=rec.get("category"),
        input_ids=input_ids, first_idx=first_idx, tags=tags, order=lab.order,
        src_tokens=src,
    )


class FelixDataset:
    def __init__(self, records: List[Dict], tokenizer, max_length: int):
        self.examples: List[FelixExample] = []
        self.skipped = 0
        for r in records:
            ex = build_example(r, tokenizer, max_length)
            if ex is None:
                self.skipped += 1
            else:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class FelixCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: List[FelixExample]):
        import torch

        t_max = max(len(ex["input_ids"]) for ex in batch)
        w_max = max(len(ex["src_tokens"]) for ex in batch)
        eos_idx = w_max                                   # EOS key column after padding
        b = len(batch)

        input_ids = torch.full((b, t_max), self.pad, dtype=torch.long)
        attn = torch.zeros((b, t_max), dtype=torch.long)
        first_idx = torch.zeros((b, w_max), dtype=torch.long)
        word_mask = torch.zeros((b, w_max), dtype=torch.long)
        tag_labels = torch.full((b, w_max), IGNORE, dtype=torch.long)
        key_keep = torch.zeros((b, w_max), dtype=torch.long)
        succ = torch.full((b, w_max + 1), IGNORE, dtype=torch.long)

        for i, ex in enumerate(batch):
            ids = ex["input_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
            n = len(ex["src_tokens"])
            first_idx[i, :n] = torch.tensor(ex["first_idx"])
            word_mask[i, :n] = 1
            tag_labels[i, :n] = torch.tensor(ex["tags"])
            for j, t in enumerate(ex["tags"]):
                if t == KEEP_ID:
                    key_keep[i, j] = 1

            order = ex["order"]
            pos = {src_j: t for t, src_j in enumerate(order)}
            succ[i, 0] = order[0] if order else eos_idx          # BOS -> first kept / EOS
            for j in range(n):
                if j in pos:
                    t = pos[j]
                    succ[i, 1 + j] = order[t + 1] if t + 1 < len(order) else eos_idx
                # deleted words stay IGNORE (no outgoing pointer)

        return {
            "input_ids": input_ids, "attention_mask": attn,
            "first_subword_idx": first_idx, "word_mask": word_mask,
            "tag_labels": tag_labels, "succ_target": succ, "key_keep_mask": key_keep,
            "meta": [{"id": ex["id"], "vie": ex["vie"], "vsl": ex["vsl"],
                      "category": ex["category"], "src_tokens": ex["src_tokens"]}
                     for ex in batch],
        }
