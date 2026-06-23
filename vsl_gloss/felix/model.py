"""The FELIX edit model: a shared encoder with a tagging head and a pointer head.

Layout for the pointer (reordering) network, per example with ``W`` source words:

* **queries** = ``[BOS, w_0, …, w_{W-1}]``  (``W+1`` rows) -- "what comes after me?"
* **keys**    = ``[w_0, …, w_{W-1}, EOS]``  (``W+1`` cols) -- candidate successors.

The pointer is a single scaled dot-product attention between projected query and
key word-representations (plus learned ``BOS``/``EOS`` vectors). Training teacher-
forces the gold tags to mask deleted words out of the key set; decoding masks by
the *predicted* tags and follows successor links greedily from ``BOS`` to ``EOS``.

Word representations are the encoder hidden state of each word's **first sub-word**
(the standard token-classification trick), gathered via ``first_subword_idx``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

KEEP_ID, DELETE_ID = 0, 1
IGNORE = -100


@dataclass
class FelixOutput:
    loss: Optional[torch.Tensor]
    tag_logits: torch.Tensor        # [B, W, 2]
    pointer_logits: torch.Tensor    # [B, W+1, W+1]  (queries x keys)
    tag_loss: Optional[torch.Tensor] = None
    pointer_loss: Optional[torch.Tensor] = None


class FelixModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, pointer_dim: int = 256,
                 dropout: float = 0.1, pointer_loss_weight: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.pointer_loss_weight = pointer_loss_weight
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden_size, 2)
        self.q_proj = nn.Linear(hidden_size, pointer_dim)
        self.k_proj = nn.Linear(hidden_size, pointer_dim)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        # Learned virtual nodes: BOS is a query (the sentence start), EOS a key (stop).
        self.bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.eos = nn.Parameter(torch.randn(hidden_size) * 0.02)

    # ----------------------------------------------------------------- encoding
    def encode_words(self, input_ids, attention_mask, first_subword_idx, word_mask):
        """Return per-word reps [B, W, H] = encoder hidden state of first sub-word."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                      # [B, T, H]
        idx = first_subword_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_reps = torch.gather(hidden, 1, idx)            # [B, W, H]
        word_reps = word_reps * word_mask.unsqueeze(-1)     # zero padded words
        return self.dropout(word_reps)

    def _pointer_scores(self, word_reps, key_valid):
        """Scaled dot-product successor scores, invalid keys masked to -inf."""
        b, w, _ = word_reps.shape
        bos = self.bos.expand(b, 1, -1)
        eos = self.eos.expand(b, 1, -1)
        queries = self.q_proj(torch.cat([bos, word_reps], dim=1))   # [B, W+1, P]
        keys = self.k_proj(torch.cat([word_reps, eos], dim=1))      # [B, W+1, P]
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale  # [B, W+1, W+1]
        scores = scores.masked_fill(~key_valid.unsqueeze(1), float("-inf"))
        return scores

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask, first_subword_idx, word_mask,
                tag_labels=None, succ_target=None, key_keep_mask=None) -> FelixOutput:
        word_reps = self.encode_words(input_ids, attention_mask, first_subword_idx, word_mask)
        tag_logits = self.tag_head(word_reps)               # [B, W, 2]

        # Key validity for the pointer: a real (non-pad) word, EOS always valid.
        # During training ``key_keep_mask`` (gold KEEP) additionally hides deleted
        # words so the model only ranks plausible successors.
        b, w, _ = word_reps.shape
        eos_col = torch.ones(b, 1, dtype=torch.bool, device=word_reps.device)
        word_valid = word_mask.bool() if key_keep_mask is None else key_keep_mask.bool()
        key_valid = torch.cat([word_valid, eos_col], dim=1)         # [B, W+1]
        pointer_logits = self._pointer_scores(word_reps, key_valid)

        loss = tag_loss = pointer_loss = None
        if tag_labels is not None and succ_target is not None:
            tag_loss = F.cross_entropy(tag_logits.reshape(-1, 2),
                                       tag_labels.reshape(-1), ignore_index=IGNORE)
            pointer_loss = F.cross_entropy(pointer_logits.reshape(-1, w + 1),
                                           succ_target.reshape(-1), ignore_index=IGNORE)
            loss = tag_loss + self.pointer_loss_weight * pointer_loss
        return FelixOutput(loss, tag_logits, pointer_logits, tag_loss, pointer_loss)

    # ------------------------------------------------------------------- decode
    @torch.no_grad()
    def decode(self, input_ids, attention_mask, first_subword_idx, word_mask) -> List[List[int]]:
        """Greedy tag -> pointer-chase. Returns, per example, the kept word indices
        in predicted target order."""
        word_reps = self.encode_words(input_ids, attention_mask, first_subword_idx, word_mask)
        tag_logits = self.tag_head(word_reps)
        keep = (tag_logits.argmax(-1) == KEEP_ID) & word_mask.bool()   # [B, W]

        b, w, _ = word_reps.shape
        eos_col = torch.ones(b, 1, dtype=torch.bool, device=word_reps.device)
        key_valid = torch.cat([keep, eos_col], dim=1)
        scores = self._pointer_scores(word_reps, key_valid)            # [B, W+1, W+1]

        orders: List[List[int]] = []
        for i in range(b):
            kept = keep[i]
            n_keep = int(kept.sum().item())
            used = torch.zeros(w + 1, dtype=torch.bool, device=scores.device)
            order: List[int] = []
            cur = 0                                                    # 0 == BOS query row
            for _ in range(n_keep + 1):
                row = scores[i, cur].clone()
                row[used] = float("-inf")
                nxt = int(row.argmax().item())
                if nxt == w or not torch.isfinite(row[nxt]):           # EOS or dead-end
                    break
                order.append(nxt)
                used[nxt] = True
                cur = nxt + 1                                          # word j -> query row j+1
            orders.append(order)
        return orders


# ------------------------------------------------------------------- builder
def build_encoder(encoder_name: str):
    """Load a HF encoder; returns (encoder_module, hidden_size)."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(encoder_name)
    hidden = encoder.config.hidden_size
    return encoder, hidden


def render(source_tokens: List[str], order: List[int]) -> str:
    """Edit program -> gloss string (emit kept source surfaces in predicted order)."""
    return " ".join(source_tokens[i] for i in order if 0 <= i < len(source_tokens))
