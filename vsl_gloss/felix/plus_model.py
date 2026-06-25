"""FELIX++ model heads for tags, reordering, lexical insertions, and format repair."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import IGNORE


@dataclass
class FelixPlusOutput:
    loss: Optional[torch.Tensor]
    tag_logits: torch.Tensor              # [B, W, 2]
    pointer_logits: torch.Tensor          # [B, W+1, W+1]
    insertion_logits: torch.Tensor        # [B, S, insertion_vocab_size]
    format_logits: torch.Tensor           # [B, format_vocab_size]
    tag_loss: Optional[torch.Tensor] = None
    pointer_loss: Optional[torch.Tensor] = None
    insertion_loss: Optional[torch.Tensor] = None
    format_loss: Optional[torch.Tensor] = None


class FelixPlusModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        pointer_dim: int = 256,
        insertion_vocab_size: int = 1,
        format_vocab_size: int = 1,
        dropout: float = 0.0,
        tag_loss_weight: float = 1.0,
        pointer_loss_weight: float = 1.0,
        insertion_loss_weight: float = 0.35,
        format_loss_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.insertion_vocab_size = insertion_vocab_size
        self.format_vocab_size = format_vocab_size
        self.tag_loss_weight = tag_loss_weight
        self.pointer_loss_weight = pointer_loss_weight
        self.insertion_loss_weight = insertion_loss_weight
        self.format_loss_weight = format_loss_weight
        self.dropout = nn.Dropout(dropout)

        self.tag_head = nn.Linear(hidden_size, 2)
        self.q_proj = nn.Linear(hidden_size, pointer_dim)
        self.k_proj = nn.Linear(hidden_size, pointer_dim)
        self.insertion_head = nn.Linear(hidden_size, insertion_vocab_size)
        self.format_head = nn.Linear(hidden_size, format_vocab_size)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.eos = nn.Parameter(torch.randn(hidden_size) * 0.02)

    # ----------------------------------------------------------------- encoding
    def encode_words(self, input_ids, attention_mask, first_subword_idx, word_mask):
        """Return per-word reps [B, W, H] = encoder hidden state of first sub-word."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        idx = first_subword_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_reps = torch.gather(hidden, 1, idx)
        word_reps = word_reps * word_mask.unsqueeze(-1)
        return self.dropout(word_reps)

    def _pointer_scores(self, word_reps, key_valid):
        """Scaled dot-product successor scores, invalid keys masked to -inf."""
        b, _, _ = word_reps.shape
        bos = self.bos.expand(b, 1, -1)
        eos = self.eos.expand(b, 1, -1)
        queries = self.q_proj(torch.cat([bos, word_reps], dim=1))
        keys = self.k_proj(torch.cat([word_reps, eos], dim=1))
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        return scores.masked_fill(~key_valid.unsqueeze(1), float("-inf"))

    def _mean_pool(self, word_reps, word_mask):
        mask = word_mask.unsqueeze(-1).to(dtype=word_reps.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (word_reps * mask).sum(dim=1) / denom

    def _slot_reps(self, word_reps, word_mask, order_target=None, insertion_mask=None):
        """Build insertion-slot reps from gold target order, using pooled reps for gaps."""
        b, w, h = word_reps.shape
        if insertion_mask is not None:
            n_slots = insertion_mask.size(1)
        elif order_target is not None:
            n_slots = order_target.size(1) + 1
        else:
            n_slots = w + 1

        pooled = self._mean_pool(word_reps, word_mask)
        slots = pooled.unsqueeze(1).expand(b, n_slots, h).clone()
        if order_target is not None and n_slots:
            n_order = min(order_target.size(1), n_slots)
            order = order_target[:, :n_order]
            valid = (order != IGNORE) & (order >= 0) & (order < w)
            gather_idx = order.clamp(min=0, max=max(w - 1, 0)).unsqueeze(-1).expand(-1, -1, h)
            gathered = torch.gather(word_reps, 1, gather_idx)
            slots[:, :n_order] = torch.where(valid.unsqueeze(-1), gathered, slots[:, :n_order])

        if insertion_mask is not None:
            slots = slots * insertion_mask.unsqueeze(-1).to(dtype=slots.dtype)
        return self.dropout(slots)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids,
        attention_mask,
        first_subword_idx,
        word_mask,
        tag_labels=None,
        succ_target=None,
        key_keep_mask=None,
        order_target=None,
        insertion_labels=None,
        insertion_mask=None,
        format_labels=None,
    ) -> FelixPlusOutput:
        word_reps = self.encode_words(input_ids, attention_mask, first_subword_idx, word_mask)
        tag_logits = self.tag_head(word_reps)

        b, w, _ = word_reps.shape
        eos_col = torch.ones(b, 1, dtype=torch.bool, device=word_reps.device)
        word_valid = word_mask.bool() if key_keep_mask is None else key_keep_mask.bool()
        key_valid = torch.cat([word_valid, eos_col], dim=1)
        pointer_logits = self._pointer_scores(word_reps, key_valid)

        slot_reps = self._slot_reps(word_reps, word_mask, order_target, insertion_mask)
        insertion_logits = self.insertion_head(slot_reps)
        format_logits = self.format_head(self._mean_pool(word_reps, word_mask))

        loss = tag_loss = pointer_loss = insertion_loss = format_loss = None
        losses = []

        if tag_labels is not None:
            tag_loss = F.cross_entropy(
                tag_logits.reshape(-1, 2),
                tag_labels.reshape(-1),
                ignore_index=IGNORE,
            )
            losses.append(self.tag_loss_weight * tag_loss)

        if succ_target is not None:
            pointer_loss = F.cross_entropy(
                pointer_logits.reshape(-1, w + 1),
                succ_target.reshape(-1),
                ignore_index=IGNORE,
            )
            losses.append(self.pointer_loss_weight * pointer_loss)

        if insertion_labels is not None:
            insertion_targets = insertion_labels
            if insertion_mask is not None:
                insertion_targets = insertion_targets.masked_fill(
                    insertion_mask.to(dtype=torch.bool).logical_not(), IGNORE
                )
            insertion_loss = F.cross_entropy(
                insertion_logits.reshape(-1, self.insertion_vocab_size),
                insertion_targets.reshape(-1),
                ignore_index=IGNORE,
            )
            losses.append(self.insertion_loss_weight * insertion_loss)

        if format_labels is not None:
            format_loss = F.cross_entropy(format_logits, format_labels)
            losses.append(self.format_loss_weight * format_loss)

        if losses:
            loss = torch.stack(losses).sum()

        return FelixPlusOutput(
            loss=loss,
            tag_logits=tag_logits,
            pointer_logits=pointer_logits,
            insertion_logits=insertion_logits,
            format_logits=format_logits,
            tag_loss=tag_loss,
            pointer_loss=pointer_loss,
            insertion_loss=insertion_loss,
            format_loss=format_loss,
        )
