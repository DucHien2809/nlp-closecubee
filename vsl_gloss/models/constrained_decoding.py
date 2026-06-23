"""Source-constrained decoding -- the project's modelling *improvement*.

Corpus analysis shows the VSL gloss is, in ~98% of pairs, a re-ordered **subset**
of the source: the target almost never introduces a token absent from the input.
We exploit that prior at inference time by masking the vocabulary so the decoder
may only emit (a) sub-word pieces that occur in the source and (b) a small
always-allowed set (eos / pad / punctuation). This is a lightweight,
training-free analogue of a copy mechanism / lexically-constrained decoding
(cf. Hokamp & Liu, 2017; Post & Vilar, 2018) tailored to a monolingual,
subset-style transduction.

Trade-off: it cannot produce the rare genuinely-lexical targets, so we always
report it as an *additional* system alongside unconstrained decoding rather than
a replacement, and quantify the effect in the ablation.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

# Surface punctuation that must always be emittable regardless of the source.
_ALWAYS_ALLOWED_SURFACE = [".", "?", "!", ",", ";", ":", "...", "</s>", "<pad>"]


def build_always_allowed_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Token ids that may always be generated (specials + punctuation pieces)."""
    ids = set()
    for tid in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "decoder_start_token_id", None)):
        if tid is not None:
            ids.add(int(tid))
    for tok in _ALWAYS_ALLOWED_SURFACE:
        for tid in tokenizer.encode(tok, add_special_tokens=False):
            ids.add(int(tid))
    return sorted(ids)


def allowed_ids_for_sources(
    source_input_ids: torch.Tensor,
    always_allowed: Sequence[int],
) -> List[List[int]]:
    """Per-example list of allowed token ids = source pieces ∪ always-allowed."""
    base = set(int(x) for x in always_allowed)
    out: List[List[int]] = []
    for row in source_input_ids:
        out.append(sorted(base | {int(x) for x in row.tolist()}))
    return out


class SourceConstrainedLogitsProcessor(LogitsProcessor):
    """Mask all logits except the per-example allowed token ids.

    ``allowed_token_ids`` is indexed by *batch example*; the processor maps each
    beam row back to its example via ``num_beams``.
    """

    def __init__(self, allowed_token_ids: List[List[int]], num_beams: int):
        self.allowed_token_ids = allowed_token_ids
        self.num_beams = max(1, num_beams)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for row in range(scores.shape[0]):
            example = row // self.num_beams
            allowed = self.allowed_token_ids[example]
            idx = torch.as_tensor(allowed, device=scores.device, dtype=torch.long)
            mask[row, idx] = scores[row, idx]
        return mask


def make_logits_processor(
    tokenizer: PreTrainedTokenizerBase,
    source_input_ids: torch.Tensor,
    num_beams: int,
    extra_allowed: Optional[Iterable[str]] = None,
) -> SourceConstrainedLogitsProcessor:
    """Convenience builder for one generation batch."""
    always = set(build_always_allowed_ids(tokenizer))
    for tok in extra_allowed or []:
        for tid in tokenizer.encode(tok, add_special_tokens=False):
            always.add(int(tid))
    allowed = allowed_ids_for_sources(source_input_ids, sorted(always))
    return SourceConstrainedLogitsProcessor(allowed, num_beams)
