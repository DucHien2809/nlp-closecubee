"""Test that inference insertion logits are conditioned on predicted order (train/inference parity).

The bug: when model(**feats) is called without order_target/insertion_mask,
_slot_reps returns BOS for every slot (position-agnostic). At training time,
the collator supplies order_target so each slot is set to the encoder rep of
the previously-kept token. The fix passes the predicted order into the model
at inference so insertion logits match the training distribution.
"""
import torch
import pytest

from vsl_gloss.felix.plus_model import FelixPlusModel


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.emb = torch.nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.emb(input_ids)})()


def _make_model(seed=42):
    torch.manual_seed(seed)
    model = FelixPlusModel(
        TinyEncoder(hidden_size=8),
        hidden_size=8,
        pointer_dim=4,
        insertion_vocab_size=5,
        format_vocab_size=2,
    )
    model.eval()
    return model


BASE_FEATS = {
    "input_ids": torch.tensor([[10, 11, 12, 13]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    "first_subword_idx": torch.tensor([[0, 1, 2]]),
    "word_mask": torch.tensor([[1, 1, 1]]),
}


def test_insertion_logits_differ_with_vs_without_order():
    """Insertion logits must differ when order_target is supplied vs not.

    Without the fix, inference calls model(**feats) with no order_target,
    so _slot_reps fills every slot with BOS. With the fix, inference builds
    order_target from the predicted order, so slots reflect actual kept-token
    reps. This test proves those two paths produce different insertion logits,
    confirming parity is restored when the fix is applied.
    """
    model = _make_model()

    # --- path A: no order (mimics the old buggy inference, all-BOS slots) ---
    with torch.no_grad():
        out_no_order = model(**BASE_FEATS)

    insert_logits_no_order = out_no_order.insertion_logits.clone()

    # --- path B: supply a non-trivial order (mimics training / fixed inference) ---
    # 3 src tokens -> keep tokens at positions [2, 0] (reversed order)
    order_target = torch.tensor([[2, 0]])          # shape [1, 2]
    insertion_mask = torch.ones(1, 3, dtype=torch.long)   # 3 slots: before, between, after

    with torch.no_grad():
        out_with_order = model(**BASE_FEATS, order_target=order_target, insertion_mask=insertion_mask)

    insert_logits_with_order = out_with_order.insertion_logits.clone()

    # They MUST differ — if they're equal, the model is ignoring the order
    assert not torch.allclose(insert_logits_no_order[:, :3], insert_logits_with_order[:, :3]), (
        "Insertion logits are identical with and without order_target. "
        "This indicates _slot_reps is not using the supplied order."
    )


def test_slot_reps_bos_fallback_when_no_order():
    """Without order_target, slot 1 onward equals BOS (the fallback behavior we're replacing)."""
    model = _make_model()

    # Manually check _slot_reps with no order — all slots should equal BOS
    word_reps = torch.randn(1, 3, 8)
    word_mask = torch.ones(1, 3, dtype=torch.long)

    with torch.no_grad():
        # dropout=0 by default; bos is a parameter
        slots = model._slot_reps(word_reps, word_mask, order_target=None, insertion_mask=None)

    bos = model.bos.detach()
    # All slots should be BOS (the no-order fallback)
    for s in range(slots.size(1)):
        assert torch.allclose(slots[0, s], bos), f"Slot {s} != BOS in no-order fallback"


def test_slot_reps_uses_order_when_provided():
    """With order_target=[2,0], slot 1 = word_rep[2], slot 2 = word_rep[0]."""
    model = _make_model()

    word_reps = torch.tensor([[[1.0] * 8, [2.0] * 8, [3.0] * 8]])
    word_mask = torch.ones(1, 3, dtype=torch.long)
    order_target = torch.tensor([[2, 0]])
    insertion_mask = torch.ones(1, 3, dtype=torch.long)

    with torch.no_grad():
        slots = model._slot_reps(word_reps, word_mask, order_target, insertion_mask)

    bos = model.bos.detach()
    assert torch.allclose(slots[0, 0], bos), "Slot 0 should be BOS"
    assert torch.allclose(slots[0, 1], word_reps[0, 2]), "Slot 1 should be word_rep[2] (order[0]=2)"
    assert torch.allclose(slots[0, 2], word_reps[0, 0]), "Slot 2 should be word_rep[0] (order[1]=0)"


def test_order_tensors_helper_produces_correct_shapes():
    """_order_tensors helper in plus_train builds tensors of the right shape."""
    from vsl_gloss.felix.plus_train import _order_tensors

    device = torch.device("cpu")

    # Non-empty order
    ot, im = _order_tensors([2, 0, 1], device)
    assert ot.shape == (1, 3), f"Expected (1,3), got {ot.shape}"
    assert im.shape == (1, 4), f"Expected (1,4), got {im.shape}"
    assert ot.tolist() == [[2, 0, 1]]
    assert im.tolist() == [[1, 1, 1, 1]]

    # Empty order (edge case: no kept tokens)
    ot_empty, im_empty = _order_tensors([], device)
    assert ot_empty.shape == (1, 0), f"Expected (1,0), got {ot_empty.shape}"
    assert im_empty.shape == (1, 1), f"Expected (1,1), got {im_empty.shape}"
    assert im_empty.tolist() == [[1]]


def test_model_insertion_logits_differ_conditioned_vs_unconditioned():
    """Model-level check: insertion logits differ when order_target is supplied vs not.

    This test operates directly on the model (not decode_records). It proves that
    conditioning on a predicted order actually changes the insertion logits, i.e., the
    conditioned path is meaningfully distinct from the all-BOS fallback.
    """
    model = _make_model()

    # Verify directly on the model: calling with vs without order changes logits
    feats = BASE_FEATS
    order = [0, 2]  # keep tokens at positions 0 and 2

    order_target, insertion_mask = torch.tensor([order]), torch.ones(1, len(order) + 1, dtype=torch.long)

    with torch.no_grad():
        out_base = model(**feats)
        out_ordered = model(**feats, order_target=order_target, insertion_mask=insertion_mask)

    # The insertion logits must differ (proving the fix has an effect)
    assert not torch.allclose(
        out_base.insertion_logits[:, : len(order) + 1],
        out_ordered.insertion_logits[:, : len(order) + 1],
    ), "Fixed path must produce different insertion logits than all-BOS fallback."


# ---------------------------------------------------------------------------
# Integration test: decode_records uses the CONDITIONED (second) model call
# for insertion logits.
# ---------------------------------------------------------------------------

class _FakeOutput:
    """Stand-in for FelixPlusModel output namedtuple."""

    def __init__(self, tag_logits, pointer_logits, insertion_logits, format_logits):
        self.tag_logits = tag_logits
        self.pointer_logits = pointer_logits
        self.insertion_logits = insertion_logits
        self.format_logits = format_logits


class _CallRecordingModel:
    """Lightweight stand-in for FelixPlusModel that records every forward call.

    Call 1 (unconditioned, order_target=None):
      - tag_logits:       keep all tokens (tag==0 for every word position)
      - insertion_logits: all-NONE (label index 0), the "wrong" result
      - pointer_logits:   identity order [0, 1, 2, ...]
      - format_logits:    label 0

    Call 2 (conditioned, order_target is not None):
      - insertion_logits: label 1 ("HELLO") for slot 0, NONE elsewhere

    This means: if decode_records reads insertion_logits from the FIRST call it
    produces no insertion; if it reads from the SECOND call it inserts "HELLO".
    The test then asserts "HELLO" appears in the output, which can only happen
    when the conditioned (second) call is used.
    """

    def __init__(self, n_words: int, insertion_vocab_size: int, format_vocab_size: int):
        self.n_words = n_words
        self.insertion_vocab_size = insertion_vocab_size
        self.format_vocab_size = format_vocab_size
        self.calls: list = []  # each entry: dict of kwargs received

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, first_subword_idx, word_mask,
                 order_target=None, insertion_mask=None, **kwargs):
        n = self.n_words
        call_kwargs = {
            "order_target": order_target,
            "insertion_mask": insertion_mask,
        }
        self.calls.append(call_kwargs)

        # tag_logits: shape [1, n, 2] — argmax==0 means KEEP for all tokens
        tag_logits = torch.zeros(1, n, 2)
        tag_logits[:, :, 0] = 10.0  # strongly predict KEEP (tag 0)

        # pointer_logits: identity order — token i points to itself
        pointer_logits = torch.eye(n).unsqueeze(0)  # [1, n, n]

        # format_logits: shape [1, 2], predict label 0
        format_logits = torch.zeros(1, self.format_vocab_size)
        format_logits[:, 0] = 10.0

        # insertion_logits: shape [1, n+1, vocab_size]
        insertion_logits = torch.zeros(1, n + 1, self.insertion_vocab_size)

        if order_target is None:
            # Unconditioned call: predict NONE (index 0) for all slots
            insertion_logits[:, :, 0] = 10.0
        else:
            # Conditioned call: predict "HELLO" (index 1) for slot 0
            insertion_logits[:, 0, 1] = 10.0   # slot 0 → label 1 ("HELLO")
            insertion_logits[:, 1:, 0] = 10.0  # other slots → NONE

        return _FakeOutput(tag_logits, pointer_logits, insertion_logits, format_logits)


def _make_fake_tokenizer_and_cfg():
    """Return (tokenizer, cfg) suitable for decode_records."""

    class FakeEncoding(dict):
        def __init__(self, tokens):
            super().__init__({"input_ids": list(range(10, 10 + len(tokens)))})
            self._word_ids = list(range(len(tokens)))

        def word_ids(self):
            return self._word_ids

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, src_tokens, is_split_into_words=True, truncation=True, max_length=128):
            return FakeEncoding(src_tokens[:max_length])

    from vsl_gloss.config import Config

    cfg = Config.from_dict({"felix_plus": {"max_source_length": 128}})
    return FakeTokenizer(), cfg


def test_decode_records_uses_conditioned_second_call():
    """Integration test: decode_records must call the model TWICE and use the
    SECOND (conditioned) call's insertion_logits.

    The stand-in model returns:
      - call 1 (order_target=None)  → insertion label 0 ("NONE") for all slots
      - call 2 (order_target!=None) → insertion label 1 ("HELLO") for slot 0

    Correct decode_records reads slot 0 from call 2 → output contains "HELLO".
    A buggy decode_records that reads from call 1 → output does NOT contain "HELLO".

    The test also asserts:
      (a) exactly 2 model calls were made,
      (b) call 2 received a non-None order_target.
    """
    from vsl_gloss.felix.plus_train import decode_records

    # Vocab: index 0 = NONE, index 1 = "HELLO"
    insertion_vocab = {"NONE": 0, "HELLO": 1}
    format_vocab = {"final=NONE|case=preserve|spacing=1": 0, "final=?|case=preserve|spacing=1": 1}

    # A simple 3-word source sentence; all tokens kept → order = [0, 1, 2]
    records = [{"id": "t1", "vie": "A B C", "vsl": "A B C", "category": "identical"}]

    tokenizer, cfg = _make_fake_tokenizer_and_cfg()
    model = _CallRecordingModel(
        n_words=3,
        insertion_vocab_size=len(insertion_vocab),
        format_vocab_size=len(format_vocab),
    )

    torch.manual_seed(0)
    preds = decode_records(records, tokenizer, model, device=torch.device("cpu"),
                           cfg=cfg, insertion_vocab=insertion_vocab, format_vocab=format_vocab)

    # (a) Model must have been called exactly twice for the one record.
    assert len(model.calls) == 2, (
        f"Expected 2 model calls (unconditioned + conditioned), got {len(model.calls)}. "
        "If only 1 call, decode_records is not making the conditioned second call. "
        "If 0 calls, build_plus_example returned None."
    )

    # (b) Second call must have received a non-None order_target.
    assert model.calls[1]["order_target"] is not None, (
        "Second model call did not receive order_target. "
        "decode_records must pass order_target to the conditioned (second) call."
    )

    # (c) Output must contain "HELLO" — only possible if insertion_logits from call 2
    #     were used (call 1 always returns NONE for all slots).
    assert len(preds) == 1
    assert "HELLO" in preds[0], (
        f"Expected 'HELLO' in prediction (from conditioned logits), got: {preds[0]!r}. "
        "If 'HELLO' is absent, decode_records is reading insertion_logits from the "
        "unconditioned first call instead of the conditioned second call."
    )
