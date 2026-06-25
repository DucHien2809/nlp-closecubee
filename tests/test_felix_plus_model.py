import torch

from vsl_gloss.felix.model import IGNORE
from vsl_gloss.felix.plus_model import FelixPlusModel


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.emb = torch.nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.emb(input_ids)})()


def test_felix_plus_model_returns_all_losses_and_logits():
    model = FelixPlusModel(
        TinyEncoder(),
        hidden_size=8,
        pointer_dim=4,
        insertion_vocab_size=3,
        format_vocab_size=2,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "first_subword_idx": torch.tensor([[0, 1]]),
        "word_mask": torch.tensor([[1, 1]]),
        "tag_labels": torch.tensor([[0, 1]]),
        "succ_target": torch.tensor([[0, 2, -100]]),
        "key_keep_mask": torch.tensor([[1, 0]]),
        "order_target": torch.tensor([[0, -100]]),
        "insertion_labels": torch.tensor([[0, 1]]),
        "insertion_mask": torch.tensor([[1, 1]]),
        "format_labels": torch.tensor([1]),
    }

    out = model(**batch)

    assert out.loss is not None
    assert out.tag_logits.shape == (1, 2, 2)
    assert out.pointer_logits.shape == (1, 3, 3)
    assert out.insertion_logits.shape[:2] == (1, 2)
    assert out.format_logits.shape == (1, 2)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.tag_loss)
    assert torch.isfinite(out.pointer_loss)
    assert torch.isfinite(out.insertion_loss)
    assert torch.isfinite(out.format_loss)

    out.loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(torch.any(grad != 0) for grad in grads)


def test_slot_reps_align_with_insertion_slots():
    model = FelixPlusModel(
        TinyEncoder(hidden_size=2),
        hidden_size=2,
        pointer_dim=2,
        insertion_vocab_size=3,
        format_vocab_size=2,
    )
    with torch.no_grad():
        model.bos.copy_(torch.tensor([1.0, 2.0]))

    word_reps = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
    word_mask = torch.tensor([[1, 1, 1]])
    order_target = torch.tensor([[2, 0, 1]])
    insertion_mask = torch.tensor([[1, 1, 1, 1]])

    slots = model._slot_reps(word_reps, word_mask, order_target, insertion_mask)

    assert torch.allclose(slots[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(slots[0, 1], word_reps[0, 2])
    assert torch.allclose(slots[0, 2], word_reps[0, 0])
    assert torch.allclose(slots[0, 3], word_reps[0, 1])


def test_felix_plus_model_ignores_padded_insertion_labels():
    model = FelixPlusModel(
        TinyEncoder(),
        hidden_size=8,
        pointer_dim=4,
        insertion_vocab_size=3,
        format_vocab_size=2,
    )
    common = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "first_subword_idx": torch.tensor([[0, 1]]),
        "word_mask": torch.tensor([[1, 1]]),
        "tag_labels": torch.tensor([[0, 1]]),
        "succ_target": torch.tensor([[0, 2, -100]]),
        "key_keep_mask": torch.tensor([[1, 0]]),
        "order_target": torch.tensor([[0, -100]]),
        "insertion_mask": torch.tensor([[1, 0]]),
        "format_labels": torch.tensor([1]),
    }
    out_ignore = model(**common, insertion_labels=torch.tensor([[0, -100]]))
    out_other = model(**common, insertion_labels=torch.tensor([[0, 2]]))

    assert torch.allclose(out_ignore.insertion_loss, out_other.insertion_loss)


def test_felix_plus_model_zeroes_all_ignored_insertion_loss():
    model = FelixPlusModel(
        TinyEncoder(),
        hidden_size=8,
        pointer_dim=4,
        insertion_vocab_size=3,
        format_vocab_size=2,
    )

    out = model(
        input_ids=torch.tensor([[10, 11, 12]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        first_subword_idx=torch.tensor([[0, 1]]),
        word_mask=torch.tensor([[1, 1]]),
        order_target=torch.tensor([[0]]),
        insertion_labels=torch.tensor([[IGNORE, 2]]),
        insertion_mask=torch.tensor([[0, 0]]),
    )

    assert out.insertion_loss is not None
    assert torch.isfinite(out.insertion_loss)
    assert out.insertion_loss.item() == 0.0
    assert torch.isfinite(out.loss)
    assert out.loss.item() == 0.0
