import torch

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
