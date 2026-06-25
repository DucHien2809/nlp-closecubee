import pytest

from vsl_gloss.felix.model import IGNORE
from vsl_gloss.felix.plus_data import (
    DEFAULT_FORMAT_KEY,
    FelixPlusCollator,
    build_plus_example,
)


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


def test_build_plus_example_contains_insertion_and_format_targets():
    rec = {
        "id": "x1",
        "vie": "Nhà bạn có mấy người ?",
        "vsl": "Gia đình bạn người mấy ?",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0, "Gia đình": 1}
    format_vocab = {
        "final=NONE|case=preserve|spacing=1": 0,
        "final=?|case=preserve|spacing=1": 1,
    }

    ex = build_plus_example(rec, FakeTokenizer(), insertion_vocab, format_vocab, max_length=128)

    assert ex is not None
    assert ex["tags"] == [1, 0, 1, 0, 0, 0]
    assert ex["order"] == [1, 4, 3, 5]
    assert ex["insertion_labels"][0] == 1
    assert ex["format_label"] == 1


def test_collator_pads_plus_targets():
    rec = {
        "id": "x2",
        "vie": "Tôi đeo đồng hồ",
        "vsl": "Tôi đeo đồng hồ .",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0, ".": 1}
    format_vocab = {
        "final=NONE|case=preserve|spacing=1": 0,
        "final=.|case=preserve|spacing=1": 1,
    }
    ex = build_plus_example(rec, FakeTokenizer(), insertion_vocab, format_vocab, max_length=128)

    batch = FelixPlusCollator(FakeTokenizer().pad_token_id)([ex])

    assert batch["input_ids"].shape[0] == 1
    assert batch["insertion_labels"].shape[1] == len(rec["vie"].split()) + 1
    assert int(batch["format_labels"][0]) == 1


def test_collator_ignores_padded_insertion_slots_and_uses_batch_eos():
    short_rec = {
        "id": "short",
        "vie": "Tôi đi",
        "vsl": "Tôi đi",
        "category": "lexical",
    }
    long_rec = {
        "id": "long",
        "vie": "Tôi đeo đồng hồ",
        "vsl": "Tôi đeo đồng hồ .",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0, ".": 1}
    format_vocab = {
        DEFAULT_FORMAT_KEY: 0,
        "final=.|case=preserve|spacing=1": 1,
    }
    tokenizer = FakeTokenizer()
    short_ex = build_plus_example(
        short_rec, tokenizer, insertion_vocab, format_vocab, max_length=128
    )
    long_ex = build_plus_example(
        long_rec, tokenizer, insertion_vocab, format_vocab, max_length=128
    )

    batch = FelixPlusCollator(tokenizer.pad_token_id)([short_ex, long_ex])

    assert batch["insertion_labels"].shape[1] == 5
    assert batch["insertion_labels"][0, 3:].tolist() == [IGNORE, IGNORE]
    assert batch["insertion_mask"][0, 3:].tolist() == [0, 0]
    assert int(batch["succ_target"][0, 2]) == 4


def test_build_plus_example_requires_vocab_defaults():
    rec = {
        "id": "x3",
        "vie": "Tôi đi",
        "vsl": "Tôi đi",
        "category": "lexical",
    }

    with pytest.raises(ValueError, match="NONE"):
        build_plus_example(rec, FakeTokenizer(), {}, {DEFAULT_FORMAT_KEY: 0}, max_length=128)

    with pytest.raises(ValueError, match="DEFAULT_FORMAT_KEY"):
        build_plus_example(rec, FakeTokenizer(), {"NONE": 0}, {}, max_length=128)


def test_build_plus_example_tracks_unknown_insertions_as_none():
    rec = {
        "id": "x4",
        "vie": "bạn mấy ?",
        "vsl": "Gia đình bạn mấy ?",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0}
    format_vocab = {
        DEFAULT_FORMAT_KEY: 0,
        "final=?|case=preserve|spacing=1": 1,
    }

    ex = build_plus_example(rec, FakeTokenizer(), insertion_vocab, format_vocab, max_length=128)

    assert ex["insertion_labels"][0] == 0
    assert ex["unknown_insertions"] == 1
