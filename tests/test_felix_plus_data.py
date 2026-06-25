from vsl_gloss.felix.plus_data import FelixPlusCollator, build_plus_example


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
