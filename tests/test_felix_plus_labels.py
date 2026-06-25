from vsl_gloss.felix.plus_labels import (
    FIRST_CASE_LOWER,
    FIRST_CASE_PRESERVE,
    FINAL_PUNCT_PERIOD,
    FINAL_PUNCT_QUESTION,
    build_insertion_vocab,
    extract_plus_labels,
)


def test_extract_plus_labels_groups_lexical_phrase_insertions():
    labels = extract_plus_labels(
        "Nhà bạn có mấy người ?".split(),
        "Gia đình bạn người mấy ?".split(),
    )

    assert labels.tags == ["DELETE", "KEEP", "DELETE", "KEEP", "KEEP", "KEEP"]
    assert labels.order == [1, 4, 3, 5]
    assert labels.insertions == [(0, ("Gia", "đình"))]
    assert labels.format.final_punct == FINAL_PUNCT_QUESTION
    assert labels.format.first_case == FIRST_CASE_PRESERVE


def test_extract_plus_labels_supervises_terminal_punctuation_as_insert_and_format():
    labels = extract_plus_labels(
        "Tôi đeo đồng hồ".split(),
        "Tôi đeo đồng hồ .".split(),
    )

    assert labels.tags == ["KEEP", "KEEP", "KEEP", "KEEP"]
    assert labels.order == [0, 1, 2, 3]
    assert labels.insertions == [(4, (".",))]
    assert labels.format.final_punct == FINAL_PUNCT_PERIOD


def test_extract_plus_labels_derives_first_case_from_reconstructed_edit_output():
    labels = extract_plus_labels(
        "x bạn".split(),
        "Gia đình bạn".split(),
    )

    assert labels.tags == ["DELETE", "KEEP"]
    assert labels.order == [1]
    assert labels.insertions == [(0, ("Gia", "đình"))]
    assert labels.format.first_case == FIRST_CASE_PRESERVE


def test_extract_plus_labels_casefolds_alignment_keys():
    labels = extract_plus_labels(
        "Straße".split(),
        "STRASSE".split(),
    )

    assert labels.tags == ["KEEP"]
    assert labels.order == [0]
    assert labels.insertions == []


def test_build_insertion_vocab_keeps_rare_insertions_with_min_count_one():
    records = [
        {"vie": "Tôi đeo đồng hồ", "vsl": "Tôi đeo đồng hồ ."},
        {"vie": "Nhà bạn có mấy người ?", "vsl": "Gia đình bạn người mấy ?"},
    ]

    vocab = build_insertion_vocab(records, min_count=1, max_phrase_len=4)

    assert vocab["NONE"] == 0
    assert vocab["."] > 0
    assert vocab["Gia đình"] > 0


def test_build_insertion_vocab_filters_rare_insertions_with_min_count_two():
    records = [
        {"vie": "Tôi đeo đồng hồ", "vsl": "Tôi đeo đồng hồ ."},
        {"vie": "Bạn đeo kính", "vsl": "Bạn đeo kính ."},
        {"vie": "Nhà bạn có mấy người ?", "vsl": "Gia đình bạn người mấy ?"},
    ]

    vocab = build_insertion_vocab(records, min_count=2, max_phrase_len=4)

    assert vocab == {"NONE": 0, ".": 1}
