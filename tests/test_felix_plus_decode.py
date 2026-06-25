from pathlib import Path

from vsl_gloss.felix.plus_decode import apply_format, render_edit, verify_prediction_alignment, build_candidates_from_predictions, pointer_orders_from_scores


def test_render_edit_inserts_phrase_before_first_kept_token():
    text = render_edit(
        src_tokens="Nhà bạn có mấy người ?".split(),
        order=[1, 4, 3, 5],
        insertions={0: ("Gia", "đình")},
        format_label="final=?|case=preserve|spacing=1",
    )

    assert text == "Gia đình bạn người mấy ?"


def test_apply_format_repairs_final_punctuation_without_forcing_case():
    assert apply_format("tôi không thể nói gì", "final=.|case=preserve|spacing=1") == "tôi không thể nói gì ."
    assert apply_format("Tôi đau đầu .", "final=NONE|case=lower|spacing=1") == "tôi đau đầu"


def test_verify_prediction_alignment_accepts_matching_ids(tmp_path: Path):
    split = tmp_path / "test.jsonl"
    pred = tmp_path / "predictions_test.jsonl"
    split.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    pred.write_text('{"id":"1","pred":"a"}\n{"id":"2","pred":"b"}\n', encoding="utf-8")

    verify_prediction_alignment(pred, split)


def test_build_candidates_from_predictions_produces_non_empty_unique_candidates():
    candidates = build_candidates_from_predictions(
        src_tokens="Con gà ăn thóc .".split(),
        orders=[[0, 1, 3, 2, 4], [0, 1, 2, 3, 4]],
        insertion_options=[{}],
        format_options=["final=.|case=preserve|spacing=1"],
        max_candidates=8,
    )

    texts = [c.text for c in candidates]
    assert "Con gà thóc ăn ." in texts
    assert "Con gà ăn thóc ." in texts
    assert len(texts) == len(set(texts))


def test_pointer_orders_from_scores_returns_reordered_path():
    # rows: BOS, word0, word1, word2, word3, word4
    # cols: word0, word1, word2, word3, word4, EOS
    scores = [
        [9, 0, 0, 0, 0, -9],  # BOS -> word0
        [0, 9, 0, 0, 0, -9],  # word0 -> word1
        [0, 0, 0, 9, 0, -9],  # word1 -> word3
        [0, 0, 0, 0, 9, -9],  # word2 -> word4
        [0, 0, 9, 0, 0, -9],  # word3 -> word2
        [0, 0, 0, 0, 0, 9],   # word4 -> EOS
    ]

    orders = pointer_orders_from_scores(scores, keep_indices=[0, 1, 2, 3, 4], top_k=2)

    assert orders[0] == [0, 1, 3, 2, 4]
