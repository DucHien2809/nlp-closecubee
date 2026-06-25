from pathlib import Path

from vsl_gloss.felix.plus_decode import apply_format, render_edit, verify_prediction_alignment


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
