from pathlib import Path

import pytest

from vsl_gloss.felix.plus_decode import verify_prediction_alignment


def test_verify_prediction_alignment_rejects_reordered_test_ids(tmp_path: Path):
    split = tmp_path / "test.jsonl"
    pred = tmp_path / "predictions_test.jsonl"
    split.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    pred.write_text('{"id":"2","pred":"b"}\n{"id":"1","pred":"a"}\n', encoding="utf-8")

    with pytest.raises(AssertionError, match="prediction ids do not match"):
        verify_prediction_alignment(pred, split)
