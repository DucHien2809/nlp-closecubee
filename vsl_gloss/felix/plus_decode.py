"""FELIX++ decoding helpers and split-fairness verification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.normalize import NormalizeOptions, normalize_text
from ..utils import read_jsonl

_NORM = NormalizeOptions(lowercase=False)


@dataclass(frozen=True)
class EditCandidate:
    text: str
    order: List[int]
    insertions: Dict[int, Tuple[str, ...]]
    format_label: str
    features: Dict[str, float]


def _parse_format(label: str) -> Dict[str, str]:
    out = {}
    for part in label.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def apply_format(text: str, format_label: str) -> str:
    fmt = _parse_format(format_label)
    tokens = normalize_text(text, _NORM).split()

    final = fmt.get("final", "NONE")
    if tokens and tokens[-1] in {".", "?", "!"}:
        tokens = tokens[:-1]
    if final != "NONE":
        tokens.append(final)

    case = fmt.get("case", "preserve")
    if case in {"lower", "upper"}:
        for i, tok in enumerate(tokens):
            if any(ch.isalpha() for ch in tok):
                tokens[i] = tok.lower() if case == "lower" else tok[:1].upper() + tok[1:]
                break

    return normalize_text(" ".join(tokens), _NORM)


def render_edit(
    src_tokens: Sequence[str],
    order: Sequence[int],
    insertions: Dict[int, Tuple[str, ...]],
    format_label: str,
) -> str:
    out: List[str] = []
    for slot in range(len(order) + 1):
        out.extend(insertions.get(slot, ()))
        if slot < len(order):
            src_i = order[slot]
            if 0 <= src_i < len(src_tokens):
                out.append(src_tokens[src_i])
    return apply_format(" ".join(out), format_label)


def verify_prediction_alignment(predictions_path: str | Path, split_path: str | Path) -> None:
    pred_rows = read_jsonl(predictions_path)
    split_rows = read_jsonl(split_path)
    pred_ids = [str(r.get("id")) for r in pred_rows]
    split_ids = [str(r.get("id")) for r in split_rows]
    assert len(pred_rows) == len(split_rows), (
        f"prediction row count {len(pred_rows)} != split row count {len(split_rows)}"
    )
    assert pred_ids == split_ids, "prediction ids do not match split ids in order"
    assert all("pred" in r for r in pred_rows), "every prediction row must include pred"
