"""Non-neural baselines that quantify how much of the task is trivial.

* ``copy``       -- emit the source unchanged. With ~24% verbatim pairs this is a
                    deceptively strong baseline and the single most important
                    reference point for reading BLEU/WER.
* ``rule_based`` -- a linguistically-motivated transducer over the *gold* parse
                    (function-word deletion + SVO->SOV + numeral re-ordering).
                    It is upper-bounded by the gold POS, so it measures how far
                    hand-written rules can go on this corpus.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from ..config import Config
from ..utils import get_logger, read_jsonl, write_jsonl

LOG = get_logger("baselines")

PredFn = Callable[[Dict], str]


def write_predictions(name: str, pred_fn: PredFn, cfg: Config, split: str = "test") -> Path:
    """Apply ``pred_fn`` to every record of ``split`` and dump a predictions file.

    The output schema (``id, vie, vsl, category, pred``) is shared by baselines
    and neural models, so ``vsl_gloss.evaluate`` can score any of them uniformly.
    """
    records = read_jsonl(cfg.paths.resolved("splits_dir") / f"{split}.jsonl")
    rows: List[Dict] = []
    for r in records:
        rows.append(
            {
                "id": r["id"],
                "vie": r["vie"],
                "vsl": r["vsl"],
                "category": r["category"],
                "pred": pred_fn(r),
            }
        )
    out = cfg.paths.resolved("output_dir") / name / f"predictions_{split}.jsonl"
    write_jsonl(out, rows)
    LOG.info("[%s] wrote %d predictions -> %s", name, len(rows), out)
    return out
