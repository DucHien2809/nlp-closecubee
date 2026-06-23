"""The copy baseline: prediction = (normalised) source sentence.

This is the reference every other system must beat. Because ~24% of pairs are
identical and most of the rest are deletions/re-orderings of the source, copy
already scores high on BLEU/WER -- which is exactly why reporting it is
non-negotiable.

Run::

    python -m vsl_gloss.baselines.copy_baseline --config configs/default.yaml --split test
"""
from __future__ import annotations

import argparse
from typing import Dict

import yaml

from ..config import Config
from ..utils import apply_overrides
from . import write_predictions


def predict(record: Dict) -> str:
    return record["vie"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy baseline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))
    write_predictions("baseline_copy", predict, cfg, split=args.split)


if __name__ == "__main__":
    main()
