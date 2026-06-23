"""Decode a split with a trained FELIX model into a standard predictions file.

The output schema is identical to every other system (``{id, vie, vsl, category,
pred}``) so it lands on the same ``vsl_gloss.evaluate`` leaderboard as the copy /
rule baselines and the ViT5 seq2seq model.

Examples whose source is too long for the sub-word budget are decoded by a copy
fallback (emit the source) so every record still produces a row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..config import Config
from ..data.normalize import NormalizeOptions, normalize_text
from ..utils import apply_overrides, get_logger, read_jsonl, write_jsonl
from .data import build_example
from .model import FelixModel, build_encoder, render

LOG = get_logger("felix.predict")
_NORM = NormalizeOptions(lowercase=False)


def truecase(text: str) -> str:
    """Capitalise the first alphabetic character (gloss sentences start upper-cased)."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def load_felix(model_dir: str, felix_cfg):
    """Rebuild encoder + heads from a saved FELIX directory."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "encoder"), use_fast=True)
    encoder = AutoModel.from_pretrained(str(model_dir / "encoder"))
    hidden = encoder.config.hidden_size
    model = FelixModel(encoder, hidden, pointer_dim=felix_cfg.pointer_dim,
                       dropout=felix_cfg.dropout,
                       pointer_loss_weight=felix_cfg.pointer_loss_weight)
    state = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def decode_records(records: List[Dict], tokenizer, model, device, cfg: Config) -> List[str]:
    """Return one gloss string per record (in input order)."""
    import torch

    from .data import FelixCollator

    fcfg = cfg.felix
    do_truecase = fcfg.truecase
    # Build examples, remembering which records fell back to copy (too long / empty).
    built: List[Optional[Dict]] = []
    for r in records:
        built.append(build_example(r, tokenizer, fcfg.max_source_length))

    collator = FelixCollator(tokenizer.pad_token_id)
    preds: List[Optional[str]] = [None] * len(records)

    order_idx = [i for i, ex in enumerate(built) if ex is not None]
    bs = fcfg.eval_batch_size
    for start in range(0, len(order_idx), bs):
        chunk = order_idx[start : start + bs]
        batch = collator([built[i] for i in chunk])
        feats = {k: batch[k].to(device) for k in
                 ("input_ids", "attention_mask", "first_subword_idx", "word_mask")}
        orders = model.decode(**feats)
        for local, gi in enumerate(chunk):
            src = built[gi]["src_tokens"]
            text = normalize_text(render(src, orders[local]), _NORM)
            preds[gi] = truecase(text) if do_truecase else text

    # Copy fallback for skipped records.
    for i, p in enumerate(preds):
        if p is None:
            preds[i] = normalize_text(records[i]["vie"], _NORM)
    return preds  # type: ignore[return-value]


def predict_split(cfg: Config, model_dir: str, split: str, name: str) -> Path:
    records = read_jsonl(cfg.paths.resolved("splits_dir") / f"{split}.jsonl")
    tokenizer, model, device = load_felix(model_dir, cfg.felix)
    LOG.info("FELIX decoding %s (%d sents)…", split, len(records))
    preds = decode_records(records, tokenizer, model, device, cfg)
    rows = [{"id": r["id"], "vie": r["vie"], "vsl": r["vsl"],
             "category": r["category"], "pred": p}
            for r, p in zip(records, preds)]
    out = cfg.paths.resolved("output_dir") / name / f"predictions_{split}.jsonl"
    write_jsonl(out, rows)
    LOG.info("Wrote %d FELIX predictions -> %s", len(rows), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode a split with a trained FELIX model")
    ap.add_argument("--config", default="configs/felix.yaml")
    ap.add_argument("--model", required=True, help="path to a trained FELIX dir (has encoder/, heads.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text", default=None, help="decode a single sentence instead")
    ap.add_argument("--name", default="felix")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    if args.text is not None:
        tokenizer, model, device = load_felix(args.model, cfg.felix)
        rec = {"id": 0, "vie": args.text, "vsl": "", "category": None}
        print(decode_records([rec], tokenizer, model, device, cfg)[0])
        return
    predict_split(cfg, args.model, args.split, args.name)


if __name__ == "__main__":
    main()
