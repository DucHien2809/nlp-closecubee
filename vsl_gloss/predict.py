"""Batched inference for a fine-tuned model, with optional source-constrained
decoding.

Two uses:

* Re-decode a split to produce the *constrained* system for the ablation::

      python -m vsl_gloss.predict --config configs/default.yaml \
          --model outputs/vit5_base_baseline/model --split test \
          --constrained --name vit5_base_constrained

* Ad-hoc translation of a sentence::

      python -m vsl_gloss.predict --model outputs/vit5_base_baseline/model \
          --text "Tôi 19 tuổi ."
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import Config
from .data.normalize import NormalizeOptions, normalize_text
from .utils import apply_overrides, get_logger, read_jsonl, write_jsonl

LOG = get_logger("predict")
_NORM = NormalizeOptions(lowercase=False)


def load_model(model_path: str, cfg: Config):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=cfg.model.use_fast_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def predict_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
    cfg: Config,
    constrained: bool = False,
    batch_size: int = 32,
) -> List[str]:
    import torch
    from transformers import LogitsProcessorList

    from .models.constrained_decoding import make_logits_processor

    prefix = cfg.model.source_prefix
    results: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            [prefix + t for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.model.max_source_length,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=cfg.decode.max_new_tokens,
            num_beams=cfg.decode.num_beams,
            length_penalty=cfg.decode.length_penalty,
            no_repeat_ngram_size=cfg.decode.no_repeat_ngram_size,
            early_stopping=cfg.decode.early_stopping,
        )
        if constrained:
            proc = make_logits_processor(tokenizer, enc["input_ids"], cfg.decode.num_beams)
            gen_kwargs["logits_processor"] = LogitsProcessorList([proc])

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(normalize_text(x, _NORM) for x in decoded)
    return results


def predict_split(cfg: Config, model_path: str, split: str, name: str,
                  constrained: bool, batch_size: int) -> Path:
    records = read_jsonl(cfg.paths.resolved("splits_dir") / f"{split}.jsonl")
    tokenizer, model, device = load_model(model_path, cfg)
    LOG.info("Decoding %s (%d sents, constrained=%s)…", split, len(records), constrained)
    preds = predict_texts([r["vie"] for r in records], tokenizer, model, device, cfg,
                          constrained=constrained, batch_size=batch_size)
    rows: List[Dict] = [
        {"id": r["id"], "vie": r["vie"], "vsl": r["vsl"],
         "category": r["category"], "pred": p}
        for r, p in zip(records, preds)
    ]
    out = cfg.paths.resolved("output_dir") / name / f"predictions_{split}.jsonl"
    write_jsonl(out, rows)
    LOG.info("Wrote %d predictions -> %s", len(rows), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference for a fine-tuned Vie->VSL model")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, help="path to a fine-tuned model dir")
    ap.add_argument("--split", default=None, help="decode a split (test/val/train)")
    ap.add_argument("--text", default=None, help="decode a single sentence")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--name", default=None, help="output system name (dir under outputs/)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    if args.text is not None:
        tokenizer, model, device = load_model(args.model, cfg)
        out = predict_texts([args.text], tokenizer, model, device, cfg, constrained=args.constrained)
        print(out[0])
        return

    if args.split is None:
        ap.error("pass --split SPLIT or --text SENTENCE")
    name = args.name or (Path(args.model).parent.name + ("_constrained" if args.constrained else ""))
    predict_split(cfg, args.model, args.split, name, args.constrained, args.batch_size)


if __name__ == "__main__":
    main()
