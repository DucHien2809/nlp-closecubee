"""Train the FELIX edit model (tagging + pointer reordering).

A plain PyTorch loop (the pointer loss + decode-based metrics don't fit the
seq2seq ``Trainer`` cleanly) that mirrors the project's seq2seq philosophy:
**evaluate by actually decoding** the validation set every epoch and keep the
checkpoint with the best BLEU. After training, the held-out test set is decoded
into a standard predictions file and scored, so FELIX lands on the leaderboard
next to the baselines and ViT5.

Run::

    python -m vsl_gloss.felix.train --config configs/felix.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import yaml

from ..config import Config
from ..metrics import compute_metrics, metrics_backend
from ..utils import apply_overrides, get_logger, read_jsonl, set_seed
from .data import FelixCollator, FelixDataset
from .model import FelixModel, build_encoder
from .predict import decode_records, predict_split

LOG = get_logger("felix.train")


def save_felix(model: FelixModel, tokenizer, out_dir: Path) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out_dir / "encoder"))
    tokenizer.save_pretrained(str(out_dir / "encoder"))
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(heads, out_dir / "heads.pt")


def _evaluate(model, tokenizer, device, cfg, val_records) -> Dict:
    model.eval()
    preds = decode_records(val_records, tokenizer, model, device, cfg)
    return compute_metrics(preds, [r["vsl"] for r in val_records])


def run(cfg: Config) -> Dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    fcfg = cfg.felix
    set_seed(fcfg.seed)
    LOG.info("FELIX encoder: %s | metric backends: %s", fcfg.encoder_name, metrics_backend())

    tokenizer = AutoTokenizer.from_pretrained(fcfg.encoder_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(f"{fcfg.encoder_name} lacks a fast tokenizer (needed for word_ids).")
    encoder, hidden = build_encoder(fcfg.encoder_name)
    model = FelixModel(encoder, hidden, pointer_dim=fcfg.pointer_dim,
                       dropout=fcfg.dropout, pointer_loss_weight=fcfg.pointer_loss_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    splits = cfg.paths.resolved("splits_dir")
    train_recs = read_jsonl(splits / "train.jsonl")
    val_recs = read_jsonl(splits / "val.jsonl")
    train_ds = FelixDataset(train_recs, tokenizer, fcfg.max_source_length)
    LOG.info("Train examples: %d (skipped %d too-long)", len(train_ds), train_ds.skipped)

    collator = FelixCollator(tokenizer.pad_token_id)
    loader = DataLoader(train_ds, batch_size=fcfg.batch_size, shuffle=True, collate_fn=collator)

    # New heads learn faster than the pretrained encoder -> a higher LR group.
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optim = torch.optim.AdamW(
        [{"params": enc_params, "lr": fcfg.learning_rate},
         {"params": head_params, "lr": fcfg.learning_rate * 20}],
        weight_decay=fcfg.weight_decay,
    )
    total_steps = max(1, len(loader)) * int(round(fcfg.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(
        optim, int(total_steps * fcfg.warmup_ratio), total_steps)

    out_dir = cfg.paths.resolved("output_dir") / cfg.experiment_name
    best_bleu, best_metrics = -1.0, {}
    n_epochs = int(round(fcfg.num_train_epochs))
    for epoch in range(1, n_epochs + 1):
        model.train()
        running = tag_run = ptr_run = 0.0
        for step, batch in enumerate(loader, 1):
            feats = {k: batch[k].to(device) for k in
                     ("input_ids", "attention_mask", "first_subword_idx", "word_mask",
                      "tag_labels", "succ_target", "key_keep_mask")}
            out = model(**feats)
            optim.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg.max_grad_norm)
            optim.step()
            scheduler.step()
            running += out.loss.item()
            tag_run += out.tag_loss.item()
            ptr_run += out.pointer_loss.item()
            if step % 50 == 0:
                LOG.info("epoch %d step %d/%d  loss=%.4f (tag=%.4f ptr=%.4f)",
                         epoch, step, len(loader), running / step, tag_run / step, ptr_run / step)

        m = _evaluate(model, tokenizer, device, cfg, val_recs)
        LOG.info("epoch %d VAL  bleu=%.2f wer=%.2f em=%.2f",
                 epoch, m.get("bleu", 0), m.get("wer", 0), m.get("exact_match", 0))
        if m.get("bleu", 0) > best_bleu:
            best_bleu, best_metrics = m["bleu"], m
            save_felix(model, tokenizer, out_dir / "model")
            LOG.info("  ↑ new best (bleu=%.2f) saved -> %s", best_bleu, out_dir / "model")

    cfg.save(out_dir / "config_used.yaml")
    LOG.info("Best VAL: %s", best_metrics)

    # -- decode held-out test with the best checkpoint + score --------------------
    pred_path = predict_split(cfg, str(out_dir / "model"), split="test", name=cfg.experiment_name)
    from ..evaluate import score_file

    report = score_file(pred_path, cfg,
                        report_categories=cfg.eval.report_categories,
                        report_normalized=cfg.eval.report_normalized,
                        error_examples=cfg.eval.error_examples_per_category)
    LOG.info("TEST overall: %s", report["overall"])
    return report


def _load_cfg(config_path: str, overrides: List[str]) -> Config:
    with open(config_path, "r", encoding="utf-8") as fh:
        return Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, overrides))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the FELIX edit model for Vie->VSL gloss")
    ap.add_argument("--config", default="configs/felix.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run(_load_cfg(args.config, args.set))


if __name__ == "__main__":
    main()
