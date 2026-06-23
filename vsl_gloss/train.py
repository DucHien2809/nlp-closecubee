"""Fine-tune a seq2seq backbone (ViT5 / BARTpho) for Vie -> VSL gloss.

Design notes / fixes over the original course script:

* **Generate-based validation.** ``predict_with_generate=True`` so BLEU / WER /
  exact-match are computed from real beam-search decodes every epoch (the
  original only tracked eval-loss), and the best checkpoint is selected on BLEU.
* **Consistent task prefix.** A single ``model.source_prefix`` is used for
  training *and* inference (the original trained with ``"sign-grammar: "`` but
  decoded with ``"vietnews: "``).
* **Leakage-free splits + label smoothing + cosine schedule + early stopping.**
* **Unified prediction dump.** After training, the held-out *test* set is decoded
  and written as a standard predictions file, then scored by ``vsl_gloss.evaluate``
  so the model lands on the same leaderboard as the baselines.

Run (locally or inside the Modal GPU container)::

    python -m vsl_gloss.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .config import Config
from .data.normalize import NormalizeOptions, normalize_text
from .metrics import compute_metrics as score_metrics
from .metrics import metrics_backend
from .utils import apply_overrides, get_logger, read_jsonl, set_seed, write_jsonl

LOG = get_logger("train")
_NORM = NormalizeOptions(lowercase=False)


def _build_training_args(cfg: Config, out_dir: Path):
    from transformers import Seq2SeqTrainingArguments

    kwargs = dict(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        label_smoothing_factor=cfg.train.label_smoothing_factor,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        optim=cfg.train.optim,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        save_total_limit=cfg.train.save_total_limit,
        save_safetensors=cfg.train.save_safetensors,
        logging_steps=cfg.train.logging_steps,
        logging_dir=str(out_dir / "logs"),
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        predict_with_generate=cfg.train.predict_with_generate,
        generation_max_length=cfg.model.max_target_length,
        generation_num_beams=cfg.decode.num_beams,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        seed=cfg.train.seed,
        report_to=[],
        save_strategy=cfg.train.save_strategy,
    )
    sig = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    # `eval_strategy` was renamed from `evaluation_strategy` around transformers 4.41.
    kwargs["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = (
        cfg.train.eval_strategy
    )
    kwargs = {k: v for k, v in kwargs.items() if k in sig}
    return Seq2SeqTrainingArguments(**kwargs)


def _load_datasets(cfg: Config):
    from datasets import load_dataset

    splits = cfg.paths.resolved("splits_dir")
    data_files = {
        "train": str(splits / "train.jsonl"),
        "validation": str(splits / "val.jsonl"),
        "test": str(splits / "test.jsonl"),
    }
    return load_dataset("json", data_files=data_files)


def run(cfg: Config) -> Dict:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
    )

    set_seed(cfg.train.seed)
    LOG.info("Backbone: %s | metric backends: %s", cfg.model.name_or_path, metrics_backend())

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path, use_fast=cfg.model.use_fast_tokenizer
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)
    if cfg.train.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    raw = _load_datasets(cfg)
    prefix = cfg.model.source_prefix

    def preprocess(batch):
        inputs = [prefix + s for s in batch["vie"]]
        model_inputs = tokenizer(
            inputs, max_length=cfg.model.max_source_length, truncation=True
        )
        labels = tokenizer(
            text_target=batch["vsl"], max_length=cfg.model.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keep_cols = raw["train"].column_names
    tokenized = raw.map(preprocess, batched=True, remove_columns=keep_cols, desc="tokenize")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = [normalize_text(x, _NORM) for x in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        dl = [normalize_text(x, _NORM) for x in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        m = score_metrics(dp, dl)
        return {k: m[k] for k in ("bleu", "wer", "exact_match") if k in m}

    out_dir = cfg.paths.resolved("output_dir") / cfg.experiment_name
    args = _build_training_args(cfg, out_dir)

    callbacks = []
    if cfg.train.load_best_model_at_end and cfg.train.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(cfg.train.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    LOG.info("Starting training (%s epochs)…", cfg.train.num_train_epochs)
    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    cfg.save(out_dir / "config_used.yaml")

    # -- decode the held-out test set into a standard predictions file ---------
    LOG.info("Decoding held-out test set…")
    pred_out = trainer.predict(
        tokenized["test"],
        max_length=cfg.model.max_target_length,
        num_beams=cfg.decode.num_beams,
    )
    preds = np.where(pred_out.predictions != -100, pred_out.predictions, tokenizer.pad_token_id)
    decoded = [normalize_text(x, _NORM) for x in tokenizer.batch_decode(preds, skip_special_tokens=True)]

    test_records = read_jsonl(cfg.paths.resolved("splits_dir") / "test.jsonl")
    assert len(test_records) == len(decoded), "test order/length mismatch"
    rows: List[Dict] = [
        {"id": r["id"], "vie": r["vie"], "vsl": r["vsl"],
         "category": r["category"], "pred": p}
        for r, p in zip(test_records, decoded)
    ]
    pred_path = out_dir / "predictions_test.jsonl"
    write_jsonl(pred_path, rows)
    LOG.info("Wrote test predictions -> %s", pred_path)

    # -- score immediately so a report is always produced ----------------------
    from .evaluate import score_file

    report = score_file(
        pred_path, cfg,
        report_categories=cfg.eval.report_categories,
        report_normalized=cfg.eval.report_normalized,
        error_examples=cfg.eval.error_examples_per_category,
    )
    LOG.info("TEST overall: %s", report["overall"])
    return report


def _load_cfg(config_path: str, overrides: List[str]) -> Config:
    with open(config_path, "r", encoding="utf-8") as fh:
        return Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, overrides))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune ViT5/BARTpho for Vie->VSL gloss")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run(_load_cfg(args.config, args.set))


if __name__ == "__main__":
    main()
