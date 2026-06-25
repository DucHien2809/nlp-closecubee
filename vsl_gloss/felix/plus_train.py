"""Train FELIX++ and emit standard prediction files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

from ..config import Config
from ..metrics import compute_metrics, metrics_backend
from ..utils import apply_overrides, get_logger, read_jsonl, set_seed, write_json, write_jsonl
from .plus_data import FelixPlusCollator, FelixPlusDataset, build_insertion_vocab, default_format_vocab
from .plus_decode import EditCandidate, build_candidates_from_predictions, verify_prediction_alignment
from .plus_model import FelixPlusModel
from .rerank import RerankWeights, choose_candidate, tune_weights

LOG = get_logger("felix.plus_train")


def read_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def loss_weights_for_epoch(epoch: int, tag_warmup_epochs: int, pointer_warmup_epochs: int) -> Dict[str, float]:
    if epoch <= tag_warmup_epochs:
        return {"tag": 1.0, "pointer": 0.0, "insertion": 0.0, "format": 0.0}
    if epoch <= tag_warmup_epochs + pointer_warmup_epochs:
        return {"tag": 1.0, "pointer": 1.0, "insertion": 0.0, "format": 0.0}
    return {"tag": 1.0, "pointer": 1.0, "insertion": 1.0, "format": 1.0}


def metric_is_better(current: Dict, best: Dict, metric: str) -> bool:
    if not best:
        return True
    if metric == "wer":
        return current.get("wer", 1e9) < best.get("wer", 1e9)
    return current.get(metric, -1e9) > best.get(metric, -1e9)


def build_encoder(encoder_name: str):
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(encoder_name)
    return encoder, encoder.config.hidden_size


def save_checkpoint(model: FelixPlusModel, tokenizer, out_dir: Path, insertion_vocab: Dict[str, int], format_vocab: Dict[str, int], rerank_weights: RerankWeights | None = None) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out_dir / "encoder"))
    tokenizer.save_pretrained(str(out_dir / "encoder"))
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(heads, out_dir / "heads.pt")
    write_json(out_dir / "insertion_vocab.json", insertion_vocab)
    write_json(out_dir / "format_vocab.json", format_vocab)
    if rerank_weights is not None:
        write_json(out_dir / "rerank_weights.json", rerank_weights.values)


def _set_model_loss_weights(model: FelixPlusModel, base_cfg, stage_weights: Dict[str, float]) -> None:
    model.tag_loss_weight = base_cfg.tag_loss_weight * stage_weights["tag"]
    model.pointer_loss_weight = base_cfg.pointer_loss_weight * stage_weights["pointer"]
    model.insertion_loss_weight = base_cfg.insertion_loss_weight * stage_weights["insertion"]
    model.format_loss_weight = base_cfg.format_loss_weight * stage_weights["format"]


def category_sample_weights(dataset: FelixPlusDataset, category_weights: Dict[str, float]) -> List[float]:
    weights = []
    for ex in dataset.examples:
        weights.append(float(category_weights.get(ex.get("category"), 1.0)))
    return weights


def run(cfg: Config) -> Dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    fcfg = cfg.felix_plus
    set_seed(fcfg.seed)
    LOG.info("FELIX++ encoder: %s | metric backends: %s", fcfg.encoder_name, metrics_backend())

    splits = cfg.paths.resolved("splits_dir")
    train_path = splits / "train.jsonl"
    val_path = splits / "val.jsonl"
    test_path = splits / "test.jsonl"
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required existing split file not found: {path}")

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    tokenizer = AutoTokenizer.from_pretrained(fcfg.encoder_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(f"{fcfg.encoder_name} needs a fast tokenizer for word_ids().")

    insertion_vocab = build_insertion_vocab(
        train_records,
        min_count=fcfg.insertion_min_count,
        max_phrase_len=fcfg.max_insertion_phrase_len,
    )
    format_vocab = default_format_vocab()

    train_ds = FelixPlusDataset(train_records, tokenizer, insertion_vocab, format_vocab, fcfg.max_source_length)
    collator = FelixPlusCollator(tokenizer.pad_token_id)
    sampler = None
    shuffle = True
    if fcfg.category_weighting:
        sampler = torch.utils.data.WeightedRandomSampler(
            category_sample_weights(train_ds, fcfg.category_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=fcfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
    )

    encoder, hidden = build_encoder(fcfg.encoder_name)
    model = FelixPlusModel(
        encoder=encoder,
        hidden_size=hidden,
        pointer_dim=fcfg.pointer_dim,
        insertion_vocab_size=len(insertion_vocab),
        format_vocab_size=len(format_vocab),
        dropout=fcfg.dropout,
        tag_loss_weight=fcfg.tag_loss_weight,
        pointer_loss_weight=fcfg.pointer_loss_weight,
        insertion_loss_weight=fcfg.insertion_loss_weight,
        format_loss_weight=fcfg.format_loss_weight,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optim = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": fcfg.learning_rate},
            {"params": head_params, "lr": fcfg.learning_rate * fcfg.head_learning_rate_multiplier},
        ],
        weight_decay=fcfg.weight_decay,
    )
    total_steps = max(1, len(train_loader)) * int(round(fcfg.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(optim, int(total_steps * fcfg.warmup_ratio), total_steps)

    out_dir = cfg.paths.resolved("output_dir") / cfg.experiment_name
    best_metrics: Dict = {}
    best_state = None

    for epoch in range(1, int(round(fcfg.num_train_epochs)) + 1):
        model.train()
        _set_model_loss_weights(model, fcfg, loss_weights_for_epoch(epoch, fcfg.tag_warmup_epochs, fcfg.pointer_warmup_epochs))
        for batch in train_loader:
            feats = {
                key: batch[key].to(device)
                for key in (
                    "input_ids",
                    "attention_mask",
                    "first_subword_idx",
                    "word_mask",
                    "tag_labels",
                    "succ_target",
                    "key_keep_mask",
                    "order_target",
                    "insertion_labels",
                    "insertion_mask",
                    "format_labels",
                )
            }
            out = model(**feats)
            optim.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg.max_grad_norm)
            optim.step()
            scheduler.step()

        val_preds = decode_records(val_records, tokenizer, model, device, cfg, insertion_vocab, format_vocab)
        val_metrics = compute_metrics(val_preds, [r["vsl"] for r in val_records])
        LOG.info("epoch %d VAL %s", epoch, val_metrics)
        if metric_is_better(val_metrics, best_metrics, fcfg.selection_metric):
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, tokenizer, out_dir / "model", insertion_vocab, format_vocab)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device).eval()

    pred_path = predict_split(cfg, str(out_dir / "model"), split="test", name=cfg.experiment_name)
    verify_prediction_alignment(pred_path, test_path)

    from ..evaluate import score_file

    report = score_file(
        pred_path,
        cfg,
        report_categories=cfg.eval.report_categories,
        report_normalized=cfg.eval.report_normalized,
        error_examples=cfg.eval.error_examples_per_category,
    )
    cfg.save(out_dir / "config_used.yaml")
    return report


def decode_records(records: List[Dict], tokenizer, model, device, cfg: Config, insertion_vocab: Dict[str, int], format_vocab: Dict[str, int]) -> List[str]:
    from .plus_decode import render_edit

    # First implementation uses deterministic tag+pointer+no-extra-candidate decode.
    # Task 9 upgrades this path to candidate generation plus reranking.
    import torch
    from .plus_data import FelixPlusCollator, build_plus_example

    inv_format = {v: k for k, v in format_vocab.items()}
    inv_insert = {v: k for k, v in insertion_vocab.items()}
    collator = FelixPlusCollator(tokenizer.pad_token_id)
    fcfg = cfg.felix_plus
    preds: List[str] = []

    model.eval()
    with torch.no_grad():
        for rec in records:
            ex = build_plus_example(rec, tokenizer, insertion_vocab, format_vocab, fcfg.max_source_length)
            if ex is None:
                preds.append(rec["vie"])
                continue
            batch = collator([ex])
            feats = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "first_subword_idx", "word_mask")
            }
            out = model(**feats)
            keep = (out.tag_logits.argmax(-1)[0].cpu().tolist())
            order = [i for i, tag in enumerate(keep[: len(ex["src_tokens"])]) if tag == 0]
            insert_logits = out.insertion_logits.argmax(-1)[0].cpu().tolist()
            insertions = {}
            for slot, label_id in enumerate(insert_logits[: len(order) + 1]):
                phrase = inv_insert.get(int(label_id), "NONE")
                if phrase != "NONE":
                    insertions[slot] = tuple(phrase.split())
            fmt_id = int(out.format_logits.argmax(-1)[0].cpu().item())
            fmt = inv_format.get(fmt_id, "final=NONE|case=preserve|spacing=1")
            preds.append(render_edit(ex["src_tokens"], order, insertions, fmt))
    return preds


def predict_split(cfg: Config, model_dir: str, split: str, name: str) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "encoder"), use_fast=True)
    encoder = AutoModel.from_pretrained(str(model_dir / "encoder"))
    insertion_vocab = {k: int(v) for k, v in read_json(model_dir / "insertion_vocab.json").items()}
    format_vocab = {k: int(v) for k, v in read_json(model_dir / "format_vocab.json").items()}

    model = FelixPlusModel(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        pointer_dim=cfg.felix_plus.pointer_dim,
        insertion_vocab_size=len(insertion_vocab),
        format_vocab_size=len(format_vocab),
        dropout=cfg.felix_plus.dropout,
    )
    heads = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.load_state_dict(heads, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    records = read_jsonl(cfg.paths.resolved("splits_dir") / f"{split}.jsonl")
    preds = decode_records(records, tokenizer, model, device, cfg, insertion_vocab, format_vocab)
    rows = [
        {"id": r["id"], "vie": r["vie"], "vsl": r["vsl"], "category": r["category"], "pred": p}
        for r, p in zip(records, preds)
    ]
    out = cfg.paths.resolved("output_dir") / name / f"predictions_{split}.jsonl"
    write_jsonl(out, rows)
    return out


def _load_cfg(config_path: str, overrides: List[str]) -> Config:
    with open(config_path, "r", encoding="utf-8") as fh:
        return Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, overrides))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FELIX++ for Vie -> VSL gloss")
    ap.add_argument("--config", default="configs/felix_plus.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run(_load_cfg(args.config, args.set))


if __name__ == "__main__":
    main()
