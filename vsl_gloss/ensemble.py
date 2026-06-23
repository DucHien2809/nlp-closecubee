"""Candidate-pool MBR ensemble of fine-tuned seq2seq members.

The members (e.g. ViT5-base + BARTpho) have *different* tokenizers, so their
logits cannot be averaged. Instead each member draws an unbiased sample of
hypotheses; we pool the samples across members and select the single hypothesis
with the highest expected utility against the pool (Minimum Bayes Risk; Eikema &
Aziz, EMNLP 2020). The pool is a Monte-Carlo estimate of the *consensus*
distribution across both models, so the chosen output is the one both models
agree is most central — a principled, tokenizer-agnostic ensemble.

Each member is referenced by its ``output_dir`` name; its own
``config_used.yaml`` supplies the right ``source_prefix`` and tokenizer type.

Run (after both members are trained)::

    python -m vsl_gloss.ensemble --config configs/ensemble_mbr.yaml --split test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from .config import Config
from .data.normalize import NormalizeOptions, normalize_text
from .models.mbr import mbr_select_batch
from .utils import apply_overrides, get_logger, read_jsonl, set_seed, write_jsonl

LOG = get_logger("ensemble")
_NORM = NormalizeOptions(lowercase=False)


@dataclass
class Member:
    name: str
    model: object
    tokenizer: object
    prefix: str


def load_member(cfg: Config, name: str, device) -> Member:
    """Load a trained member by its ``output_dir`` name, using the member's own
    saved config for the task prefix / tokenizer type when present.

    Raises a clear ``FileNotFoundError`` if the member was never trained, and
    repairs a missing pad token so batched generation cannot crash.
    """
    member_dir = cfg.paths.resolved("output_dir") / name
    model_dir = member_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Ensemble member '{name}' has no trained model at {model_dir}. "
            f"Train it first (e.g. modal run modal_app.py::train --config configs/bartpho.yaml)."
        )

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    used = member_dir / "config_used.yaml"
    mcfg = Config.load(used) if used.exists() else cfg
    if not used.exists():
        LOG.warning("%s has no config_used.yaml; falling back to the active config "
                    "(prefix=%r, fast=%s)", name, mcfg.model.source_prefix,
                    mcfg.model.use_fast_tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=mcfg.model.use_fast_tokenizer
    )
    # Batched generation needs padding; some checkpoints ship without a pad token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        LOG.warning("%s tokenizer had no pad token; using %r", name, tokenizer.pad_token)

    # bf16 on CUDA halves memory + speeds sampling; fp32 elsewhere for safety.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), torch_dtype=dtype)
    except (TypeError, ValueError):       # older transformers / unsupported dtype
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model = model.to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    LOG.info("Loaded member %s (prefix=%r, dtype=%s)", name, mcfg.model.source_prefix, dtype)
    return Member(name=name, model=model, tokenizer=tokenizer, prefix=mcfg.model.source_prefix)


def _decode(tokenizer, output) -> List[str]:
    return [normalize_text(x, _NORM)
            for x in tokenizer.batch_decode(output, skip_special_tokens=True)]


def _is_oom(err: Exception) -> bool:
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def _generate_block(member: Member, batch: List[str], device, cfg: Config,
                    gen_kwargs: dict, n_return: int) -> List[List[str]]:
    """Generate for one block of sources, returning ``n_return`` decoded strings
    per source. On CUDA OOM the block is split in half and retried recursively so
    a single oversized batch never aborts the whole run."""
    import torch

    try:
        enc = member.tokenizer(
            [member.prefix + t for t in batch],
            return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.model.max_source_length,
        ).to(device)
        with torch.no_grad():
            out = member.model.generate(**enc, **gen_kwargs)
        dec = _decode(member.tokenizer, out)
        return [dec[i * n_return:(i + 1) * n_return] for i in range(len(batch))]
    except Exception as err:                      # noqa: BLE001 - we re-raise non-OOM
        if _is_oom(err) and len(batch) > 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(batch) // 2
            LOG.warning("CUDA OOM on a block of %d; retrying as %d + %d",
                        len(batch), mid, len(batch) - mid)
            left = _generate_block(member, batch[:mid], device, cfg, gen_kwargs, n_return)
            right = _generate_block(member, batch[mid:], device, cfg, gen_kwargs, n_return)
            return left + right
        raise


def member_candidates(
    member: Member, texts: List[str], device, cfg: Config
) -> "tuple[List[List[str]], List[str]]":
    """Return (samples_per_text, beam_best_per_text) for one member.

    ``samples_per_text[i]`` holds ``num_samples`` ancestral samples (the MBR pool
    contribution); ``beam_best_per_text[i]`` is the deterministic beam-1-best.
    Sampling parameters are clamped to valid ranges so a misconfigured YAML can
    never raise inside ``generate``.
    """
    ec = cfg.ensemble
    n = len(texts)
    samples: List[List[str]] = [[] for _ in range(n)]
    beam_best: List[str] = [""] * n
    bs = max(1, int(ec.batch_size))
    n_samples = max(1, int(ec.num_samples))
    temperature = max(1e-3, float(ec.temperature))    # 0 would crash sampling
    top_p = min(1.0, max(1e-3, float(ec.top_p)))
    max_new = max(1, int(cfg.decode.max_new_tokens))

    sample_kwargs = dict(
        do_sample=True, num_return_sequences=n_samples, top_p=top_p,
        temperature=temperature, num_beams=1, max_new_tokens=max_new,
    )
    beam_kwargs = dict(
        num_beams=max(1, int(ec.num_beams)), num_return_sequences=1,
        max_new_tokens=max_new, length_penalty=cfg.decode.length_penalty,
        early_stopping=cfg.decode.early_stopping,
    )

    for start in range(0, n, bs):
        batch = texts[start : start + bs]
        block = _generate_block(member, batch, device, cfg, sample_kwargs, n_samples)
        for i, cands in enumerate(block):
            samples[start + i] = cands
        if ec.include_beam:
            beam = _generate_block(member, batch, device, cfg, beam_kwargs, 1)
            for i, cands in enumerate(beam):
                beam_best[start + i] = cands[0] if cands else ""
    return samples, beam_best


def run(cfg: Config, split: str = "test") -> Path:
    import torch

    members = [m for m in (cfg.ensemble.members or []) if m and m.strip()]
    if not members:
        raise ValueError("ensemble.members is empty — list at least one trained system.")
    if cfg.ensemble.utility not in ("chrf", "bleu"):
        LOG.warning("Unknown ensemble.utility %r; defaulting to chrf", cfg.ensemble.utility)

    split_file = cfg.paths.resolved("splits_dir") / f"{split}.jsonl"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Run data prepare+split first."
        )

    set_seed(cfg.ensemble.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOG.warning("No CUDA device — MBR sampling on CPU will be slow.")

    records = read_jsonl(split_file)
    texts = [r["vie"] for r in records]
    if not texts:
        raise ValueError(f"Split {split} is empty ({split_file}).")
    LOG.info("MBR ensemble of %s over %d sents (split=%s, samples/member=%d, utility=%s)",
             members, len(texts), split, cfg.ensemble.num_samples, cfg.ensemble.utility)

    pools: List[List[str]] = [[] for _ in texts]
    extras: List[List[str]] = [[] for _ in texts]
    for name in members:
        member = load_member(cfg, name, device)
        try:
            samples, beam_best = member_candidates(member, texts, device, cfg)
        finally:
            del member  # free GPU memory even if generation raised
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for i in range(len(texts)):
            pools[i].extend(samples[i])
            if cfg.ensemble.include_beam and beam_best[i]:
                extras[i].append(beam_best[i])

    LOG.info("Selecting MBR-optimal hypotheses…")
    preds = mbr_select_batch(pools, extras, utility_name=cfg.ensemble.utility)
    n_empty = sum(1 for p in preds if not p.strip())
    if n_empty:
        LOG.warning("%d/%d MBR predictions are empty (degenerate samples).",
                    n_empty, len(preds))

    rows: List[Dict] = [
        {"id": r["id"], "vie": r["vie"], "vsl": r["vsl"],
         "category": r["category"], "pred": p}
        for r, p in zip(records, preds)
    ]
    out = cfg.paths.resolved("output_dir") / cfg.ensemble.name / f"predictions_{split}.jsonl"
    write_jsonl(out, rows)
    LOG.info("Wrote %d MBR predictions -> %s", len(rows), out)

    from .evaluate import score_file

    report = score_file(
        out, cfg,
        report_categories=cfg.eval.report_categories,
        report_normalized=cfg.eval.report_normalized,
        error_examples=cfg.eval.error_examples_per_category,
    )
    LOG.info("[%s] %s overall: %s", cfg.ensemble.name, split, report["overall"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MBR-ensemble already-trained seq2seq members")
    ap.add_argument("--config", default="configs/ensemble_mbr.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))
    run(cfg, split=args.split)


if __name__ == "__main__":
    main()
