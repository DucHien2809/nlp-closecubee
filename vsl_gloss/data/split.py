"""Stage 2 of the data pipeline: a leakage-free, stratified train/val/test split.

Two properties make the split trustworthy -- without them the headline scores
are easy to inflate:

1. **No source leakage.** Identical source sentences (377 of them recur, 21 with
   *different* glosses) are grouped and assigned to a single split, so the model
   is never tested on a sentence it memorised verbatim during training.
2. **Stratification by transformation category.** Each split preserves the
   corpus-wide ratio of copy / reorder / deletion / lexical pairs, so val and
   test are representative and per-category metrics are comparable across splits.

Run::

    python -m vsl_gloss.data.split --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from ..config import Config
from ..utils import apply_overrides, get_logger, read_jsonl, write_json, write_jsonl

LOG = get_logger("split")


def _group_key(rec: Dict) -> str:
    """Group identical source sentences (case-folded) together."""
    return rec["vie"].lower()


def _group_category(recs: List[Dict]) -> str:
    """Representative category of a group = its most common member category."""
    return Counter(r["category"] for r in recs).most_common(1)[0][0]


def make_splits(records: List[Dict], cfg: Config) -> Dict[str, List[Dict]]:
    rng = random.Random(cfg.data.seed)

    # 1. Optionally drop exact duplicate (src, tgt) pairs (keep first occurrence).
    if cfg.data.dedup_exact_pairs:
        seen, deduped = set(), []
        for r in records:
            key = (r["vie"], r["vsl"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        LOG.info("Dedup exact pairs: %d -> %d", len(records), len(deduped))
        records = deduped

    # 2. Group by source sentence to prevent leakage.
    if cfg.data.group_by_source:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            groups[_group_key(r)].append(r)
        group_items = list(groups.values())
    else:
        group_items = [[r] for r in records]
    LOG.info("Formed %d groups from %d records", len(group_items), len(records))

    # 3. Bucket groups by representative category for stratification.
    buckets: Dict[str, List[List[Dict]]] = defaultdict(list)
    for g in group_items:
        cat = _group_category(g) if cfg.data.stratify_by_category else "_all"
        buckets[cat].append(g)

    val_r, test_r = cfg.data.val_ratio, cfg.data.test_ratio
    out: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    # 4. Split each bucket independently, then merge.
    for cat, groups in sorted(buckets.items()):
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_r))
        n_val = int(round(n * val_r))
        # Guarantee tiny buckets still contribute to train.
        n_val = min(n_val, max(0, n - n_test - 1)) if n - n_test >= 1 else 0
        test_g = groups[:n_test]
        val_g = groups[n_test : n_test + n_val]
        train_g = groups[n_test + n_val :]
        for split, gs in (("train", train_g), ("val", val_g), ("test", test_g)):
            for g in gs:
                out[split].extend(g)
        LOG.info("  [%-16s] groups train/val/test = %d/%d/%d",
                 cat, len(train_g), len(val_g), len(test_g))

    # Stable, reproducible ordering within each split.
    for split in out:
        out[split].sort(key=lambda r: r["id"])
    return out


def verify_no_leakage(splits: Dict[str, List[Dict]]) -> None:
    src = {name: {r["vie"].lower() for r in recs} for name, recs in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = src[a] & src[b]
        if overlap:
            raise AssertionError(f"Source leakage between {a} and {b}: {len(overlap)} sentences")
    LOG.info("Leakage check passed: no source sentence is shared across splits.")


def split_stats(splits: Dict[str, List[Dict]]) -> Dict:
    stats: Dict = {"sizes": {k: len(v) for k, v in splits.items()}}
    stats["category_pct"] = {}
    for name, recs in splits.items():
        c = Counter(r["category"] for r in recs)
        total = max(1, len(recs))
        stats["category_pct"][name] = {k: round(100 * v / total, 2) for k, v in sorted(c.items())}
    return stats


def run(cfg: Config) -> Dict[str, Path]:
    corpus = cfg.paths.resolved("processed_dir") / "corpus.jsonl"
    records = read_jsonl(corpus)
    splits = make_splits(records, cfg)
    verify_no_leakage(splits)

    out_dir = cfg.paths.resolved("splits_dir")
    paths: Dict[str, Path] = {}
    for name, recs in splits.items():
        p = out_dir / f"{name}.jsonl"
        write_jsonl(p, recs)
        paths[name] = p
        LOG.info("Wrote %5d -> %s", len(recs), p)

    stats = split_stats(splits)
    write_json(cfg.paths.resolved("reports_dir") / "split_stats.json", stats)
    LOG.info("Split sizes: %s", stats["sizes"])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Create leakage-free stratified splits")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg_dict = apply_overrides(yaml.safe_load(fh) or {}, args.set)
    run(Config.from_dict(cfg_dict))


if __name__ == "__main__":
    main()
