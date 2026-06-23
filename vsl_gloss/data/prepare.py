"""Stage 1 of the data pipeline: build a single clean, analysed corpus file.

Reads the four raw, line-aligned files (Vie / VSL text + their gold parses),
normalises every line, attaches gold word-segmentation + POS from the Vie tree,
labels each pair with a transformation :class:`~vsl_gloss.data.taxonomy.Category`,
and writes ``data/processed/corpus.jsonl`` together with a corpus-statistics
report under ``reports/``.

Run::

    python -m vsl_gloss.data.prepare --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from ..config import Config
from ..utils import apply_overrides, get_logger, read_lines, write_json, write_jsonl
from .normalize import NormalizeOptions, normalize_text, tokenize
from .parse_tree import parse_line
from .taxonomy import ALL_CATEGORIES, classify

LOG = get_logger("prepare")


def _norm_opts(cfg: Config, lowercase: bool = False) -> NormalizeOptions:
    return NormalizeOptions(
        fix_punct_spacing=cfg.data.fix_punct_spacing,
        collapse_whitespace=cfg.data.collapse_whitespace,
        lowercase=lowercase,
    )


def build_corpus(cfg: Config) -> List[Dict]:
    raw = cfg.paths.resolved("raw_dir")
    vie = read_lines(raw / cfg.data.vie_file)
    vsl = read_lines(raw / cfg.data.vsl_file)
    vie_tree_lines = read_lines(raw / cfg.data.vie_tree_file)
    vsl_tree_lines = read_lines(raw / cfg.data.vsl_tree_file)

    n = len(vie)
    if not (len(vsl) == len(vie_tree_lines) == len(vsl_tree_lines) == n):
        raise ValueError(
            "Raw files are not line-aligned: "
            f"{len(vie)=}, {len(vsl)=}, {len(vie_tree_lines)=}, {len(vsl_tree_lines)=}"
        )
    LOG.info("Loaded %d line-aligned raw pairs from %s", n, raw)

    opts = _norm_opts(cfg)
    records: List[Dict] = []
    tree_ok_count = 0
    for i in range(n):
        v = normalize_text(vie[i], opts)
        s = normalize_text(vsl[i], opts)
        if cfg.data.drop_empty and (not v or not s):
            continue

        vt = parse_line(vie_tree_lines[i])
        st = parse_line(vsl_tree_lines[i])
        vie_pos = vt.pos_tags() if vt else None
        vie_seg = vt.words(syllables=False) if vt else None  # underscore-joined words

        # Does the gold parse's surface form match the normalised source?
        tree_ok = False
        if vt is not None:
            tyield = normalize_text(vt.yield_text(syllables=True), opts)
            tree_ok = tyield.lower().rstrip(" .?!") == v.lower().rstrip(" .?!")
            tree_ok_count += int(tree_ok)

        src_tokens, tgt_tokens = tokenize(v), tokenize(s)
        records.append(
            {
                "id": i,
                "vie": v,
                "vsl": s,
                "src_len": len(src_tokens),
                "tgt_len": len(tgt_tokens),
                "category": classify(src_tokens, tgt_tokens),
                "vie_pos": vie_pos,
                "vie_seg": vie_seg,
                "vie_tree": vie_tree_lines[i].strip(),
                "vsl_tree": vsl_tree_lines[i].strip(),
                "tree_ok": tree_ok,
            }
        )
    LOG.info("Kept %d records; gold parse matched source on %d (%.1f%%)",
             len(records), tree_ok_count, 100 * tree_ok_count / max(1, len(records)))
    return records


def corpus_stats(records: List[Dict]) -> Dict:
    cats = Counter(r["category"] for r in records)
    src_lens = [r["src_len"] for r in records]
    tgt_lens = [r["tgt_len"] for r in records]
    pairs = [(r["vie"], r["vsl"]) for r in records]
    src_vocab = Counter(t for r in records for t in r["vie"].split())
    tgt_vocab = Counter(t for r in records for t in r["vsl"].split())

    def _avg(xs):
        return round(sum(xs) / max(1, len(xs)), 3)

    return {
        "num_records": len(records),
        "num_unique_pairs": len(set(pairs)),
        "num_unique_sources": len({v for v, _ in pairs}),
        "category_counts": {c: cats.get(c, 0) for c in ALL_CATEGORIES},
        "category_pct": {
            c: round(100 * cats.get(c, 0) / max(1, len(records)), 2) for c in ALL_CATEGORIES
        },
        "src_len": {"mean": _avg(src_lens), "max": max(src_lens), "min": min(src_lens)},
        "tgt_len": {"mean": _avg(tgt_lens), "max": max(tgt_lens), "min": min(tgt_lens)},
        "len_delta_mean": _avg([t - s for s, t in zip(src_lens, tgt_lens)]),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "tgt_only_tokens": len(set(tgt_vocab) - set(src_vocab)),
        "tree_ok_count": sum(int(r["tree_ok"]) for r in records),
    }


def _stats_markdown(stats: Dict) -> str:
    lines = ["# Corpus statistics", ""]
    lines.append(f"- Records: **{stats['num_records']}**  |  unique pairs: "
                 f"{stats['num_unique_pairs']}  |  unique sources: {stats['num_unique_sources']}")
    lines.append(f"- Source length (tokens): mean {stats['src_len']['mean']}, "
                 f"max {stats['src_len']['max']}")
    lines.append(f"- Target length (tokens): mean {stats['tgt_len']['mean']}, "
                 f"max {stats['tgt_len']['max']}  |  mean Δlen {stats['len_delta_mean']}")
    lines.append(f"- Vocab: src {stats['src_vocab_size']}, tgt {stats['tgt_vocab_size']}, "
                 f"target-only tokens {stats['tgt_only_tokens']}")
    lines += ["", "## Transformation categories", "", "| category | count | % |", "|---|---:|---:|"]
    for c in ALL_CATEGORIES:
        lines.append(f"| {c} | {stats['category_counts'][c]} | {stats['category_pct'][c]} |")
    return "\n".join(lines) + "\n"


def run(cfg: Config) -> Path:
    records = build_corpus(cfg)
    out = cfg.paths.resolved("processed_dir") / "corpus.jsonl"
    write_jsonl(out, records)
    LOG.info("Wrote %d records -> %s", len(records), out)

    stats = corpus_stats(records)
    reports = cfg.paths.resolved("reports_dir")
    write_json(reports / "corpus_stats.json", stats)
    (reports / "corpus_stats.md").write_text(_stats_markdown(stats), encoding="utf-8")
    LOG.info("Category distribution: %s", stats["category_pct"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cleaned, analysed corpus.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.lowercase=true")
    args = ap.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg_dict = apply_overrides(yaml.safe_load(fh) or {}, args.set)
    run(Config.from_dict(cfg_dict))


if __name__ == "__main__":
    main()
