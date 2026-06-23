"""Canonical scorer and report generator.

Any system -- copy baseline, rule baseline, ViT5, BARTpho -- emits a predictions
file with the schema ``{id, vie, vsl, category, pred}``. This module scores such
a file uniformly, writing:

* ``metrics_<split>.json`` -- machine-readable overall + per-category metrics,
* ``report_<split>.md``    -- a human-readable table,
* ``errors_<split>.md``    -- qualitative error examples per category,

and can build a cross-system ``leaderboard.md``.

Run::

    python -m vsl_gloss.evaluate --predictions outputs/baseline_copy/predictions_test.jsonl
    python -m vsl_gloss.evaluate --compare            # score every system, build leaderboard
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import Config
from .data.taxonomy import ALL_CATEGORIES
from .metrics import compute_report, metrics_backend
from .utils import apply_overrides, get_logger, read_jsonl, write_json, write_text

LOG = get_logger("evaluate")

_METRIC_ORDER = ["bleu", "chrf", "wer", "ter", "exact_match", "n"]


def _split_of(path: Path) -> str:
    stem = path.stem  # predictions_test
    return stem.split("_")[-1] if "_" in stem else "test"


def _fmt_row(name: str, m: Dict) -> str:
    cells = [name]
    for k in _METRIC_ORDER:
        v = m.get(k)
        cells.append("-" if v is None else (str(v) if k == "n" else f"{v:.2f}"))
    return "| " + " | ".join(cells) + " |"


def _report_markdown(system: str, split: str, report: Dict) -> str:
    header = "| set | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = [f"# {system} — {split}", "", header, _fmt_row("overall", report["overall"])]
    if "overall_lowercased" in report:
        lines.append(_fmt_row("overall (lc)", report["overall_lowercased"]))
    if "by_category" in report:
        lines += ["", "## By transformation category", "", header.replace("set", "category")]
        for cat in ALL_CATEGORIES:
            if cat in report["by_category"]:
                lines.append(_fmt_row(cat, report["by_category"][cat]))
    return "\n".join(lines) + "\n"


def _error_markdown(system: str, split: str, rows: List[Dict], per_cat: int) -> str:
    out = [f"# {system} — error examples ({split})", ""]
    for cat in ALL_CATEGORIES:
        wrong = [r for r in rows if r["category"] == cat and r["pred"].strip() != r["vsl"].strip()]
        if not wrong:
            continue
        out.append(f"## {cat}  ({len(wrong)} mismatched)")
        out.append("")
        for r in wrong[:per_cat]:
            out.append(f"- SRC : `{r['vie']}`")
            out.append(f"  REF : `{r['vsl']}`")
            out.append(f"  HYP : `{r['pred']}`")
        out.append("")
    return "\n".join(out) + "\n"


def score_file(
    predictions_path: str | Path,
    cfg: Optional[Config] = None,
    report_categories: bool = True,
    report_normalized: bool = True,
    error_examples: int = 15,
) -> Dict:
    predictions_path = Path(predictions_path)
    rows = read_jsonl(predictions_path)
    system = predictions_path.parent.name
    split = _split_of(predictions_path)

    preds = [r["pred"] for r in rows]
    refs = [r["vsl"] for r in rows]
    cats = [r.get("category") for r in rows] if report_categories else None

    LOG.info("Scoring %s (%s, n=%d); backends=%s",
             system, split, len(rows), metrics_backend())
    report = compute_report(preds, refs, categories=cats, normalized=report_normalized)
    report["_meta"] = {"system": system, "split": split, "n": len(rows),
                       "predictions": str(predictions_path)}

    out_dir = predictions_path.parent
    write_json(out_dir / f"metrics_{split}.json", report)
    write_text(out_dir / f"report_{split}.md", _report_markdown(system, split, report))
    write_text(out_dir / f"errors_{split}.md", _error_markdown(system, split, rows, error_examples))
    LOG.info("[%s] overall: %s", system, report["overall"])
    return report


def build_leaderboard(cfg: Config, split: str = "test") -> Path:
    """Scan ``output_dir`` for predictions files and rank systems by BLEU/WER."""
    output_dir = cfg.paths.resolved("output_dir")
    files = sorted(output_dir.glob(f"*/predictions_{split}.jsonl"))
    rows = []
    for f in files:
        rep = score_file(f, cfg,
                         report_categories=cfg.eval.report_categories,
                         report_normalized=cfg.eval.report_normalized,
                         error_examples=cfg.eval.error_examples_per_category)
        rows.append((f.parent.name, rep["overall"]))

    rows.sort(key=lambda x: (-x[1].get("bleu", 0), x[1].get("wer", 1e9)))
    header = "| system | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |\n|---|---:|---:|---:|---:|---:|---:|"
    md = [f"# Leaderboard — {split}", "", header]
    md += [_fmt_row(name, m) for name, m in rows]
    out = cfg.paths.resolved("reports_dir") / f"leaderboard_{split}.md"
    write_text(out, "\n".join(md) + "\n")
    LOG.info("Wrote leaderboard (%d systems) -> %s", len(rows), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score predictions / build leaderboard")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictions", help="path to a predictions_*.jsonl file")
    ap.add_argument("--compare", action="store_true", help="score all systems + leaderboard")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    if args.compare:
        build_leaderboard(cfg, split=args.split)
    elif args.predictions:
        score_file(args.predictions, cfg,
                   report_categories=cfg.eval.report_categories,
                   report_normalized=cfg.eval.report_normalized,
                   error_examples=cfg.eval.error_examples_per_category)
    else:
        ap.error("pass --predictions PATH or --compare")


if __name__ == "__main__":
    main()
