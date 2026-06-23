"""Oracle edit-label extraction + coverage analysis (the FELIX go/no-go gate).

Given a parallel pair ``(vie, vsl)`` of whitespace-tokenised sentences, derive the
*gold* edit program a tag+pointer model would have to predict:

* a **tag** per source token -- ``KEEP`` or ``DELETE``;
* an **order** -- the kept source indices listed in target order (the pointer
  network's job); ``order`` is monotonic exactly when no reordering is needed;
* any **insertions** -- target tokens with no matching source token (the rare
  genuinely-lexical ~2%, which a pure tag+pointer model cannot produce).

The alignment is a greedy left-to-right matching on case-folded surfaces with a
per-surface queue, so duplicate tokens (e.g. ``bạn ... bạn``) are matched to
distinct source positions. This mirrors the corpus taxonomy, which also matches
on the case-folded multiset.

Running this module prints the **coverage** of the corpus -- what fraction of
pairs a tag+pointer model can represent exactly, split into *tag-only* (no
reorder), *needs-pointer*, and *needs-insertion* -- and the **oracle ceiling**
(scoring gold edit programs through the project's own metrics). Together these
say how much headroom an edit model has *before* a single parameter is trained::

    python -m vsl_gloss.felix.labels --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import Config
from ..utils import apply_overrides, get_logger, read_jsonl, write_json

LOG = get_logger("felix.labels")

KEEP, DELETE = "KEEP", "DELETE"


@dataclass
class EditLabels:
    source_tokens: List[str]
    target_tokens: List[str]
    tags: List[str]                              # len == len(source_tokens)
    order: List[int]                             # kept source indices, in target order
    insertions: List[Tuple[int, str]] = field(default_factory=list)  # (slot in order, token)

    @property
    def reconstructable(self) -> bool:
        """True when delete+reorder alone reproduces the target (no insertion)."""
        return not self.insertions

    @property
    def needs_reorder(self) -> bool:
        return any(a > b for a, b in zip(self.order, self.order[1:]))

    @property
    def kind(self) -> str:
        if not self.reconstructable:
            return "needs_insertion"
        return "needs_pointer" if self.needs_reorder else "tag_only"


def tokenize(text: str) -> List[str]:
    return text.split()


def extract_labels(src_tokens: List[str], tgt_tokens: List[str]) -> EditLabels:
    """Greedy case-folded alignment of target tokens back onto source positions."""
    buckets: Dict[str, deque] = defaultdict(deque)
    for i, w in enumerate(src_tokens):
        buckets[w.lower()].append(i)

    order: List[int] = []
    insertions: List[Tuple[int, str]] = []
    used = [False] * len(src_tokens)
    for tok in tgt_tokens:
        q = buckets.get(tok.lower())
        if q:
            i = q.popleft()
            used[i] = True
            order.append(i)
        else:
            insertions.append((len(order), tok))

    tags = [KEEP if used[i] else DELETE for i in range(len(src_tokens))]
    return EditLabels(src_tokens, tgt_tokens, tags, order, insertions)


def reconstruct(labels: EditLabels, use_insertions: bool = True) -> str:
    """Render an edit program back to a gloss string (the oracle prediction)."""
    out: List[str] = [labels.source_tokens[i] for i in labels.order]
    if use_insertions and labels.insertions:
        # Splice insertion tokens back at their recorded slot (right-to-left to
        # keep earlier slots valid).
        for slot, tok in sorted(labels.insertions, reverse=True):
            out.insert(slot, tok)
    return " ".join(out)


def labels_for_record(rec: Dict) -> EditLabels:
    return extract_labels(tokenize(rec["vie"]), tokenize(rec["vsl"]))


# ---------------------------------------------------------------- coverage gate
def coverage(records: List[Dict]) -> Dict:
    kinds = Counter()
    kinds_by_cat: Dict[str, Counter] = defaultdict(Counter)
    n_ins_tokens = 0
    for r in records:
        lab = labels_for_record(r)
        kinds[lab.kind] += 1
        kinds_by_cat[r.get("category", "?")][lab.kind] += 1
        n_ins_tokens += len(lab.insertions)

    n = max(1, len(records))
    pct = {k: round(100 * v / n, 2) for k, v in kinds.items()}
    return {
        "n": len(records),
        "counts": dict(kinds),
        "pct": pct,
        "representable_pct": round(100 * (kinds["tag_only"] + kinds["needs_pointer"]) / n, 2),
        "insertion_tokens": n_ins_tokens,
        "by_category": {c: dict(v) for c, v in sorted(kinds_by_cat.items())},
    }


def oracle_predictions(records: List[Dict], use_insertions: bool) -> List[Dict]:
    """Apply the gold edit program to every record -> a predictions-style list."""
    rows = []
    for r in records:
        lab = labels_for_record(r)
        rows.append({
            "id": r["id"], "vie": r["vie"], "vsl": r["vsl"],
            "category": r.get("category"), "pred": reconstruct(lab, use_insertions),
        })
    return rows


def _oracle_ceiling(records: List[Dict]) -> Dict:
    """Score gold edit programs through the real metrics (the model's ceiling)."""
    from ..metrics import compute_report

    out = {}
    for tag, use_ins in (("tag+pointer+insertion", True), ("tag+pointer only", False)):
        rows = oracle_predictions(records, use_insertions=use_ins)
        rep = compute_report([r["pred"] for r in rows], [r["vsl"] for r in rows],
                             categories=[r["category"] for r in rows], normalized=False)
        out[tag] = rep["overall"]
    return out


def run(cfg: Config, split: str = "test") -> Dict:
    splits_dir = cfg.paths.resolved("splits_dir")
    report: Dict = {"splits": {}}
    for name in ("train", "val", "test"):
        recs = read_jsonl(splits_dir / f"{name}.jsonl")
        cov = coverage(recs)
        report["splits"][name] = cov
        LOG.info("[%-5s] n=%d  tag_only=%.1f%%  needs_pointer=%.1f%%  needs_insertion=%.1f%%  (representable=%.2f%%)",
                 name, cov["n"], cov["pct"].get("tag_only", 0.0),
                 cov["pct"].get("needs_pointer", 0.0), cov["pct"].get("needs_insertion", 0.0),
                 cov["representable_pct"])

    eval_recs = read_jsonl(splits_dir / f"{split}.jsonl")
    report["oracle_ceiling_" + split] = _oracle_ceiling(eval_recs)
    LOG.info("Oracle ceiling on %s: %s", split, report["oracle_ceiling_" + split])

    out = cfg.paths.resolved("reports_dir") / "felix_coverage.json"
    write_json(out, report)
    LOG.info("Wrote coverage report -> %s", out)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="FELIX edit-label coverage / oracle ceiling")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import yaml

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))
    run(cfg, split=args.split)


if __name__ == "__main__":
    main()
