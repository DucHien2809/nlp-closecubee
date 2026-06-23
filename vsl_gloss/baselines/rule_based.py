"""A linguistically-motivated rule baseline over the *gold* constituency parse.

The transducer encodes the canonical VSL word-order tendencies described in the
sign-linguistics literature and the corpus guidelines:

* **Function-word deletion** -- drop a small, high-precision set of particles /
  intensifiers / coordinators (``thì``, ``rất``, ``và`` ...).
* **SVO -> SOV** -- inside a VP, post-verbal NP/AP/PP complements move in front
  of the head verb.
* **Numeral after noun** -- ``[M N]`` (e.g. ``19 tuổi``) -> ``[N M]`` (``tuổi 19``).
* **Wh-fronting reversal** -- a clause-initial interrogative subject in a
  question moves to the end (``Ai biết bơi ?`` -> ``Biết bơi ai ?``).

Each rule is independently toggleable, which doubles as a *rule ablation*. The
baseline is upper-bounded by the gold POS; where a sentence has no usable parse
it falls back to copy. Mismatches it produces (the corpus is not perfectly
consistent, e.g. ``học môn 4`` keeps SVO) are themselves evidence for why a
learned model is needed -- discussed in the error analysis.

Run::

    python -m vsl_gloss.baselines.rule_based --config configs/default.yaml --split test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import yaml

from ..config import Config
from ..data.normalize import NormalizeOptions, normalize_text, to_syllables
from ..data.parse_tree import Tree, parse_line
from ..utils import apply_overrides
from . import write_predictions

# High-precision function words to delete (surface form, case-folded).
DEFAULT_STOPWORDS = {"thì", "rất", "và", "các", "những"}
# POS labels that count as nominal heads (so a leading numeral can move after them).
_NOUN_TAGS = {"N", "Nc", "Nu", "Np", "Ny", "Nb"}
_COMPLEMENT_LABELS = {"NP", "AP", "PP"}
_PUNCT_LABELS = {".", "?", "!", "...", ";", ":"}
_WH_WORDS = {"ai", "gì", "nào", "đâu", "mấy", "sao", "bao_nhiêu"}


@dataclass(frozen=True)
class RuleOptions:
    delete_stopwords: bool = True
    svo_to_sov: bool = True
    numeral_after_noun: bool = True
    wh_to_end: bool = True
    capitalize_first: bool = True


def _is_interrogative(np: Tree) -> bool:
    for pos, word in np.leaves():
        if pos.startswith("WH") or word.lower() in _WH_WORDS:
            return True
    return False


def _reorder_vp(children: List[Tree]) -> List[Tree]:
    """Move post-verbal NP/AP/PP complements in front of the head verb (SOV)."""
    v_idx = next((i for i, c in enumerate(children) if c.label == "V"), None)
    if v_idx is None:
        return children
    pre = children[:v_idx]
    tail = children[v_idx:]
    comps = [c for c in tail if c.label in _COMPLEMENT_LABELS]
    rest = [c for c in tail if c.label not in _COMPLEMENT_LABELS]
    return pre + comps + rest


def _reorder_np(children: List[Tree]) -> List[Tree]:
    """``[M N...]`` -> ``[N... M]`` (numeral after the noun it counts)."""
    if len(children) >= 2 and children[0].label == "M":
        if any(c.label in _NOUN_TAGS for c in children[1:]):
            return children[1:] + [children[0]]
    return children


def _reorder_clause(node: Tree) -> List[Tree]:
    """In a question, move a leading interrogative subject NP to the end."""
    children = node.children
    if node.label == "SQ" and children and children[0].label == "NP" \
            and _is_interrogative(children[0]):
        first, rest = children[0], children[1:]
        punct = [c for c in rest if c.label in _PUNCT_LABELS]
        mid = [c for c in rest if c.label not in _PUNCT_LABELS]
        return mid + [first] + punct
    return children


def _reordered_children(node: Tree, opts: RuleOptions) -> List[Tree]:
    if opts.wh_to_end and node.label in {"S", "SQ", "SBAR"}:
        children = _reorder_clause(node)
    else:
        children = node.children
    if opts.svo_to_sov and node.label == "VP":
        children = _reorder_vp(children)
    if opts.numeral_after_noun and node.label == "NP":
        children = _reorder_np(children)
    return children


def _delete(node: Tree, opts: RuleOptions, stopwords) -> bool:
    return opts.delete_stopwords and node.word is not None and node.word.lower() in stopwords


def _realize(node: Tree, opts: RuleOptions, stopwords) -> List[str]:
    if node.is_preterminal:
        return [] if _delete(node, opts, stopwords) else [node.word]
    out: List[str] = []
    for c in _reordered_children(node, opts):
        out.extend(_realize(c, opts, stopwords))
    return out


def _capitalize_first(tokens: List[str]) -> List[str]:
    for i, t in enumerate(tokens):
        if any(ch.isalpha() for ch in t):
            tokens[i] = t[:1].upper() + t[1:]
            break
    return tokens


def transduce(tree: Tree, opts: RuleOptions = RuleOptions(),
              stopwords=DEFAULT_STOPWORDS) -> str:
    """Tree -> VSL-gloss string (syllable tokens, normalised)."""
    words = _realize(tree, opts, stopwords)
    toks: List[str] = []
    for w in words:
        toks.extend(to_syllables(w).split())
    if opts.capitalize_first:
        toks = _capitalize_first(toks)
    return normalize_text(" ".join(toks), NormalizeOptions(lowercase=False))


def make_predict(opts: RuleOptions = RuleOptions()):
    def predict(record: Dict) -> str:
        tree = parse_line(record.get("vie_tree", "")) if record.get("vie_tree") else None
        if tree is None:
            return record["vie"]  # graceful fall-back to copy
        return transduce(tree, opts)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold-parse rule baseline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    # Rule toggles (for the rule ablation).
    ap.add_argument("--no-delete", action="store_true")
    ap.add_argument("--no-sov", action="store_true")
    ap.add_argument("--no-numeral", action="store_true")
    ap.add_argument("--no-wh", action="store_true")
    ap.add_argument("--name", default="baseline_rule")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    opts = RuleOptions(
        delete_stopwords=not args.no_delete,
        svo_to_sov=not args.no_sov,
        numeral_after_noun=not args.no_numeral,
        wh_to_end=not args.no_wh,
    )
    write_predictions(args.name, make_predict(opts), cfg, split=args.split)


if __name__ == "__main__":
    main()
