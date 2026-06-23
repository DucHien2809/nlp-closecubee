"""A minimal reader for the Penn-Treebank-style constituency parses shipped
with the corpus (``*_phantich.txt``).

Each line looks like::

    ( (S (NP (P Tôi)) (NP (M 19) (N tuổi)) (. .)) )

These parses are gold and 100%-aligned (by line index) with the raw corpus.
They give us, for free: word segmentation (multi-syllable words use an
underscore, e.g. ``Bây_giờ``), POS tags (VietTreebank tagset) and phrase
structure. We exploit all three in the rule-based baseline and the error
analysis. The reader is intentionally dependency-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .normalize import to_syllables

_TOK = re.compile(r"\(|\)|[^()\s]+")


@dataclass
class Tree:
    """A constituency node.

    * Internal node: ``label`` set, ``children`` populated, ``word is None``.
    * Pre-terminal (POS): ``label`` is the POS tag, ``word`` is the surface
      form, ``children`` empty.
    """

    label: str
    children: List["Tree"] = field(default_factory=list)
    word: Optional[str] = None

    @property
    def is_preterminal(self) -> bool:
        return self.word is not None

    # -- terminal yields --------------------------------------------------------
    def leaves(self) -> List[Tuple[str, str]]:
        """Return ``(pos, word)`` pairs in left-to-right order."""
        if self.is_preterminal:
            return [(self.label, self.word)]
        out: List[Tuple[str, str]] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def words(self, syllables: bool = False) -> List[str]:
        ws = [w for _, w in self.leaves()]
        return [to_syllables(w) for w in ws] if syllables else ws

    def pos_tags(self) -> List[str]:
        return [p for p, _ in self.leaves()]

    def yield_text(self, syllables: bool = True) -> str:
        """Surface string of the leaves; ``syllables`` expands ``_`` to spaces."""
        toks: List[str] = []
        for _, w in self.leaves():
            w = to_syllables(w) if syllables else w
            toks.extend(w.split())
        return " ".join(toks)


def _parse_tokens(tokens: List[str], i: int) -> Tuple[Tree, int]:
    """Recursive-descent parse starting at ``tokens[i] == '('``."""
    assert tokens[i] == "(", f"expected '(' at {i}, got {tokens[i]!r}"
    i += 1
    # An empty-label node (the implicit ROOT) is immediately followed by '('.
    if tokens[i] in ("(", ")"):
        label = ""
    else:
        label = tokens[i]
        i += 1
    children: List[Tree] = []
    word: Optional[str] = None
    while tokens[i] != ")":
        if tokens[i] == "(":
            child, i = _parse_tokens(tokens, i)
            children.append(child)
        else:
            # A bare atom following a label is the surface word of a pre-terminal.
            word = tokens[i]
            i += 1
    i += 1  # consume ')'
    return Tree(label=label, children=children, word=word), i


def parse_line(line: str) -> Optional[Tree]:
    """Parse one bracketed line into a :class:`Tree`. ``None`` if blank.

    The corpus wraps every sentence as ``( (S ...) )`` — an empty-label outer
    node with a single real child, which we unwrap.
    """
    line = line.strip()
    if not line:
        return None
    tokens = _TOK.findall(line)
    tree, _ = _parse_tokens(tokens, 0)
    if tree.label == "" and tree.word is None and len(tree.children) == 1:
        return tree.children[0]
    return tree


def load_trees(path: str) -> List[Optional[Tree]]:
    """Read a ``*_phantich.txt`` file into a list of trees (index-aligned)."""
    from ..utils import read_lines

    return [parse_line(ln) for ln in read_lines(path)]
