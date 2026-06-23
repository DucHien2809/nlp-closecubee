"""Text normalisation for Vietnamese / VSL-gloss token strings.

The corpus is already whitespace-tokenised at the *syllable* level
(e.g. ``"Tôi 19 tuổi ."``), but a small fraction of lines carry noise that
hurts both training and evaluation (punctuation glued to a word: ``"ai?"``,
``".Bạn"``; doubled spaces). We repair that here. The same normaliser is used
for the source, the target, and every system prediction so that all metrics are
computed on identically-tokenised strings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

# Punctuation that should always stand as its own whitespace-separated token.
_PUNCT_NODOT = re.compile(r"\s*([?!,;:])\s*")
# A period that is NOT part of a decimal number (so "3.5" stays intact).
_PERIOD = re.compile(r"(?<![0-9])\.(?![0-9])")
_MULTISPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizeOptions:
    fix_punct_spacing: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    unicode_nfc: bool = True


def normalize_text(text: str, opts: NormalizeOptions = NormalizeOptions()) -> str:
    """Return a cleaned, whitespace-tokenised string."""
    if text is None:
        return ""
    s = text.strip()
    if opts.unicode_nfc:
        # Vietnamese diacritics have several Unicode encodings; pin to NFC.
        s = unicodedata.normalize("NFC", s)
    if opts.fix_punct_spacing:
        s = _PUNCT_NODOT.sub(r" \1 ", s)
        s = _PERIOD.sub(" . ", s)
    if opts.collapse_whitespace:
        s = _MULTISPACE.sub(" ", s).strip()
    if opts.lowercase:
        s = s.lower()
    return s


def tokenize(text: str) -> List[str]:
    """Whitespace tokenisation (the corpus convention). Empty -> []."""
    return text.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def to_syllables(word: str) -> str:
    """Tree words are segmented with underscores (``Bây_giờ``); the gloss
    convention is syllable-level, so expand them back to spaces."""
    return word.replace("_", " ")
