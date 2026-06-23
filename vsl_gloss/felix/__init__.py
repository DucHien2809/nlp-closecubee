"""FELIX-style edit model for Vie -> VSL gloss (tagging + pointer reordering).

A second, non-seq2seq system that models the transduction the way the corpus
analysis says it actually works: **delete function words** (a KEEP/DELETE tag per
source token) and **reorder the survivors** (a pointer network over the kept
tokens). ~98% of the corpus is a re-ordered *subset* of the source, so this
inductive bias is a near-exact fit, and the model is faster and more interpretable
than the seq2seq backbone.

Reference: Mallinson et al., *FELIX: Flexible Text Editing Through Tagging and
Insertion*, Findings of EMNLP 2020. (We keep the tag + pointer-reorder core; the
rare ~2% genuinely-lexical insertions are reported as the residual ceiling rather
than modelled with a masked-LM insertion head in this first version.)
"""
