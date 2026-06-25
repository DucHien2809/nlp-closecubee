# FELIX++ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build FELIX++ as an edit-based third method for Vietnamese-to-VSL gloss generation, using the exact same train/val/test split as the existing ViT5-base and ViT5+BARTpho+MBR systems.

**Architecture:** FELIX++ extends the existing `vsl_gloss.felix` package with edit labels, insertion labels, a format head, candidate decoding, validation-tuned reranking, and a staged training loop. The final system emits the same prediction schema as existing systems and is scored through `vsl_gloss.evaluate.score_file`.

**Tech Stack:** Python 3.9+, PyTorch, HuggingFace Transformers, PyYAML, existing `vsl_gloss` utilities, pytest.

---

## File Structure

- Modify: `pyproject.toml`
  - Include `vsl_gloss.felix` in packaged modules so FELIX++ imports work after installation.
- Modify: `vsl_gloss/config.py`
  - Add `FelixPlusConfig` and `Config.felix_plus`.
- Create: `configs/felix_plus.yaml`
  - Default FELIX++ experiment config using the existing split paths.
- Create: `vsl_gloss/felix/plus_labels.py`
  - Build tag, pointer, insertion, and format labels from `(vie, vsl)`.
- Create: `vsl_gloss/felix/plus_data.py`
  - Dataset and collator for FELIX++ tensors.
- Create: `vsl_gloss/felix/plus_model.py`
  - Encoder, tag head, pointer head, insertion head, format head, and multi-task loss.
- Create: `vsl_gloss/felix/plus_decode.py`
  - Edit rendering, candidate generation, deterministic normalization, and prediction alignment checks.
- Create: `vsl_gloss/felix/rerank.py`
  - Validation-tuned candidate reranker.
- Create: `vsl_gloss/felix/plus_train.py`
  - Staged training, validation decode, best checkpoint save, final test decode, score, and leaderboard refresh.
- Modify: `modal_app.py`
  - Add a `felix_plus` Modal entrypoint.
- Create: `tests/test_felix_plus_config.py`
- Create: `tests/test_felix_plus_labels.py`
- Create: `tests/test_felix_plus_data.py`
- Create: `tests/test_felix_plus_decode.py`
- Create: `tests/test_felix_plus_rerank.py`
- Create: `tests/test_felix_plus_split_fairness.py`
- Create: `tests/test_felix_plus_train_import.py`

---

### Task 1: Config, Packaging, And FELIX++ YAML

**Files:**
- Modify: `pyproject.toml`
- Modify: `vsl_gloss/config.py`
- Create: `configs/felix_plus.yaml`
- Test: `tests/test_felix_plus_config.py`

- [ ] **Step 1: Write the failing config test**

Create `tests/test_felix_plus_config.py`:

```python
from vsl_gloss.config import Config


def test_felix_plus_config_defaults_are_available():
    cfg = Config.from_dict({"experiment_name": "felix_plus"})

    assert cfg.felix_plus.encoder_name == "xlm-roberta-base"
    assert cfg.felix_plus.max_source_length == 128
    assert cfg.felix_plus.insertion_min_count == 1
    assert cfg.felix_plus.max_candidates == 32
    assert cfg.felix_plus.selection_metric == "wer"
    assert cfg.felix_plus.category_weights["lexical"] > cfg.felix_plus.category_weights["identical"]


def test_felix_plus_yaml_overrides_nested_values():
    cfg = Config.from_dict(
        {
            "felix_plus": {
                "encoder_name": "xlm-roberta-large",
                "max_candidates": 64,
                "category_weights": {"identical": 1.0, "lexical": 6.0},
            }
        }
    )

    assert cfg.felix_plus.encoder_name == "xlm-roberta-large"
    assert cfg.felix_plus.max_candidates == 64
    assert cfg.felix_plus.category_weights["lexical"] == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_config.py -v
```

Expected: FAIL with `AttributeError: 'Config' object has no attribute 'felix_plus'`.

- [ ] **Step 3: Add `FelixPlusConfig` to `vsl_gloss/config.py`**

Insert this dataclass after `FelixConfig`:

```python
@dataclass
class FelixPlusConfig:
    """FELIX++ edit model with insertion, format repair, and reranking."""
    encoder_name: str = "xlm-roberta-base"
    max_source_length: int = 128
    pointer_dim: int = 256
    dropout: float = 0.1

    tag_loss_weight: float = 1.0
    pointer_loss_weight: float = 1.0
    insertion_loss_weight: float = 0.35
    format_loss_weight: float = 0.05

    num_train_epochs: float = 20.0
    tag_warmup_epochs: int = 2
    pointer_warmup_epochs: int = 2
    learning_rate: float = 2.0e-5
    head_learning_rate_multiplier: float = 20.0
    batch_size: int = 16
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 42

    insertion_min_count: int = 1
    max_insertion_phrase_len: int = 4
    max_candidates: int = 32
    tag_top_k: int = 2
    pointer_top_k: int = 4
    insertion_top_k: int = 3
    rerank_grid: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    selection_metric: str = "wer"
    category_weighting: bool = True
    category_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "identical": 1.0,
            "deletion_only": 1.2,
            "reorder_only": 3.0,
            "deletion_reorder": 3.0,
            "lexical": 5.0,
        }
    )
```

Then add the field to `Config`:

```python
felix_plus: FelixPlusConfig = field(default_factory=FelixPlusConfig)
```

Then add it to `Config.from_dict`:

```python
felix_plus=cls._build(FelixPlusConfig, d.get("felix_plus")),
```

- [ ] **Step 4: Update `pyproject.toml` package list**

Change the package list to include `vsl_gloss.felix`:

```toml
[tool.setuptools]
packages = [
    "vsl_gloss",
    "vsl_gloss.data",
    "vsl_gloss.baselines",
    "vsl_gloss.models",
    "vsl_gloss.felix",
]
```

- [ ] **Step 5: Create `configs/felix_plus.yaml`**

Create:

```yaml
# =====================================================================
# FELIX++ edit model: tag + pointer + insertion + format + reranking.
# Uses the existing leakage-free train/val/test split. Do not resplit.
# =====================================================================
experiment_name: felix_plus_rerank

paths:
  raw_dir: ../Parallel-Corpus-Vie-VSL-main
  processed_dir: data/processed
  splits_dir: data/splits
  output_dir: outputs
  reports_dir: reports

felix_plus:
  encoder_name: xlm-roberta-base
  max_source_length: 128
  pointer_dim: 256
  dropout: 0.1
  tag_loss_weight: 1.0
  pointer_loss_weight: 1.0
  insertion_loss_weight: 0.35
  format_loss_weight: 0.05
  num_train_epochs: 20
  tag_warmup_epochs: 2
  pointer_warmup_epochs: 2
  learning_rate: 2.0e-5
  head_learning_rate_multiplier: 20.0
  batch_size: 16
  eval_batch_size: 64
  weight_decay: 0.01
  warmup_ratio: 0.06
  max_grad_norm: 1.0
  seed: 42
  insertion_min_count: 1
  max_insertion_phrase_len: 4
  max_candidates: 32
  tag_top_k: 2
  pointer_top_k: 4
  insertion_top_k: 3
  selection_metric: wer
  category_weighting: true
  category_weights:
    identical: 1.0
    deletion_only: 1.2
    reorder_only: 3.0
    deletion_reorder: 3.0
    lexical: 5.0

eval:
  report_categories: true
  report_normalized: true
  error_examples_per_category: 15
```

- [ ] **Step 6: Run the config test**

Run:

```bash
pytest tests/test_felix_plus_config.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml vsl_gloss/config.py configs/felix_plus.yaml tests/test_felix_plus_config.py
git commit -m "feat: add FELIX++ config"
```

---

### Task 2: FELIX++ Edit Labels

**Files:**
- Create: `vsl_gloss/felix/plus_labels.py`
- Test: `tests/test_felix_plus_labels.py`

- [ ] **Step 1: Write failing label tests**

Create `tests/test_felix_plus_labels.py`:

```python
from vsl_gloss.felix.plus_labels import (
    FIRST_CASE_LOWER,
    FIRST_CASE_PRESERVE,
    FINAL_PUNCT_QUESTION,
    build_insertion_vocab,
    extract_plus_labels,
)


def test_extract_plus_labels_groups_lexical_phrase_insertions():
    labels = extract_plus_labels(
        "Nhà bạn có mấy người ?".split(),
        "Gia đình bạn người mấy ?".split(),
    )

    assert labels.tags == ["DELETE", "KEEP", "DELETE", "KEEP", "KEEP", "KEEP"]
    assert labels.order == [1, 4, 3, 5]
    assert labels.insertions == [(0, ("Gia", "đình"))]
    assert labels.format.final_punct == FINAL_PUNCT_QUESTION
    assert labels.format.first_case == FIRST_CASE_PRESERVE


def test_extract_plus_labels_handles_final_punctuation_insertion():
    labels = extract_plus_labels(
        "Tôi đeo đồng hồ".split(),
        "Tôi đeo đồng hồ .".split(),
    )

    assert labels.tags == ["KEEP", "KEEP", "KEEP", "KEEP"]
    assert labels.order == [0, 1, 2, 3]
    assert labels.insertions == [(4, (".",))]


def test_build_insertion_vocab_keeps_rare_insertions_with_min_count_one():
    records = [
        {"vie": "Tôi đeo đồng hồ", "vsl": "Tôi đeo đồng hồ ."},
        {"vie": "Nhà bạn có mấy người ?", "vsl": "Gia đình bạn người mấy ?"},
    ]

    vocab = build_insertion_vocab(records, min_count=1, max_phrase_len=4)

    assert vocab["NONE"] == 0
    assert vocab["."] > 0
    assert vocab["Gia đình"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_felix_plus_labels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.plus_labels'`.

- [ ] **Step 3: Create `vsl_gloss/felix/plus_labels.py`**

Create:

```python
"""FELIX++ label extraction: tag, pointer, insertion, and format labels."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

KEEP = "KEEP"
DELETE = "DELETE"

NONE_PHRASE = "NONE"

FINAL_PUNCT_NONE = "NONE"
FINAL_PUNCT_PERIOD = "."
FINAL_PUNCT_QUESTION = "?"
FINAL_PUNCT_EXCLAM = "!"

FIRST_CASE_PRESERVE = "preserve"
FIRST_CASE_LOWER = "lower"
FIRST_CASE_UPPER = "upper"

PUNCT = {".", "?", "!", ",", ";", ":"}


@dataclass(frozen=True)
class FormatLabels:
    final_punct: str = FINAL_PUNCT_NONE
    first_case: str = FIRST_CASE_PRESERVE
    punct_spacing: bool = True


@dataclass(frozen=True)
class PlusLabels:
    source_tokens: List[str]
    target_tokens: List[str]
    tags: List[str]
    order: List[int]
    insertions: List[Tuple[int, Tuple[str, ...]]] = field(default_factory=list)
    format: FormatLabels = field(default_factory=FormatLabels)


def _first_alpha_token(tokens: Sequence[str]) -> str:
    for tok in tokens:
        if any(ch.isalpha() for ch in tok):
            return tok
    return ""


def _first_case(src_tokens: Sequence[str], tgt_tokens: Sequence[str]) -> str:
    src = _first_alpha_token(src_tokens)
    tgt = _first_alpha_token(tgt_tokens)
    if not src or not tgt:
        return FIRST_CASE_PRESERVE
    if tgt == src:
        return FIRST_CASE_PRESERVE
    if tgt == src.lower():
        return FIRST_CASE_LOWER
    if tgt[:1] == tgt[:1].upper() and src[:1] != tgt[:1]:
        return FIRST_CASE_UPPER
    return FIRST_CASE_PRESERVE


def _final_punct(tokens: Sequence[str]) -> str:
    if not tokens:
        return FINAL_PUNCT_NONE
    last = tokens[-1]
    if last in {FINAL_PUNCT_PERIOD, FINAL_PUNCT_QUESTION, FINAL_PUNCT_EXCLAM}:
        return last
    return FINAL_PUNCT_NONE


def extract_plus_labels(src_tokens: List[str], tgt_tokens: List[str]) -> PlusLabels:
    buckets: Dict[str, deque[int]] = defaultdict(deque)
    for i, tok in enumerate(src_tokens):
        buckets[tok.lower()].append(i)

    tags = [DELETE] * len(src_tokens)
    order: List[int] = []
    raw_insertions: List[Tuple[int, str]] = []

    for tok in tgt_tokens:
        q = buckets.get(tok.lower())
        if q:
            src_i = q.popleft()
            tags[src_i] = KEEP
            order.append(src_i)
        else:
            raw_insertions.append((len(order), tok))

    grouped: List[Tuple[int, Tuple[str, ...]]] = []
    current_slot = None
    current_tokens: List[str] = []
    for slot, tok in raw_insertions:
        if current_slot is None or slot == current_slot:
            current_slot = slot
            current_tokens.append(tok)
        else:
            grouped.append((current_slot, tuple(current_tokens)))
            current_slot = slot
            current_tokens = [tok]
    if current_slot is not None:
        grouped.append((current_slot, tuple(current_tokens)))

    fmt = FormatLabels(
        final_punct=_final_punct(tgt_tokens),
        first_case=_first_case(src_tokens, tgt_tokens),
        punct_spacing=True,
    )
    return PlusLabels(src_tokens, tgt_tokens, tags, order, grouped, fmt)


def _phrase_key(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def build_insertion_vocab(
    records: Iterable[Dict],
    min_count: int = 1,
    max_phrase_len: int = 4,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for rec in records:
        labels = extract_plus_labels(rec["vie"].split(), rec["vsl"].split())
        for _, phrase in labels.insertions:
            if 0 < len(phrase) <= max_phrase_len:
                counts[_phrase_key(phrase)] += 1

    vocab = {NONE_PHRASE: 0}
    for phrase, count in sorted(counts.items()):
        if count >= min_count:
            vocab[phrase] = len(vocab)
    return vocab
```

- [ ] **Step 4: Run label tests**

Run:

```bash
pytest tests/test_felix_plus_labels.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_labels.py tests/test_felix_plus_labels.py
git commit -m "feat: add FELIX++ label extraction"
```

---

### Task 3: FELIX++ Dataset And Collator

**Files:**
- Create: `vsl_gloss/felix/plus_data.py`
- Test: `tests/test_felix_plus_data.py`

- [ ] **Step 1: Write failing data tests**

Create `tests/test_felix_plus_data.py`:

```python
from vsl_gloss.felix.plus_data import FelixPlusCollator, build_plus_example


class FakeEncoding(dict):
    def __init__(self, tokens):
        super().__init__({"input_ids": list(range(10, 10 + len(tokens)))})
        self._word_ids = list(range(len(tokens)))

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, src_tokens, is_split_into_words=True, truncation=True, max_length=128):
        return FakeEncoding(src_tokens[:max_length])


def test_build_plus_example_contains_insertion_and_format_targets():
    rec = {
        "id": "x1",
        "vie": "Nhà bạn có mấy người ?",
        "vsl": "Gia đình bạn người mấy ?",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0, "Gia đình": 1}
    format_vocab = {
        "final=NONE|case=preserve|spacing=1": 0,
        "final=?|case=preserve|spacing=1": 1,
    }

    ex = build_plus_example(rec, FakeTokenizer(), insertion_vocab, format_vocab, max_length=128)

    assert ex is not None
    assert ex["tags"] == [1, 0, 1, 0, 0, 0]
    assert ex["order"] == [1, 4, 3, 5]
    assert ex["insertion_labels"][0] == 1
    assert ex["format_label"] == 1


def test_collator_pads_plus_targets():
    rec = {
        "id": "x2",
        "vie": "Tôi đeo đồng hồ",
        "vsl": "Tôi đeo đồng hồ .",
        "category": "lexical",
    }
    insertion_vocab = {"NONE": 0, ".": 1}
    format_vocab = {
        "final=NONE|case=preserve|spacing=1": 0,
        "final=.|case=preserve|spacing=1": 1,
    }
    ex = build_plus_example(rec, FakeTokenizer(), insertion_vocab, format_vocab, max_length=128)

    batch = FelixPlusCollator(FakeTokenizer().pad_token_id)([ex])

    assert batch["input_ids"].shape[0] == 1
    assert batch["insertion_labels"].shape[1] == len(rec["vie"].split()) + 1
    assert int(batch["format_labels"][0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_felix_plus_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.plus_data'`.

- [ ] **Step 3: Create `vsl_gloss/felix/plus_data.py`**

Create the file with these public functions and classes:

```python
"""Dataset and collator for FELIX++."""
from __future__ import annotations

from typing import Dict, List, Optional

from .model import DELETE_ID, IGNORE, KEEP_ID
from .plus_labels import NONE_PHRASE, FormatLabels, build_insertion_vocab, extract_plus_labels


def encode_source(tokenizer, src_tokens: List[str], max_length: int):
    enc = tokenizer(src_tokens, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    first: Dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in first:
            first[wid] = pos
    n_covered = sum(1 for w in range(len(src_tokens)) if w in first)
    first_idx = [first.get(w, 0) for w in range(len(src_tokens))]
    return enc["input_ids"], first_idx, n_covered


def format_key(fmt: FormatLabels) -> str:
    return f"final={fmt.final_punct}|case={fmt.first_case}|spacing={1 if fmt.punct_spacing else 0}"


def default_format_vocab() -> Dict[str, int]:
    keys = [
        "final=NONE|case=preserve|spacing=1",
        "final=.|case=preserve|spacing=1",
        "final=?|case=preserve|spacing=1",
        "final=!|case=preserve|spacing=1",
        "final=NONE|case=lower|spacing=1",
        "final=.|case=lower|spacing=1",
        "final=?|case=lower|spacing=1",
        "final=!|case=lower|spacing=1",
        "final=NONE|case=upper|spacing=1",
        "final=.|case=upper|spacing=1",
        "final=?|case=upper|spacing=1",
        "final=!|case=upper|spacing=1",
    ]
    return {k: i for i, k in enumerate(keys)}


class FelixPlusExample(dict):
    pass


def build_plus_example(
    rec: Dict,
    tokenizer,
    insertion_vocab: Dict[str, int],
    format_vocab: Dict[str, int],
    max_length: int,
) -> Optional[FelixPlusExample]:
    src = rec["vie"].split()
    if not src:
        return None
    labels = extract_plus_labels(src, rec["vsl"].split())
    input_ids, first_idx, n_covered = encode_source(tokenizer, src, max_length)
    if n_covered != len(src):
        return None

    insertion_labels = [insertion_vocab[NONE_PHRASE]] * (len(labels.order) + 1)
    for slot, phrase in labels.insertions:
        phrase_key = " ".join(phrase)
        if slot < len(insertion_labels) and phrase_key in insertion_vocab:
            insertion_labels[slot] = insertion_vocab[phrase_key]

    fmt_key = format_key(labels.format)
    format_label = format_vocab.get(fmt_key, format_vocab["final=NONE|case=preserve|spacing=1"])

    return FelixPlusExample(
        id=rec.get("id"),
        vie=rec["vie"],
        vsl=rec["vsl"],
        category=rec.get("category"),
        input_ids=input_ids,
        first_idx=first_idx,
        tags=[KEEP_ID if tag == "KEEP" else DELETE_ID for tag in labels.tags],
        order=labels.order,
        insertion_labels=insertion_labels,
        format_label=format_label,
        src_tokens=src,
    )
```

Append the dataset and collator:

```python
class FelixPlusDataset:
    def __init__(self, records, tokenizer, insertion_vocab, format_vocab, max_length):
        self.examples = []
        self.skipped = 0
        for rec in records:
            ex = build_plus_example(rec, tokenizer, insertion_vocab, format_vocab, max_length)
            if ex is None:
                self.skipped += 1
            else:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class FelixPlusCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: List[FelixPlusExample]):
        import torch

        t_max = max(len(ex["input_ids"]) for ex in batch)
        w_max = max(len(ex["src_tokens"]) for ex in batch)
        slot_max = max(len(ex["insertion_labels"]) for ex in batch)
        eos_idx = w_max
        bsz = len(batch)

        input_ids = torch.full((bsz, t_max), self.pad, dtype=torch.long)
        attention_mask = torch.zeros((bsz, t_max), dtype=torch.long)
        first_subword_idx = torch.zeros((bsz, w_max), dtype=torch.long)
        word_mask = torch.zeros((bsz, w_max), dtype=torch.long)
        tag_labels = torch.full((bsz, w_max), IGNORE, dtype=torch.long)
        key_keep_mask = torch.zeros((bsz, w_max), dtype=torch.long)
        succ_target = torch.full((bsz, w_max + 1), IGNORE, dtype=torch.long)
        order_target = torch.full((bsz, w_max), IGNORE, dtype=torch.long)
        insertion_labels = torch.full((bsz, slot_max), IGNORE, dtype=torch.long)
        insertion_mask = torch.zeros((bsz, slot_max), dtype=torch.long)
        format_labels = torch.zeros((bsz,), dtype=torch.long)

        for i, ex in enumerate(batch):
            ids = ex["input_ids"]
            n_words = len(ex["src_tokens"])
            order = ex["order"]

            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
            first_subword_idx[i, :n_words] = torch.tensor(ex["first_idx"])
            word_mask[i, :n_words] = 1
            tag_labels[i, :n_words] = torch.tensor(ex["tags"])
            format_labels[i] = int(ex["format_label"])

            for j, tag_id in enumerate(ex["tags"]):
                if tag_id == KEEP_ID:
                    key_keep_mask[i, j] = 1

            pos = {src_j: t for t, src_j in enumerate(order)}
            succ_target[i, 0] = order[0] if order else eos_idx
            for src_j in range(n_words):
                if src_j in pos:
                    t = pos[src_j]
                    succ_target[i, 1 + src_j] = order[t + 1] if t + 1 < len(order) else eos_idx

            if order:
                order_target[i, : len(order)] = torch.tensor(order)

            ins = ex["insertion_labels"]
            insertion_labels[i, : len(ins)] = torch.tensor(ins)
            insertion_mask[i, : len(ins)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "first_subword_idx": first_subword_idx,
            "word_mask": word_mask,
            "tag_labels": tag_labels,
            "succ_target": succ_target,
            "key_keep_mask": key_keep_mask,
            "order_target": order_target,
            "insertion_labels": insertion_labels,
            "insertion_mask": insertion_mask,
            "format_labels": format_labels,
            "meta": [
                {
                    "id": ex["id"],
                    "vie": ex["vie"],
                    "vsl": ex["vsl"],
                    "category": ex["category"],
                    "src_tokens": ex["src_tokens"],
                }
                for ex in batch
            ],
        }
```

- [ ] **Step 4: Run data tests**

Run:

```bash
pytest tests/test_felix_plus_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_data.py tests/test_felix_plus_data.py
git commit -m "feat: add FELIX++ dataset"
```

---

### Task 4: FELIX++ Model Heads And Loss

**Files:**
- Create: `vsl_gloss/felix/plus_model.py`
- Test: extend `tests/test_felix_plus_data.py` or create `tests/test_felix_plus_model.py`

- [ ] **Step 1: Write a model shape test**

Create `tests/test_felix_plus_model.py`:

```python
import torch

from vsl_gloss.felix.plus_model import FelixPlusModel


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.emb = torch.nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.emb(input_ids)})()


def test_felix_plus_model_returns_all_losses_and_logits():
    model = FelixPlusModel(
        TinyEncoder(),
        hidden_size=8,
        pointer_dim=4,
        insertion_vocab_size=3,
        format_vocab_size=2,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "first_subword_idx": torch.tensor([[0, 1]]),
        "word_mask": torch.tensor([[1, 1]]),
        "tag_labels": torch.tensor([[0, 1]]),
        "succ_target": torch.tensor([[0, 2, -100]]),
        "key_keep_mask": torch.tensor([[1, 0]]),
        "order_target": torch.tensor([[0, -100]]),
        "insertion_labels": torch.tensor([[0, 1]]),
        "insertion_mask": torch.tensor([[1, 1]]),
        "format_labels": torch.tensor([1]),
    }

    out = model(**batch)

    assert out.loss is not None
    assert out.tag_logits.shape == (1, 2, 2)
    assert out.pointer_logits.shape == (1, 3, 3)
    assert out.insertion_logits.shape[:2] == (1, 2)
    assert out.format_logits.shape == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.plus_model'`.

- [ ] **Step 3: Create `vsl_gloss/felix/plus_model.py`**

Create public constants, output dataclass, and model class:

```python
"""FELIX++ model: tag + pointer + insertion + format heads."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DELETE_ID, IGNORE, KEEP_ID


@dataclass
class FelixPlusOutput:
    loss: Optional[torch.Tensor]
    tag_logits: torch.Tensor
    pointer_logits: torch.Tensor
    insertion_logits: torch.Tensor
    format_logits: torch.Tensor
    tag_loss: Optional[torch.Tensor] = None
    pointer_loss: Optional[torch.Tensor] = None
    insertion_loss: Optional[torch.Tensor] = None
    format_loss: Optional[torch.Tensor] = None


class FelixPlusModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        pointer_dim: int,
        insertion_vocab_size: int,
        format_vocab_size: int,
        dropout: float = 0.1,
        tag_loss_weight: float = 1.0,
        pointer_loss_weight: float = 1.0,
        insertion_loss_weight: float = 0.35,
        format_loss_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.pointer_dim = pointer_dim
        self.tag_loss_weight = tag_loss_weight
        self.pointer_loss_weight = pointer_loss_weight
        self.insertion_loss_weight = insertion_loss_weight
        self.format_loss_weight = format_loss_weight

        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden_size, 2)
        self.q_proj = nn.Linear(hidden_size, pointer_dim)
        self.k_proj = nn.Linear(hidden_size, pointer_dim)
        self.insertion_head = nn.Linear(hidden_size, insertion_vocab_size)
        self.format_head = nn.Linear(hidden_size, format_vocab_size)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.eos = nn.Parameter(torch.randn(hidden_size) * 0.02)
```

Append encoding, pointer, slot, and forward methods:

```python
    def encode_words(self, input_ids, attention_mask, first_subword_idx, word_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        idx = first_subword_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_reps = torch.gather(hidden, 1, idx)
        return self.dropout(word_reps * word_mask.unsqueeze(-1))

    def _pointer_scores(self, word_reps, key_valid):
        bsz, width, _ = word_reps.shape
        bos = self.bos.expand(bsz, 1, -1)
        eos = self.eos.expand(bsz, 1, -1)
        queries = self.q_proj(torch.cat([bos, word_reps], dim=1))
        keys = self.k_proj(torch.cat([word_reps, eos], dim=1))
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        return scores.masked_fill(~key_valid.unsqueeze(1), float("-inf"))

    def _slot_reps(self, word_reps, order_target, insertion_mask):
        bsz, slot_max = insertion_mask.shape
        reps = self.bos.expand(bsz, 1, -1).new_zeros((bsz, slot_max, self.hidden_size))
        reps[:, 0, :] = self.bos
        for i in range(bsz):
            for slot in range(1, slot_max):
                src_idx = int(order_target[i, slot - 1].item()) if slot - 1 < order_target.size(1) else IGNORE
                if src_idx != IGNORE and 0 <= src_idx < word_reps.size(1):
                    reps[i, slot, :] = word_reps[i, src_idx, :]
        return self.dropout(reps)

    def forward(
        self,
        input_ids,
        attention_mask,
        first_subword_idx,
        word_mask,
        tag_labels=None,
        succ_target=None,
        key_keep_mask=None,
        order_target=None,
        insertion_labels=None,
        insertion_mask=None,
        format_labels=None,
    ) -> FelixPlusOutput:
        word_reps = self.encode_words(input_ids, attention_mask, first_subword_idx, word_mask)
        tag_logits = self.tag_head(word_reps)

        bsz, width, _ = word_reps.shape
        eos_col = torch.ones(bsz, 1, dtype=torch.bool, device=word_reps.device)
        word_valid = word_mask.bool() if key_keep_mask is None else key_keep_mask.bool()
        key_valid = torch.cat([word_valid, eos_col], dim=1)
        pointer_logits = self._pointer_scores(word_reps, key_valid)

        if insertion_mask is None:
            insertion_mask = torch.ones(bsz, 1, dtype=torch.long, device=word_reps.device)
            order_target = torch.full((bsz, 0), IGNORE, dtype=torch.long, device=word_reps.device)
        slot_reps = self._slot_reps(word_reps, order_target, insertion_mask)
        insertion_logits = self.insertion_head(slot_reps)

        denom = word_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (word_reps * word_mask.unsqueeze(-1)).sum(dim=1) / denom
        format_logits = self.format_head(self.dropout(pooled))

        tag_loss = pointer_loss = insertion_loss = format_loss = None
        loss = None
        losses = []

        if tag_labels is not None:
            tag_loss = F.cross_entropy(tag_logits.reshape(-1, 2), tag_labels.reshape(-1), ignore_index=IGNORE)
            losses.append(self.tag_loss_weight * tag_loss)
        if succ_target is not None:
            pointer_loss = F.cross_entropy(pointer_logits.reshape(-1, width + 1), succ_target.reshape(-1), ignore_index=IGNORE)
            losses.append(self.pointer_loss_weight * pointer_loss)
        if insertion_labels is not None:
            insertion_loss = F.cross_entropy(
                insertion_logits.reshape(-1, insertion_logits.size(-1)),
                insertion_labels.reshape(-1),
                ignore_index=IGNORE,
            )
            losses.append(self.insertion_loss_weight * insertion_loss)
        if format_labels is not None:
            format_loss = F.cross_entropy(format_logits, format_labels)
            losses.append(self.format_loss_weight * format_loss)
        if losses:
            loss = sum(losses)

        return FelixPlusOutput(
            loss=loss,
            tag_logits=tag_logits,
            pointer_logits=pointer_logits,
            insertion_logits=insertion_logits,
            format_logits=format_logits,
            tag_loss=tag_loss,
            pointer_loss=pointer_loss,
            insertion_loss=insertion_loss,
            format_loss=format_loss,
        )
```

- [ ] **Step 4: Run model test**

Run:

```bash
pytest tests/test_felix_plus_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_model.py tests/test_felix_plus_model.py
git commit -m "feat: add FELIX++ model heads"
```

---

### Task 5: Edit Rendering, Normalization, And Split Fairness Checks

**Files:**
- Create: `vsl_gloss/felix/plus_decode.py`
- Test: `tests/test_felix_plus_decode.py`
- Test: `tests/test_felix_plus_split_fairness.py`

- [ ] **Step 1: Write decode tests**

Create `tests/test_felix_plus_decode.py`:

```python
from pathlib import Path

from vsl_gloss.felix.plus_decode import apply_format, render_edit, verify_prediction_alignment


def test_render_edit_inserts_phrase_before_first_kept_token():
    text = render_edit(
        src_tokens="Nhà bạn có mấy người ?".split(),
        order=[1, 4, 3, 5],
        insertions={0: ("Gia", "đình")},
        format_label="final=?|case=preserve|spacing=1",
    )

    assert text == "Gia đình bạn người mấy ?"


def test_apply_format_repairs_final_punctuation_without_forcing_case():
    assert apply_format("tôi không thể nói gì", "final=.|case=preserve|spacing=1") == "tôi không thể nói gì ."
    assert apply_format("Tôi đau đầu .", "final=NONE|case=lower|spacing=1") == "tôi đau đầu"


def test_verify_prediction_alignment_accepts_matching_ids(tmp_path: Path):
    split = tmp_path / "test.jsonl"
    pred = tmp_path / "predictions_test.jsonl"
    split.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    pred.write_text('{"id":"1","pred":"a"}\n{"id":"2","pred":"b"}\n', encoding="utf-8")

    verify_prediction_alignment(pred, split)
```

Create `tests/test_felix_plus_split_fairness.py`:

```python
from pathlib import Path

import pytest

from vsl_gloss.felix.plus_decode import verify_prediction_alignment


def test_verify_prediction_alignment_rejects_reordered_test_ids(tmp_path: Path):
    split = tmp_path / "test.jsonl"
    pred = tmp_path / "predictions_test.jsonl"
    split.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    pred.write_text('{"id":"2","pred":"b"}\n{"id":"1","pred":"a"}\n', encoding="utf-8")

    with pytest.raises(AssertionError, match="prediction ids do not match"):
        verify_prediction_alignment(pred, split)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_felix_plus_decode.py tests/test_felix_plus_split_fairness.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.plus_decode'`.

- [ ] **Step 3: Create `vsl_gloss/felix/plus_decode.py`**

Create:

```python
"""FELIX++ decoding helpers and split-fairness verification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.normalize import NormalizeOptions, normalize_text
from ..utils import read_jsonl

_NORM = NormalizeOptions(lowercase=False)


@dataclass(frozen=True)
class EditCandidate:
    text: str
    order: List[int]
    insertions: Dict[int, Tuple[str, ...]]
    format_label: str
    features: Dict[str, float]


def _parse_format(label: str) -> Dict[str, str]:
    out = {}
    for part in label.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def apply_format(text: str, format_label: str) -> str:
    fmt = _parse_format(format_label)
    tokens = normalize_text(text, _NORM).split()

    final = fmt.get("final", "NONE")
    if tokens and tokens[-1] in {".", "?", "!"}:
        tokens = tokens[:-1]
    if final != "NONE":
        tokens.append(final)

    case = fmt.get("case", "preserve")
    if case in {"lower", "upper"}:
        for i, tok in enumerate(tokens):
            if any(ch.isalpha() for ch in tok):
                tokens[i] = tok.lower() if case == "lower" else tok[:1].upper() + tok[1:]
                break

    return normalize_text(" ".join(tokens), _NORM)


def render_edit(
    src_tokens: Sequence[str],
    order: Sequence[int],
    insertions: Dict[int, Tuple[str, ...]],
    format_label: str,
) -> str:
    out: List[str] = []
    for slot in range(len(order) + 1):
        out.extend(insertions.get(slot, ()))
        if slot < len(order):
            src_i = order[slot]
            if 0 <= src_i < len(src_tokens):
                out.append(src_tokens[src_i])
    return apply_format(" ".join(out), format_label)


def verify_prediction_alignment(predictions_path: str | Path, split_path: str | Path) -> None:
    pred_rows = read_jsonl(predictions_path)
    split_rows = read_jsonl(split_path)
    pred_ids = [str(r.get("id")) for r in pred_rows]
    split_ids = [str(r.get("id")) for r in split_rows]
    assert len(pred_rows) == len(split_rows), (
        f"prediction row count {len(pred_rows)} != split row count {len(split_rows)}"
    )
    assert pred_ids == split_ids, "prediction ids do not match split ids in order"
    assert all("pred" in r for r in pred_rows), "every prediction row must include pred"
```

- [ ] **Step 4: Run decode and fairness tests**

Run:

```bash
pytest tests/test_felix_plus_decode.py tests/test_felix_plus_split_fairness.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_decode.py tests/test_felix_plus_decode.py tests/test_felix_plus_split_fairness.py
git commit -m "feat: add FELIX++ decode utilities"
```

---

### Task 6: Validation-Tuned Reranker

**Files:**
- Create: `vsl_gloss/felix/rerank.py`
- Test: `tests/test_felix_plus_rerank.py`

- [ ] **Step 1: Write reranker tests**

Create `tests/test_felix_plus_rerank.py`:

```python
from vsl_gloss.felix.plus_decode import EditCandidate
from vsl_gloss.felix.rerank import RerankWeights, choose_candidate, tune_weights


def test_choose_candidate_uses_weighted_features():
    candidates = [
        EditCandidate("A", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -2.0, "repeat": 0.0}),
        EditCandidate("B", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -0.1, "repeat": 1.0}),
    ]
    weights = RerankWeights({"model": 1.0, "repeat": -2.0})

    assert choose_candidate(candidates, weights).text == "A"


def test_tune_weights_prefers_lower_validation_wer():
    candidate_groups = [
        [
            EditCandidate("Tôi đau đầu .", [], {}, "final=.|case=preserve|spacing=1", {"model": -1.0}),
            EditCandidate("Tôi đau đầu", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -0.1}),
        ]
    ]
    refs = ["Tôi đau đầu ."]

    weights = tune_weights(candidate_groups, refs, feature_names=["model"], grid=[-1.0, 1.0])
    assert choose_candidate(candidate_groups[0], weights).text == "Tôi đau đầu ."
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_rerank.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.rerank'`.

- [ ] **Step 3: Create `vsl_gloss/felix/rerank.py`**

Create:

```python
"""Validation-tuned reranking for FELIX++ edit candidates."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

from ..metrics import compute_metrics
from .plus_decode import EditCandidate


@dataclass(frozen=True)
class RerankWeights:
    values: Dict[str, float]


def score_candidate(candidate: EditCandidate, weights: RerankWeights) -> float:
    return sum(weights.values.get(name, 0.0) * value for name, value in candidate.features.items())


def choose_candidate(candidates: Sequence[EditCandidate], weights: RerankWeights) -> EditCandidate:
    if not candidates:
        raise ValueError("choose_candidate requires at least one candidate")
    return max(candidates, key=lambda c: (score_candidate(c, weights), c.text))


def _metric_value(preds: List[str], refs: List[str], metric: str) -> float:
    m = compute_metrics(preds, refs)
    if metric == "wer":
        return -m["wer"]
    if metric == "exact_match":
        return m["exact_match"]
    return m.get(metric, m["bleu"])


def tune_weights(
    candidate_groups: Sequence[Sequence[EditCandidate]],
    refs: Sequence[str],
    feature_names: Sequence[str],
    grid: Sequence[float],
    metric: str = "wer",
) -> RerankWeights:
    if len(candidate_groups) != len(refs):
        raise ValueError("candidate_groups and refs must have the same length")

    best = RerankWeights({name: 0.0 for name in feature_names})
    best_value = float("-inf")
    for values in product(grid, repeat=len(feature_names)):
        weights = RerankWeights(dict(zip(feature_names, values)))
        preds = [choose_candidate(group, weights).text for group in candidate_groups]
        value = _metric_value(preds, list(refs), metric)
        if value > best_value:
            best = weights
            best_value = value
    return best
```

- [ ] **Step 4: Run reranker tests**

Run:

```bash
pytest tests/test_felix_plus_rerank.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vsl_gloss/felix/rerank.py tests/test_felix_plus_rerank.py
git commit -m "feat: add FELIX++ reranker"
```

---

### Task 7: Candidate Generation From FELIX++ Logits

**Files:**
- Modify: `vsl_gloss/felix/plus_decode.py`
- Test: extend `tests/test_felix_plus_decode.py`

- [ ] **Step 1: Add candidate generation test**

Append to `tests/test_felix_plus_decode.py`:

```python
from vsl_gloss.felix.plus_decode import build_candidates_from_predictions, pointer_orders_from_scores


def test_build_candidates_from_predictions_produces_non_empty_unique_candidates():
    candidates = build_candidates_from_predictions(
        src_tokens="Con gà ăn thóc .".split(),
        orders=[[0, 1, 3, 2, 4], [0, 1, 2, 3, 4]],
        insertion_options=[{}],
        format_options=["final=.|case=preserve|spacing=1"],
        max_candidates=8,
    )

    texts = [c.text for c in candidates]
    assert "Con gà thóc ăn ." in texts
    assert "Con gà ăn thóc ." in texts
    assert len(texts) == len(set(texts))


def test_pointer_orders_from_scores_returns_reordered_path():
    # rows: BOS, word0, word1, word2, word3, word4
    # cols: word0, word1, word2, word3, word4, EOS
    scores = [
        [9, 0, 0, 0, 0, -9],  # BOS -> word0
        [0, 9, 0, 0, 0, -9],  # word0 -> word1
        [0, 0, 0, 9, 0, -9],  # word1 -> word3
        [0, 0, 0, 0, 9, -9],  # word2 -> word4
        [0, 0, 9, 0, 0, -9],  # word3 -> word2
        [0, 0, 0, 0, 0, 9],   # word4 -> EOS
    ]

    orders = pointer_orders_from_scores(scores, keep_indices=[0, 1, 2, 3, 4], top_k=2)

    assert orders[0] == [0, 1, 3, 2, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_decode.py::test_build_candidates_from_predictions_produces_non_empty_unique_candidates -v
```

Expected: FAIL with `ImportError` for `build_candidates_from_predictions` or `pointer_orders_from_scores`.

- [ ] **Step 3: Add pointer path extraction to `vsl_gloss/felix/plus_decode.py`**

Append:

```python
def pointer_orders_from_scores(
    pointer_scores,
    keep_indices: Sequence[int],
    top_k: int,
) -> List[List[int]]:
    """Return up to top_k successor paths from pointer scores.

    `pointer_scores` is indexed as query rows `[BOS, w0, ...]` and key columns
    `[w0, ..., EOS]`. This function is deterministic and forbids repeated source
    indices in one path.
    """
    import heapq

    if hasattr(pointer_scores, "detach"):
        pointer_scores = pointer_scores.detach().cpu().tolist()
    if not keep_indices:
        return [[]]

    width = len(pointer_scores[0]) - 1
    eos = width
    keep = set(keep_indices)
    heap = [(0.0, 0, [], frozenset())]
    finished: List[Tuple[float, List[int]]] = []

    while heap and len(finished) < top_k:
        neg_score, query_row, path, used = heapq.heappop(heap)
        row = pointer_scores[query_row]
        ranked = sorted(range(len(row)), key=lambda j: row[j], reverse=True)
        for nxt in ranked[: max(2, top_k)]:
            if nxt == eos:
                finished.append((neg_score - float(row[nxt]), path))
                continue
            if nxt not in keep or nxt in used:
                continue
            heapq.heappush(
                heap,
                (
                    neg_score - float(row[nxt]),
                    nxt + 1,
                    path + [nxt],
                    frozenset(set(used) | {nxt}),
                ),
            )

    unique: List[List[int]] = []
    seen = set()
    for _, path in sorted(finished, key=lambda x: x[0]):
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
        if len(unique) >= top_k:
            break
    if not unique:
        unique.append(list(keep_indices))
    return unique
```

- [ ] **Step 4: Add candidate builder to `vsl_gloss/felix/plus_decode.py`**

Append:

```python
def _candidate_features(text: str, src_tokens: Sequence[str], order: Sequence[int], insertions: Dict[int, Tuple[str, ...]]) -> Dict[str, float]:
    toks = text.split()
    src_set = set(tok.lower() for tok in src_tokens)
    inserted = sum(len(v) for v in insertions.values())
    repeats = max(0, len(toks) - len(set((i, tok) for i, tok in enumerate(toks))))
    outside = sum(1 for tok in toks if tok.lower() not in src_set and tok not in {".", "?", "!", ",", ";", ":"})
    return {
        "len_ratio": len(toks) / max(1, len(src_tokens)),
        "deleted": max(0, len(src_tokens) - len(order)),
        "inserted": inserted,
        "outside_source": outside,
        "repeat": repeats,
    }


def build_candidates_from_predictions(
    src_tokens: Sequence[str],
    orders: Sequence[Sequence[int]],
    insertion_options: Sequence[Dict[int, Tuple[str, ...]]],
    format_options: Sequence[str],
    max_candidates: int,
) -> List[EditCandidate]:
    seen = set()
    out: List[EditCandidate] = []
    for order in orders:
        for insertions in insertion_options:
            for fmt in format_options:
                text = render_edit(src_tokens, order, insertions, fmt)
                if text in seen:
                    continue
                seen.add(text)
                out.append(
                    EditCandidate(
                        text=text,
                        order=list(order),
                        insertions=dict(insertions),
                        format_label=fmt,
                        features=_candidate_features(text, src_tokens, order, insertions),
                    )
                )
                if len(out) >= max_candidates:
                    return out
    return out
```

- [ ] **Step 5: Run candidate generation test**

Run:

```bash
pytest tests/test_felix_plus_decode.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_decode.py tests/test_felix_plus_decode.py
git commit -m "feat: add FELIX++ candidate builder"
```

---

### Task 8: Training Loop And Prediction Pipeline

**Files:**
- Create: `vsl_gloss/felix/plus_train.py`
- Test: no GPU unit test; use import and helper smoke tests

- [ ] **Step 1: Write import smoke test**

Create `tests/test_felix_plus_train_import.py`:

```python
from vsl_gloss.felix.plus_train import category_sample_weights, loss_weights_for_epoch, metric_is_better


def test_loss_weights_for_epoch_stages_training():
    assert loss_weights_for_epoch(epoch=1, tag_warmup_epochs=2, pointer_warmup_epochs=2) == {
        "tag": 1.0,
        "pointer": 0.0,
        "insertion": 0.0,
        "format": 0.0,
    }
    assert loss_weights_for_epoch(epoch=3, tag_warmup_epochs=2, pointer_warmup_epochs=2)["pointer"] == 1.0
    assert loss_weights_for_epoch(epoch=5, tag_warmup_epochs=2, pointer_warmup_epochs=2)["insertion"] == 1.0


def test_metric_is_better_uses_lower_wer():
    assert metric_is_better({"wer": 1.8}, {"wer": 2.0}, "wer")
    assert not metric_is_better({"wer": 2.1}, {"wer": 2.0}, "wer")
    assert metric_is_better({"bleu": 97.1}, {"bleu": 96.9}, "bleu")


def test_category_sample_weights_upweight_rare_categories():
    dataset = type(
        "Dataset",
        (),
        {"examples": [{"category": "identical"}, {"category": "lexical"}]},
    )()

    weights = category_sample_weights(dataset, {"identical": 1.0, "lexical": 5.0})

    assert weights == [1.0, 5.0]
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_train_import.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vsl_gloss.felix.plus_train'`.

- [ ] **Step 3: Create `vsl_gloss/felix/plus_train.py` helpers**

Start the file with:

```python
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
```

- [ ] **Step 4: Run smoke test**

Run:

```bash
pytest tests/test_felix_plus_train_import.py -v
```

Expected: PASS.

- [ ] **Step 5: Add full training functions to `plus_train.py`**

Append these public functions:

```python
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
```

Append the `run` skeleton with exact split usage:

```python
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
```

This skeleton references `decode_records` and `predict_split`; define them in the same file in the next step.

- [ ] **Step 6: Add decode/predict functions to `plus_train.py`**

Append:

```python
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
```

- [ ] **Step 7: Add CLI entrypoint to `plus_train.py`**

Append:

```python
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
```

- [ ] **Step 8: Run import tests**

Run:

```bash
pytest tests/test_felix_plus_train_import.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add vsl_gloss/felix/plus_train.py tests/test_felix_plus_train_import.py
git commit -m "feat: add FELIX++ training entrypoint"
```

---

### Task 9: Upgrade Decode To Candidate Reranking

**Files:**
- Modify: `vsl_gloss/felix/plus_train.py`
- Modify: `vsl_gloss/felix/plus_decode.py`
- Test: `tests/test_felix_plus_rerank.py`

- [ ] **Step 1: Add integration test for reranked candidate selection**

Append to `tests/test_felix_plus_rerank.py`:

```python
from vsl_gloss.felix.rerank import select_predictions


def test_select_predictions_applies_one_candidate_per_group():
    groups = [
        [
            EditCandidate("wrong", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -1.0}),
            EditCandidate("right", [], {}, "final=NONE|case=preserve|spacing=1", {"model": 0.0}),
        ]
    ]
    weights = RerankWeights({"model": 1.0})

    assert select_predictions(groups, weights) == ["right"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_felix_plus_rerank.py::test_select_predictions_applies_one_candidate_per_group -v
```

Expected: FAIL with `ImportError` for `select_predictions`.

- [ ] **Step 3: Add `select_predictions` to `vsl_gloss/felix/rerank.py`**

Append:

```python
def select_predictions(candidate_groups: Sequence[Sequence[EditCandidate]], weights: RerankWeights) -> List[str]:
    return [choose_candidate(group, weights).text for group in candidate_groups]
```

- [ ] **Step 4: Run reranker tests**

Run:

```bash
pytest tests/test_felix_plus_rerank.py -v
```

Expected: PASS.

- [ ] **Step 5: Modify validation flow in `plus_train.py`**

Add a function that builds candidate groups from model logits. It must use tag,
pointer, insertion, and format outputs before final experiments.

```python
def candidate_groups_for_records(records, tokenizer, model, device, cfg, insertion_vocab, format_vocab):
    import torch
    from .plus_data import FelixPlusCollator, build_plus_example
    from .plus_decode import build_candidates_from_predictions, pointer_orders_from_scores

    inv_format = {v: k for k, v in format_vocab.items()}
    inv_insert = {v: k for k, v in insertion_vocab.items()}
    collator = FelixPlusCollator(tokenizer.pad_token_id)
    fcfg = cfg.felix_plus
    out = []
    model.eval()
    with torch.no_grad():
        for rec in records:
            ex = build_plus_example(rec, tokenizer, insertion_vocab, format_vocab, fcfg.max_source_length)
            if ex is None:
                out.append([
                    EditCandidate(
                        text=rec["vie"],
                        order=[],
                        insertions={},
                        format_label="final=NONE|case=preserve|spacing=1",
                        features={"model": 0.0, "fallback": 1.0},
                    )
                ])
                continue

            batch = collator([ex])
            feats = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "first_subword_idx", "word_mask")
            }
            pred = model(**feats)
            tag_probs = pred.tag_logits.softmax(-1)[0]
            keep_indices = [
                i
                for i in range(len(ex["src_tokens"]))
                if int(tag_probs[i].argmax().item()) == 0
            ]
            if not keep_indices:
                keep_indices = list(range(len(ex["src_tokens"])))

            orders = pointer_orders_from_scores(
                pred.pointer_logits[0],
                keep_indices=keep_indices,
                top_k=fcfg.pointer_top_k,
            )

            insertion_options = [{}]
            top_insert = pred.insertion_logits.softmax(-1)[0]
            for slot in range(min(top_insert.size(0), len(orders[0]) + 1)):
                vals, ids = torch.topk(top_insert[slot], k=min(fcfg.insertion_top_k, top_insert.size(-1)))
                for prob, label_id in zip(vals.tolist(), ids.tolist()):
                    phrase = inv_insert.get(int(label_id), "NONE")
                    if phrase != "NONE":
                        insertion_options.append({slot: tuple(phrase.split())})

            fmt_vals, fmt_ids = torch.topk(
                pred.format_logits.softmax(-1)[0],
                k=min(3, pred.format_logits.size(-1)),
            )
            format_options = [inv_format.get(int(i), "final=NONE|case=preserve|spacing=1") for i in fmt_ids.tolist()]

            group = build_candidates_from_predictions(
                src_tokens=ex["src_tokens"],
                orders=orders,
                insertion_options=insertion_options,
                format_options=format_options,
                max_candidates=fcfg.max_candidates,
            )
            out.append(group)
    return out
```

Insert this block in `run()` after `best_state` is restored and before
`pred_path = predict_split(...)`. It tunes rerank weights on validation and
saves them into the already selected best checkpoint:

```python
val_groups = candidate_groups_for_records(val_records, tokenizer, model, device, cfg, insertion_vocab, format_vocab)
feature_names = sorted({name for group in val_groups for cand in group for name in cand.features})
rerank_weights = tune_weights(
    val_groups,
    [r["vsl"] for r in val_records],
    feature_names=feature_names,
    grid=fcfg.rerank_grid,
    metric=fcfg.selection_metric,
)
save_checkpoint(model, tokenizer, out_dir / "model", insertion_vocab, format_vocab, rerank_weights)
```

- [ ] **Step 6: Use saved rerank weights in `predict_split`**

Modify `predict_split` so final test predictions use reranking when
`rerank_weights.json` exists:

```python
from .rerank import RerankWeights, select_predictions


def _load_rerank_weights(model_dir: Path) -> RerankWeights:
    path = model_dir / "rerank_weights.json"
    if not path.exists():
        return RerankWeights({"model": 1.0})
    return RerankWeights({k: float(v) for k, v in read_json(path).items()})
```

Inside `predict_split`, replace:

```python
preds = decode_records(records, tokenizer, model, device, cfg, insertion_vocab, format_vocab)
```

with:

```python
groups = candidate_groups_for_records(records, tokenizer, model, device, cfg, insertion_vocab, format_vocab)
preds = select_predictions(groups, _load_rerank_weights(model_dir))
```

This keeps validation tuning and test decoding separate: `test.jsonl` receives
frozen rerank weights learned earlier from `val.jsonl`.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
pytest tests/test_felix_plus_rerank.py tests/test_felix_plus_train_import.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add vsl_gloss/felix/rerank.py vsl_gloss/felix/plus_train.py tests/test_felix_plus_rerank.py
git commit -m "feat: add FELIX++ validation reranking"
```

---

### Task 10: Modal Entrypoint And Leaderboard Integration

**Files:**
- Modify: `modal_app.py`
- Test: import smoke by running module help

- [ ] **Step 1: Add Modal training function**

In `modal_app.py`, add after `train_felix`:

```python
@app.function(gpu="A10G", timeout=60 * 60 * 6, volumes={REMOTE_ARTIFACTS: artifacts})
def train_felix_plus(config: str = "configs/felix_plus.yaml", extra_overrides=None):
    """Train FELIX++ and refresh the leaderboard using the existing split."""
    import os
    import sys

    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)

    from vsl_gloss.data import prepare, split
    from vsl_gloss.evaluate import build_leaderboard
    from vsl_gloss.felix import labels as felix_labels
    from vsl_gloss.felix import plus_train

    cfg = _load_cfg(config, extra_overrides)
    if not (cfg.paths.resolved("splits_dir") / "test.jsonl").exists():
        prepare.run(cfg)
        split.run(cfg)
    felix_labels.run(cfg, split="test")
    plus_train.run(cfg)
    artifacts.reload()
    build_leaderboard(cfg, split="test")
    artifacts.commit()
    print("Done. FELIX++ trained; leaderboard refreshed in volume 'vsl-artifacts'.")
```

- [ ] **Step 2: Add local entrypoint**

Append near the existing `felix` entrypoint:

```python
@app.local_entrypoint()
def felix_plus(config: str = "configs/felix_plus.yaml", gpu: str = "A10G", set: str = ""):
    """Train FELIX++ + refresh the leaderboard.

        modal run modal_app.py::felix_plus
        modal run modal_app.py::felix_plus --gpu A100 --set "felix_plus.encoder_name=xlm-roberta-large"
    """
    extra = set.split() if set else []
    fn = train_felix_plus.with_options(gpu=gpu) if gpu else train_felix_plus
    fn.remote(config=config, extra_overrides=extra)
```

- [ ] **Step 3: Run import/help smoke command**

Run:

```bash
python -m vsl_gloss.felix.plus_train --help
```

Expected: prints CLI help with `--config` and `--set`.

- [ ] **Step 4: Commit**

Run:

```bash
git add modal_app.py
git commit -m "feat: add FELIX++ Modal entrypoint"
```

---

### Task 11: Local Verification Suite

**Files:**
- No new source files

- [ ] **Step 1: Run all unit tests**

Run:

```bash
pytest tests/test_felix_plus_config.py tests/test_felix_plus_labels.py tests/test_felix_plus_data.py tests/test_felix_plus_model.py tests/test_felix_plus_decode.py tests/test_felix_plus_rerank.py tests/test_felix_plus_split_fairness.py tests/test_felix_plus_train_import.py -v
```

Expected: PASS.

- [ ] **Step 2: Verify package import path**

Run:

```bash
python -c "from vsl_gloss.felix import plus_labels, plus_data, plus_model, plus_decode, rerank, plus_train; print('felix_plus imports ok')"
```

Expected:

```text
felix_plus imports ok
```

- [ ] **Step 3: Verify split counts before GPU training**

Run:

```bash
python -c "from vsl_gloss.utils import read_jsonl; print(len(read_jsonl('data/splits/train.jsonl')), len(read_jsonl('data/splits/val.jsonl')), len(read_jsonl('data/splits/test.jsonl')))"
```

Expected:

```text
7586 942 949
```

- [ ] **Step 4: Commit verification-only fixes if needed**

If Step 1, Step 2, or Step 3 required fixes, commit those exact files:

```bash
git status --short
git add pyproject.toml configs/felix_plus.yaml modal_app.py vsl_gloss/config.py vsl_gloss/felix/plus_labels.py vsl_gloss/felix/plus_data.py vsl_gloss/felix/plus_model.py vsl_gloss/felix/plus_decode.py vsl_gloss/felix/rerank.py vsl_gloss/felix/plus_train.py tests/test_felix_plus_config.py tests/test_felix_plus_labels.py tests/test_felix_plus_data.py tests/test_felix_plus_model.py tests/test_felix_plus_decode.py tests/test_felix_plus_rerank.py tests/test_felix_plus_split_fairness.py tests/test_felix_plus_train_import.py
git commit -m "test: stabilize FELIX++ verification"
```

If no fixes were needed, skip this commit.

---

### Task 12: GPU Training And Experimental Runs

**Files:**
- Generated outputs only under `outputs/` and `reports/`

- [ ] **Step 1: Train default FELIX++ on Modal**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A10G
```

Expected:

- `outputs/felix_plus_rerank/model/encoder`
- `outputs/felix_plus_rerank/model/heads.pt`
- `outputs/felix_plus_rerank/model/insertion_vocab.json`
- `outputs/felix_plus_rerank/predictions_test.jsonl`
- `outputs/felix_plus_rerank/metrics_test.json`
- refreshed `reports/leaderboard_test.md`

- [ ] **Step 2: Download artifacts if run remotely**

Run:

```bash
modal volume get vsl-artifacts /outputs/felix_plus_rerank ./outputs/felix_plus_rerank
modal volume get vsl-artifacts /reports ./reports_remote
```

Expected: local folders contain the new FELIX++ outputs and reports.

- [ ] **Step 3: Verify test alignment**

Run:

```bash
python -c "from vsl_gloss.felix.plus_decode import verify_prediction_alignment; verify_prediction_alignment('outputs/felix_plus_rerank/predictions_test.jsonl', 'data/splits/test.jsonl'); print('alignment ok')"
```

Expected:

```text
alignment ok
```

- [ ] **Step 4: Re-score locally**

Run:

```bash
python -m vsl_gloss.evaluate --predictions outputs/felix_plus_rerank/predictions_test.jsonl
python -m vsl_gloss.evaluate --compare
```

Expected:

- `outputs/felix_plus_rerank/report_test.md`
- `outputs/felix_plus_rerank/errors_test.md`
- `reports/leaderboard_test.md` includes `felix_plus_rerank`

- [ ] **Step 5: Run large encoder experiment**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A100 --set "experiment_name=felix_plus_xlmr_large felix_plus.encoder_name=xlm-roberta-large felix_plus.batch_size=8"
```

Expected: `outputs/felix_plus_xlmr_large/predictions_test.jsonl` and metrics appear.

- [ ] **Step 6: Run PhoBERT experiment**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A10G --set "experiment_name=felix_plus_phobert felix_plus.encoder_name=vinai/phobert-base-v2"
```

Expected: either successful training or a clear tokenizer error if PhoBERT fast tokenizer does not provide stable `word_ids()`. If it errors, keep the error in experiment notes and use XLM-R variants for final reporting.

---

### Task 13: Ablation Runs And Report Inputs

**Files:**
- Generated outputs only under `outputs/` and `reports/`

- [ ] **Step 1: Run no-insertion ablation**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A10G --set "experiment_name=felix_plus_no_insert felix_plus.insertion_loss_weight=0.0 felix_plus.insertion_top_k=1"
```

Expected: `outputs/felix_plus_no_insert/metrics_test.json`.

- [ ] **Step 2: Run no-category-weighting ablation**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A10G --set "experiment_name=felix_plus_no_cat_weight felix_plus.category_weighting=false"
```

Expected: `outputs/felix_plus_no_cat_weight/metrics_test.json`.

- [ ] **Step 3: Run greedy-style ablation**

Run:

```bash
modal run modal_app.py::felix_plus --config configs/felix_plus.yaml --gpu A10G --set "experiment_name=felix_plus_greedy felix_plus.max_candidates=1 felix_plus.tag_top_k=1 felix_plus.pointer_top_k=1 felix_plus.insertion_top_k=1"
```

Expected: `outputs/felix_plus_greedy/metrics_test.json`.

- [ ] **Step 4: Build final leaderboard**

Run:

```bash
python -m vsl_gloss.evaluate --compare
```

Expected: `reports/leaderboard_test.md` includes baseline systems, FELIX, FELIX++ final, and FELIX++ ablations.

- [ ] **Step 5: Record headline metrics**

Open these files and copy their overall metrics into the report draft:

```text
outputs/felix_plus_rerank/metrics_test.json
outputs/felix_plus_no_insert/metrics_test.json
outputs/felix_plus_no_cat_weight/metrics_test.json
outputs/felix_plus_greedy/metrics_test.json
reports/leaderboard_test.md
```

Expected comparison statements:

- FELIX++ versus existing `felix`
- FELIX++ versus `vit5_base_15ep`
- FELIX++ versus `ensemble_mbr`
- category-specific wins or losses for `deletion_only`, `reorder_only`, `deletion_reorder`, and `lexical`

---

### Task 14: Final Verification Before Claiming Results

**Files:**
- Generated outputs only

- [ ] **Step 1: Confirm shared split**

Run:

```bash
python -c "from vsl_gloss.utils import read_jsonl; s=read_jsonl('data/splits/test.jsonl'); p=read_jsonl('outputs/felix_plus_rerank/predictions_test.jsonl'); assert [str(x['id']) for x in s] == [str(x['id']) for x in p]; print(len(s), len(p), 'same test ids')"
```

Expected:

```text
949 949 same test ids
```

- [ ] **Step 2: Confirm no tuning used test labels**

Inspect `vsl_gloss/felix/plus_train.py`:

```bash
Select-String -Path vsl_gloss/felix/plus_train.py -Pattern "test.jsonl|val.jsonl|train.jsonl|tune_weights|score_file"
```

Expected:

- `train.jsonl` used for training.
- `val.jsonl` used for validation and reranker tuning.
- `test.jsonl` used only in final `predict_split`, alignment verification, and `score_file`.

- [ ] **Step 3: Confirm final files**

Run:

```bash
Get-ChildItem outputs/felix_plus_rerank | Select-Object Name,Length
```

Expected names:

```text
model
predictions_test.jsonl
metrics_test.json
report_test.md
errors_test.md
correct_test.md
config_used.yaml
```

- [ ] **Step 4: Commit code if local implementation changes are ready**

Run:

```bash
git status --short
git add pyproject.toml configs/felix_plus.yaml modal_app.py vsl_gloss/config.py vsl_gloss/felix/plus_labels.py vsl_gloss/felix/plus_data.py vsl_gloss/felix/plus_model.py vsl_gloss/felix/plus_decode.py vsl_gloss/felix/rerank.py vsl_gloss/felix/plus_train.py tests/test_felix_plus_config.py tests/test_felix_plus_labels.py tests/test_felix_plus_data.py tests/test_felix_plus_model.py tests/test_felix_plus_decode.py tests/test_felix_plus_rerank.py tests/test_felix_plus_split_fairness.py tests/test_felix_plus_train_import.py
git commit -m "feat: implement FELIX++ edit model"
```

Expected: one final implementation commit only if earlier task commits were not already made. If earlier task commits were made, this command should report nothing to commit.

---

## Execution Notes

- The implementation must never call `vsl_gloss.data.split.run(cfg)` inside local FELIX++ training when `data/splits/test.jsonl` already exists. Modal may prepare and split only if the volume has no split files yet.
- The primary checkpoint metric is validation WER, lower is better.
- BLEU, chrF, TER, and EM are still reported for direct comparison.
- If `felix_plus_rerank` does not beat `ensemble_mbr`, preserve the result and emphasize category-level analysis. Do not tune on test to recover points.
- Pointer beam candidates, insertion alternatives, and format alternatives must be active before final GPU runs. Greedy-only decode is used only for the explicit `felix_plus_greedy` ablation.
