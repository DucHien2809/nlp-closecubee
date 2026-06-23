"""Central, typed configuration for the VSL text-to-gloss pipeline.

A single YAML file (see ``configs/default.yaml``) is parsed into nested
dataclasses so the rest of the codebase consumes attributes, not dict keys.
All filesystem paths are resolved relative to the repository root, which makes
the project runnable unchanged on a laptop and on a remote GPU (Modal) where
the working directory differs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repository root = parent of the ``vsl_gloss`` package directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path) -> Path:
    """Resolve ``p`` against the repo root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass
class PathsConfig:
    # Folder holding the original corpus (Vie10k.txt, VSL10k.txt, *_phantich.txt).
    raw_dir: str = "../Parallel-Corpus-Vie-VSL-main"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    output_dir: str = "outputs"
    reports_dir: str = "reports"

    def resolved(self, name: str) -> Path:
        return _resolve(getattr(self, name))


@dataclass
class DataConfig:
    # Source filenames inside ``paths.raw_dir``.
    vie_file: str = "Vie10k.txt"
    vsl_file: str = "VSL10k.txt"
    vie_tree_file: str = "Vie10k_phantich.txt"
    vsl_tree_file: str = "VSL10k_phantich.txt"

    # Cleaning / normalisation.
    lowercase: bool = False          # keep original casing for training; eval also reports lower-cased.
    fix_punct_spacing: bool = True   # ensure tokens are whitespace-separated around .?!,
    collapse_whitespace: bool = True
    drop_empty: bool = True

    # Splitting.
    seed: int = 42
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    # Group identical source sentences into the same split (prevents train/test leakage).
    group_by_source: bool = True
    # Stratify the split by transformation category (copy / reorder / deletion / ...).
    stratify_by_category: bool = True
    dedup_exact_pairs: bool = True


@dataclass
class ModelConfig:
    # HuggingFace id or local path. ViT5-base is the project default; BARTpho is a drop-in alt.
    name_or_path: str = "VietAI/vit5-base"
    # Task prefix prepended to the source. MUST match between train & inference.
    source_prefix: str = "chuyen gloss: "
    max_source_length: int = 96
    max_target_length: int = 96
    use_fast_tokenizer: bool = False  # ViT5 ships a sentencepiece model; slow tokenizer is safest.


@dataclass
class TrainConfig:
    num_train_epochs: float = 15.0
    learning_rate: float = 3.0e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing_factor: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adafactor"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    # ViT5/T5 share weights -> some tensors are non-contiguous, which the
    # safetensors format refuses to save. Use the .bin serializer instead.
    save_safetensors: bool = False
    logging_steps: int = 50
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    predict_with_generate: bool = True
    early_stopping_patience: int = 3
    dataloader_num_workers: int = 2
    seed: int = 42


@dataclass
class DecodeConfig:
    num_beams: int = 5
    max_new_tokens: int = 96
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True
    # Restrict generation to the source's sub-word vocabulary + an allow-list
    # (VSL gloss is ~99% a re-ordered subset of the source). See models/constrained_decoding.py.
    constrained: bool = False


@dataclass
class FelixConfig:
    """The FELIX edit model (tagging + pointer reordering). A second, non-seq2seq
    system; only consumed by ``vsl_gloss.felix.*``."""
    encoder_name: str = "xlm-roberta-base"   # fast tokenizer (word_ids) + multilingual
    max_source_length: int = 128             # sub-word budget (must cover every word)
    pointer_dim: int = 256
    dropout: float = 0.1
    pointer_loss_weight: float = 1.0
    num_train_epochs: float = 10.0
    learning_rate: float = 2.0e-5
    batch_size: int = 16
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 42
    # Capitalise the first output token (gloss sentences start upper-cased in the
    # corpus); other tokens keep their source surface. Lifts the case-noise ceiling.
    truecase: bool = True


@dataclass
class EvalConfig:
    # Report metrics broken down by these transformation categories.
    report_categories: bool = True
    # Also report a lower-cased / punctuation-normalised variant of every metric.
    report_normalized: bool = True
    # Number of qualitative error examples to dump per category.
    error_examples_per_category: int = 15


@dataclass
class Config:
    experiment_name: str = "vit5_base_baseline"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    felix: FelixConfig = field(default_factory=FelixConfig)

    # ---- (de)serialisation helpers ------------------------------------------------
    @staticmethod
    def _build(cls, data: Optional[Dict[str, Any]]):
        if not data:
            return cls()
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d or {})
        return cls(
            experiment_name=d.get("experiment_name", "vit5_base_baseline"),
            paths=cls._build(PathsConfig, d.get("paths")),
            data=cls._build(DataConfig, d.get("data")),
            model=cls._build(ModelConfig, d.get("model")),
            train=cls._build(TrainConfig, d.get("train")),
            decode=cls._build(DecodeConfig, d.get("decode")),
            eval=cls._build(EvalConfig, d.get("eval")),
            felix=cls._build(FelixConfig, d.get("felix")),
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
