"""Run the whole pipeline on a Modal GPU.

Prerequisite (one-time, after you get your token):

    pip install modal
    modal token set --token-id <ID> --token-secret <SECRET>

Then, from this directory:

    # full pipeline: prepare -> split -> baselines -> train -> constrained -> leaderboard
    modal run modal_app.py

    # pick a config / GPU / extra overrides
    modal run modal_app.py --config configs/bartpho.yaml --gpu A100
    modal run modal_app.py --set "train.num_train_epochs=20"

Artifacts (trained model, predictions, reports) are written to a persistent
Modal Volume named ``vsl-artifacts``; download them with::

    modal volume get vsl-artifacts /reports ./reports_remote
    modal volume get vsl-artifacts /outputs ./outputs_remote
"""
from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
RAW_DATA = HERE.parent / "Parallel-Corpus-Vie-VSL-main"

REMOTE_APP = "/root/app"
REMOTE_RAW = "/root/Parallel-Corpus-Vie-VSL-main"
REMOTE_ARTIFACTS = "/artifacts"

# Absolute artifact paths so config resolution ignores the repo-relative defaults.
ARTIFACT_OVERRIDES = [
    f"paths.raw_dir={REMOTE_RAW}",
    f"paths.processed_dir={REMOTE_ARTIFACTS}/data/processed",
    f"paths.splits_dir={REMOTE_ARTIFACTS}/data/splits",
    f"paths.output_dir={REMOTE_ARTIFACTS}/outputs",
    f"paths.reports_dir={REMOTE_ARTIFACTS}/reports",
]

# Only the 4 line-aligned .txt files are needed remotely; skip the heavy
# archives / Access DB / Excel / build artefacts so the upload stays tiny.
_RAW_IGNORE = [
    "*.rar", "*.xlsx", "*.accdb", "*.exe", "*.pdb", "*.suo", "~$*",
    "VSL-Lexicon-extracted", "VSL-Lexicon-extracted/**",
]
_CODE_IGNORE = ["outputs", "outputs/**", "data", "data/**", "reports", "reports/**",
                ".git", ".git/**", "**/__pycache__", "**/__pycache__/**"]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .add_local_dir(str(HERE), REMOTE_APP, ignore=_CODE_IGNORE)
    .add_local_dir(str(RAW_DATA), REMOTE_RAW, ignore=_RAW_IGNORE)
)

app = modal.App("vsl-text2gloss", image=image)
artifacts = modal.Volume.from_name("vsl-artifacts", create_if_missing=True)


def _load_cfg(config: str, extra_overrides):
    """Build a Config from a yaml file + artifact + user overrides."""
    import sys

    sys.path.insert(0, REMOTE_APP)
    import yaml

    from vsl_gloss.config import Config
    from vsl_gloss.utils import apply_overrides

    with open(f"{REMOTE_APP}/{config}", "r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh) or {}
    apply_overrides(cfg_dict, list(ARTIFACT_OVERRIDES) + list(extra_overrides or []))
    return Config.from_dict(cfg_dict)


@app.function(gpu="A100", timeout=60 * 60 * 5, volumes={REMOTE_ARTIFACTS: artifacts})
def run_all(config: str = "configs/default.yaml", extra_overrides=None):
    """prepare -> split -> baselines -> train -> constrained decode -> leaderboard."""
    import os
    import sys

    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)

    import torch

    from vsl_gloss.baselines import copy_baseline, rule_based, write_predictions
    from vsl_gloss.data import prepare, split
    from vsl_gloss.evaluate import build_leaderboard
    from vsl_gloss import predict as predict_mod
    from vsl_gloss import train as train_mod

    print("CUDA available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    cfg = _load_cfg(config, extra_overrides)

    # 1-2. data
    prepare.run(cfg)
    split.run(cfg)

    # 3. non-neural baselines
    write_predictions("baseline_copy", copy_baseline.predict, cfg, split="test")
    write_predictions("baseline_rule", rule_based.make_predict(), cfg, split="test")

    # 4. train the main model (also writes + scores test predictions)
    train_mod.run(cfg)

    # 5. improvement: constrained decoding from the trained checkpoint
    model_dir = f"{cfg.paths.resolved('output_dir')}/{cfg.experiment_name}/model"
    predict_mod.predict_split(
        cfg, model_dir, split="test",
        name=f"{cfg.experiment_name}_constrained", constrained=True, batch_size=64,
    )

    # 6. score everything + leaderboard
    artifacts.reload()   # pick up predictions any concurrent job committed
    build_leaderboard(cfg, split="test")
    artifacts.commit()
    print("Done. Artifacts committed to volume 'vsl-artifacts'.")


@app.function(gpu="A10G", timeout=60 * 60 * 5, volumes={REMOTE_ARTIFACTS: artifacts})
def train_only(config: str = "configs/default.yaml", extra_overrides=None):
    """Train a single seq2seq backbone (also writes + scores test predictions),
    then refresh the leaderboard. Used to add e.g. BARTpho without re-running the
    baselines. Idempotently ensures the leakage-free splits exist first."""
    import os
    import sys

    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)
    from vsl_gloss.data import prepare, split
    from vsl_gloss.evaluate import build_leaderboard
    from vsl_gloss import train as train_mod

    cfg = _load_cfg(config, extra_overrides)
    if not (cfg.paths.resolved("splits_dir") / "test.jsonl").exists():
        prepare.run(cfg)
        split.run(cfg)
    train_mod.run(cfg)
    artifacts.reload()
    build_leaderboard(cfg, split="test")
    artifacts.commit()
    print(f"Done. Trained {cfg.experiment_name}; leaderboard refreshed.")


@app.function(gpu="A10G", timeout=60 * 60 * 2, volumes={REMOTE_ARTIFACTS: artifacts})
def run_ensemble(config: str = "configs/ensemble_mbr.yaml", extra_overrides=None,
                 split: str = "test"):
    """MBR-ensemble already-trained members (no training) + refresh leaderboard.

    Requires every ``ensemble.members`` entry to already exist as
    ``outputs/<name>/model`` in the volume.
    """
    import os
    import sys

    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)
    from vsl_gloss.data import prepare, split as split_mod
    from vsl_gloss import ensemble as ensemble_mod
    from vsl_gloss.evaluate import build_leaderboard

    cfg = _load_cfg(config, extra_overrides)
    artifacts.reload()   # see members/splits committed by earlier jobs
    missing = [m for m in cfg.ensemble.members
               if not (cfg.paths.resolved("output_dir") / m / "model").exists()]
    if missing:
        raise FileNotFoundError(
            f"Ensemble members not found in volume: {missing}. Train them first "
            f"(e.g. modal run modal_app.py::train --config configs/bartpho.yaml)."
        )
    if not (cfg.paths.resolved("splits_dir") / f"{split}.jsonl").exists():
        prepare.run(cfg)
        split_mod.run(cfg)
    ensemble_mod.run(cfg, split=split)
    artifacts.reload()
    build_leaderboard(cfg, split=split)
    artifacts.commit()
    print(f"Done. MBR ensemble '{cfg.ensemble.name}' written; leaderboard refreshed.")


@app.function(gpu="A10G", timeout=60 * 60 * 4, volumes={REMOTE_ARTIFACTS: artifacts})
def train_felix(config: str = "configs/felix.yaml", extra_overrides=None):
    """Train the FELIX edit model (tag + pointer) and refresh the leaderboard.

    Idempotently ensures the leakage-free splits exist, trains FELIX, writes
    ``outputs/felix/predictions_test.jsonl``, then rebuilds the leaderboard so
    FELIX is ranked alongside the baselines / ViT5 already in the volume.
    """
    import os
    import sys

    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)

    from vsl_gloss.data import prepare, split
    from vsl_gloss.evaluate import build_leaderboard
    from vsl_gloss.felix import labels as felix_labels
    from vsl_gloss.felix import train as felix_train

    cfg = _load_cfg(config, extra_overrides)
    if not (cfg.paths.resolved("splits_dir") / "test.jsonl").exists():
        prepare.run(cfg)
        split.run(cfg)
    felix_labels.run(cfg, split="test")     # coverage + oracle ceiling report
    felix_train.run(cfg)
    artifacts.reload()   # pick up predictions any concurrent job committed
    build_leaderboard(cfg, split="test")
    artifacts.commit()
    print("Done. FELIX trained; leaderboard refreshed in volume 'vsl-artifacts'.")


@app.local_entrypoint()
def main(config: str = "configs/vit5_large.yaml", gpu: str = "A100", set: str = ""):
    """Kick off the full pipeline.

    ``--gpu`` overrides the accelerator at call time (e.g. ``A10G`` for the
    cheaper ViT5-base run); ``--set`` takes space-separated dotted overrides.
    """
    extra = set.split() if set else []
    fn = run_all.with_options(gpu=gpu) if gpu else run_all
    fn.remote(config=config, extra_overrides=extra)


@app.local_entrypoint()
def train(config: str = "configs/bartpho.yaml", gpu: str = "A10G", set: str = ""):
    """Train a single backbone (default: BARTpho) + refresh the leaderboard.

        modal run modal_app.py::train --config configs/bartpho.yaml
    """
    extra = set.split() if set else []
    fn = train_only.with_options(gpu=gpu) if gpu else train_only
    fn.remote(config=config, extra_overrides=extra)


@app.local_entrypoint()
def ensemble(config: str = "configs/ensemble_mbr.yaml", gpu: str = "A10G",
             split: str = "test", set: str = ""):
    """MBR-ensemble already-trained members + refresh the leaderboard.

        modal run modal_app.py::ensemble
        modal run modal_app.py::ensemble --set "ensemble.num_samples=32"
    """
    extra = set.split() if set else []
    fn = run_ensemble.with_options(gpu=gpu) if gpu else run_ensemble
    fn.remote(config=config, extra_overrides=extra, split=split)


@app.local_entrypoint()
def felix(config: str = "configs/felix.yaml", gpu: str = "A10G", set: str = ""):
    """Train the FELIX edit model + refresh the leaderboard.

        modal run modal_app.py::felix
        modal run modal_app.py::felix --set "felix.num_train_epochs=15"
    """
    extra = set.split() if set else []
    fn = train_felix.with_options(gpu=gpu) if gpu else train_felix
    fn.remote(config=config, extra_overrides=extra)
