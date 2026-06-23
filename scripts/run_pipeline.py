"""End-to-end orchestrator (importable, no subprocess).

    python scripts/run_pipeline.py                 # data + baselines + leaderboard (CPU)
    python scripts/run_pipeline.py --train         # also fine-tune the model (needs GPU)
    python scripts/run_pipeline.py --config configs/bartpho.yaml --train

The data and baseline stages are CPU-only and fast; ``--train`` adds the neural
model + constrained decoding and is meant to run on a GPU (locally or on Modal).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script: add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from vsl_gloss.config import Config  # noqa: E402
from vsl_gloss.utils import apply_overrides, get_logger  # noqa: E402

LOG = get_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Vie->VSL gloss pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train", action="store_true", help="fine-tune the model (GPU)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    from vsl_gloss.baselines import copy_baseline, rule_based, write_predictions
    from vsl_gloss.data import prepare, split
    from vsl_gloss.evaluate import build_leaderboard

    LOG.info("[1/5] prepare corpus")
    prepare.run(cfg)
    LOG.info("[2/5] split train/val/test")
    split.run(cfg)

    LOG.info("[3/5] baselines (copy, rule)")
    write_predictions("baseline_copy", copy_baseline.predict, cfg, split=args.split)
    write_predictions("baseline_rule", rule_based.make_predict(), cfg, split=args.split)

    if args.train:
        LOG.info("[4/5] fine-tune model + constrained decoding")
        from vsl_gloss import predict as predict_mod
        from vsl_gloss import train as train_mod

        train_mod.run(cfg)
        model_dir = cfg.paths.resolved("output_dir") / cfg.experiment_name / "model"
        predict_mod.predict_split(
            cfg, str(model_dir), split=args.split,
            name=f"{cfg.experiment_name}_constrained", constrained=True, batch_size=64,
        )
    else:
        LOG.info("[4/5] skipping training (pass --train to enable)")

    LOG.info("[5/5] scoring + leaderboard")
    build_leaderboard(cfg, split=args.split)
    LOG.info("Done. See reports/ and outputs/<system>/report_%s.md", args.split)


if __name__ == "__main__":
    main()
