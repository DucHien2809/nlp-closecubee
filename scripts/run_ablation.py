"""Rule-baseline ablation: measure each hand-written rule's contribution.

Produces one prediction system per rule configuration so they all appear on the
leaderboard next to copy and the neural model. This is the *non-neural* half of
the ablation study; the neural half (unconstrained vs constrained decoding,
ViT5 vs BARTpho) comes from running the corresponding configs.

    python scripts/run_ablation.py            # then: python -m vsl_gloss.evaluate --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from vsl_gloss.baselines import write_predictions  # noqa: E402
from vsl_gloss.baselines.rule_based import RuleOptions, make_predict  # noqa: E402
from vsl_gloss.config import Config  # noqa: E402
from vsl_gloss.utils import apply_overrides, get_logger  # noqa: E402

LOG = get_logger("ablation")

# name -> RuleOptions. Each row turns exactly one capability on/off vs the full set.
VARIANTS = {
    "rule_full": RuleOptions(),
    "rule_delete_only": RuleOptions(svo_to_sov=False, numeral_after_noun=False, wh_to_end=False),
    "rule_no_delete": RuleOptions(delete_stopwords=False),
    "rule_no_sov": RuleOptions(svo_to_sov=False),
    "rule_no_numeral": RuleOptions(numeral_after_noun=False),
    "rule_no_wh": RuleOptions(wh_to_end=False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-baseline ablation")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = Config.from_dict(apply_overrides(yaml.safe_load(fh) or {}, args.set))

    for name, opts in VARIANTS.items():
        write_predictions(name, make_predict(opts), cfg, split=args.split)
        LOG.info("wrote %s", name)
    LOG.info("Now run: python -m vsl_gloss.evaluate --compare --split %s", args.split)


if __name__ == "__main__":
    main()
