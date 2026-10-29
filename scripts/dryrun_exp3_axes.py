"""Dry run of the joint Experiment 3 analysis (main confirmation + architecture and dose axes).

Synthetic block records are generated for every configuration from its own effect-blind pilot
dispersions, under declared truths per configuration, validated with full record checks, analyzed
with ``analyze_2d`` per configuration and then jointly with ``analyze_exp3_axes``. Reports the
probability of each joint label and the primary claim, so the registration documents what the
frozen joint code decides.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_2d import analyze_2d  # noqa: E402
from analyze_confirmation import validate_records  # noqa: E402
from analyze_exp3_axes import analyze_axes  # noqa: E402
from dryrun_2d import dispersion_from, synthetic_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AXIS_CONFIGS = {
    "latent_dinov2": (
        "experiments/exposure2d_confirm_latent_dinov2.json",
        "results/latent_dinov2_pilot_v1_blinded.json",
    ),
    "latent_vjepa2": (
        "experiments/exposure2d_confirm_latent_vjepa2.json",
        "results/latent_vjepa2_pilot_v1_blinded.json",
    ),
    "dose16": (
        "experiments/exposure2d_confirm_dose16.json",
        "results/dose16_pilot_v1_blinded.json",
    ),
    "dose256": (
        "experiments/exposure2d_confirm_dose256.json",
        "results/dose256_pilot_v1_blinded.json",
    ),
}
BASE = dict(H=0.0, Q=0.0, E=0.004)
# truths: S (= G when H = 0) per configuration, in arena widths
SCENARIOS = {
    "all_null": dict(main=0.0, latent_dinov2=0.0, latent_vjepa2=0.0, dose16=0.0, dose256=0.0),
    "main_consequential_recipes_detectable_dose_rising": dict(
        main=0.015, latent_dinov2=0.004, latent_vjepa2=0.004, dose16=0.015, dose256=0.004
    ),
    "main_consequential_recipes_null_dose_flat": dict(
        main=0.015, latent_dinov2=0.0, latent_vjepa2=0.0, dose16=0.015, dose256=0.015
    ),
    "recipes_tiny_0.001": dict(
        main=0.006, latent_dinov2=0.001, latent_vjepa2=0.001, dose16=0.006, dose256=0.006
    ),
    "dose_rising_from_null_low_dose": dict(
        main=0.006, latent_dinov2=0.003, latent_vjepa2=0.003, dose16=0.010, dose256=0.0
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=120)
    parser.add_argument("--sd-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main_config = json.loads((ROOT / "experiments/exposure2d_confirmation.json").read_text())
    main_sds = {
        k: v * args.sd_scale
        for k, v in dispersion_from(ROOT / "results/exposure2d_pilot_v1_blinded.json").items()
    }
    axes_cfg = {
        name: json.loads((ROOT / cfg).read_text()) for name, (cfg, _) in AXIS_CONFIGS.items()
    }
    axes_sds = {
        name: {k: v * args.sd_scale for k, v in dispersion_from(ROOT / rep).items()}
        for name, (_, rep) in AXIS_CONFIGS.items()
    }
    rng = np.random.default_rng(20261028)
    out = {"sd_scale": args.sd_scale, "replications": args.replications, "results": {}}
    for scen, truth in SCENARIOS.items():
        labels: dict[str, Counter] = {
            k: Counter() for k in ("latent_dinov2", "latent_vjepa2", "dose")
        }
        claims = Counter()
        capacity = Counter()
        for _ in range(args.replications):
            main_runs = synthetic_runs(
                rng,
                main_config,
                dict(BASE, S=truth["main"]),
                main_sds,
                int(main_config["blocks"]),
                0.97,
            )
            main_runs = validate_records(main_runs, int(main_config["blocks"]), main_config)
            main_result = analyze_2d(main_runs, main_config, check_records=False)
            axes = {}
            for name, cfg in axes_cfg.items():
                runs = synthetic_runs(
                    rng, cfg, dict(BASE, S=truth[name]), axes_sds[name], int(cfg["blocks"]), 0.97
                )
                axes[name] = (validate_records(runs, int(cfg["blocks"]), cfg), cfg)
            joint = analyze_axes({"result": main_result, "config": main_config}, axes)
            claims[joint["main_primary_claim"]] += 1
            for k in labels:
                labels[k][joint["labels"][k]] += 1
            capacity[joint["main_labels_under_joint_holm"].get("capacity", "n/a")] += 1
        out["results"][scen] = {
            "truth": truth,
            "main_primary_claim": {k: v / args.replications for k, v in claims.items()},
            "labels": {
                k: {kk: vv / args.replications for kk, vv in c.items()} for k, c in labels.items()
            },
            "capacity_label_under_joint_holm": {
                k: v / args.replications for k, v in capacity.items()
            },
        }
        print(scen, {k: dict(c.most_common(2)) for k, c in labels.items()}, flush=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
