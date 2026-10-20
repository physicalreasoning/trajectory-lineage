"""Interim exploratory run: an independent small sample with the confirmation's exact protocol.

Writes raw block records directly because this sample is not the confirmation (different
experiment seed, role ``interim_exploratory``). The confirmation's freeze gate is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from exposure_experiment import provenance, run_with_policy, spec_from  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/exposure_interim.json"))
    parser.add_argument("--out", type=Path, default=Path("results/exposure_interim_raw.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("role") != "interim_exploratory":
        raise SystemExit("interim_run.py only runs interim_exploratory configurations")
    confirmation = json.loads(Path("experiments/exposure_confirmation.json").read_text())
    if int(config["experiment_seed"]) == int(confirmation["experiment_seed"]):
        raise SystemExit("the interim sample must not reuse the confirmation seed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    spec = spec_from(config)
    runs = []
    for block_index in range(int(config["blocks"])):
        runs.append(run_with_policy(config, block_index, spec, device))
        print(
            f"interim block {block_index + 1}/{config['blocks']}: {runs[-1]['status']}", flush=True
        )
    export = {"provenance": provenance(device), "config": config, "runs": runs}
    args.out.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
