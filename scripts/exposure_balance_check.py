"""Pre-treatment balance check: base-model error on source versus independent populations.

Uses only base checkpoints, so it reveals no exposure effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from exposure_experiment import (  # noqa: E402
    POPULATIONS,
    build_block,
    evaluate,
    new_model,
    provenance,
    spec_from,
    to_tensors,
    train_steps,
)

from trajectory_lineage.exposure import block_streams, render_event, sample_event  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/exposure_pilot_v3.json"))
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--experiment-seed", type=int, default=2026101624)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--out", type=Path, default=Path("results/exposure_balance_check.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = spec_from(config)
    device_name = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    cap = float(config["error_cap"])
    rows = []
    for block_index in range(args.blocks):
        streams = block_streams(args.experiment_seed, block_index)
        torch.manual_seed(int(streams["torch"].generate_state(1)[0]))
        data_rng = np.random.default_rng(streams["data"])
        block = build_block(config, data_rng, spec)
        video, target = block["base"]
        base_rng = np.random.default_rng(streams["base_batches"])
        batches = []
        for _ in range(int(config["base_training_steps"])):
            index = torch.from_numpy(
                base_rng.integers(0, len(video), size=int(config["batch_size"]))
            )
            batches.append((video[index], target[index]))
        model = new_model(config, spec, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
        train_steps(model, optimizer, batches, device)
        extra = [sample_event(data_rng, spec) for _ in range(int(config["source_trajectories"]))]
        fresh = to_tensors(
            [render_event(event, spec, int(data_rng.integers(0, 2**31 - 1))) for event in extra],
            extra,
            spec,
        )
        row = {
            population: evaluate(model, *block[population], device, cap)
            for population in POPULATIONS
        }
        row["fresh"] = evaluate(model, *fresh, device, cap)
        row["block_index"] = block_index
        rows.append(row)
        print(f"balance block {block_index + 1}/{args.blocks}", flush=True)

    summary = {}
    for population in ("source", "exact", "fresh"):
        difference = np.array([row[population] - row["independent"] for row in rows])
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        summary[f"{population}_minus_independent"] = {
            "mean": float(difference.mean()),
            "standard_deviation": float(difference.std(ddof=1)),
            "t_statistic": float(difference.mean() / standard_error),
        }
    result = {
        "config": config,
        "experiment_seed": args.experiment_seed,
        "blocks": len(rows),
        "summary": summary,
        "rows": rows,
        "provenance": provenance(device),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
