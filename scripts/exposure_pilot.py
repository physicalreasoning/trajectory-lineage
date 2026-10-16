"""Superseded v1/v2 pilot runner, kept only to document how those discarded pilots ran.

Known defects (fixed in ``scripts/exposure_experiment.py``): overlapping random streams across
consecutive blocks, an uncapped error metric, exclusion of nonfinite blocks, and incomplete
appearance-seed disjointness. Do not use for new runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from trajectory_lineage.convlstm import ConvLSTMStatePredictor
from trajectory_lineage.detectors import state_trajectory_distance
from trajectory_lineage.exposure import (
    Event,
    EventSpec,
    blinded_summary,
    independent_distance_band,
    matched_arm_schedule,
    render_event,
    reuse_contrasts,
    sample_event,
    select_reservoir_neighbors,
)

ARMS = ("B", "C", "D", "E")
POPULATIONS = ("source", "exact", "independent")


def to_tensors(videos: list[np.ndarray], events: list[Event], spec: EventSpec):
    video = torch.from_numpy(np.stack(videos).astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3)
    target = torch.from_numpy(
        np.stack([event.trajectory.positions[spec.observed_frames :] for event in events]).astype(
            np.float32
        )
    )
    return video, target


def appearance(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def build_block(config: dict[str, Any], seed: int, spec: EventSpec) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base_events = [sample_event(rng, spec) for _ in range(int(config["base_train_examples"]))]
    base_videos = [render_event(event, spec, appearance(rng)) for event in base_events]

    sources = [sample_event(rng, spec) for _ in range(int(config["source_trajectories"]))]
    exact_seeds = [appearance(rng) for _ in sources]
    exact_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(sources, exact_seeds, strict=True)
    ]

    used_seeds = set(exact_seeds)

    def fresh_seed() -> int:
        seed_value = appearance(rng)
        while seed_value in used_seeds:
            seed_value = appearance(rng)
        used_seeds.add(seed_value)
        return seed_value

    heldout_videos = [render_event(event, spec, fresh_seed()) for event in sources]
    rerender_videos = [render_event(event, spec, fresh_seed()) for event in sources]

    independent = [sample_event(rng, spec) for _ in range(int(config["independent_eval_examples"]))]
    independent_videos = [render_event(event, spec, appearance(rng)) for event in independent]

    replacement = [sample_event(rng, spec) for _ in sources]
    replacement_videos = [render_event(event, spec, appearance(rng)) for event in replacement]

    nearby_config = config["nearby"]
    reservoir = [sample_event(rng, spec) for _ in range(int(nearby_config["reservoir_size"]))]
    low, high = independent_distance_band(
        rng,
        reservoir,
        low_quantile=float(nearby_config["low_quantile"]),
        high_quantile=float(nearby_config["high_quantile"]),
    )
    nearby, nearby_distances = select_reservoir_neighbors(
        rng, sources, reservoir, min_distance=low, max_distance=high
    )
    nearby_videos = [render_event(event, spec, appearance(rng)) for event in nearby]

    replacement_distances = [
        state_trajectory_distance(source.trajectory.positions, other.trajectory.positions)
        for source, other in zip(sources, replacement, strict=True)
    ]

    return {
        "base": to_tensors(base_videos, base_events, spec),
        "source": to_tensors(heldout_videos, sources, spec),
        "exact": to_tensors(exact_videos, sources, spec),
        "independent": to_tensors(independent_videos, independent, spec),
        "arms": {
            "B": to_tensors(replacement_videos, replacement, spec),
            "C": to_tensors(rerender_videos, sources, spec),
            "D": to_tensors(nearby_videos, nearby, spec),
            "E": to_tensors(exact_videos, sources, spec),
        },
        "nearby_band": (low, high),
        "nearby_distances": nearby_distances,
        "replacement_distances": replacement_distances,
        "schedule_rng": np.random.default_rng(seed + 1),
        "base_rng": np.random.default_rng(seed + 2),
    }


def center_error(
    model: torch.nn.Module, video: torch.Tensor, target: torch.Tensor, device
) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(video.to(device)).cpu()
    return float(torch.linalg.vector_norm(prediction - target, dim=-1).mean())


def train_steps(model, optimizer, batches, device) -> list[float]:
    losses = []
    model.train()
    for video, target in batches:
        prediction = model(video.to(device))
        loss = torch.square(prediction - target.to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def run_block(config: dict[str, Any], seed: int, spec: EventSpec, device) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    block = build_block(config, seed, spec)
    build_seconds = time.perf_counter() - started

    base_video, base_target = block["base"]
    batch_size = int(config["batch_size"])
    base_batches = []
    for _ in range(int(config["base_training_steps"])):
        indices = torch.from_numpy(block["base_rng"].integers(0, len(base_video), size=batch_size))
        base_batches.append((base_video[indices], base_target[indices]))

    model = ConvLSTMStatePredictor(
        future_frames=spec.future_frames,
        hidden_channels=int(config["hidden_channels"]),
        image_size=spec.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    base_losses = train_steps(model, optimizer, base_batches, device)
    base_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    base_errors = {
        population: center_error(model, *block[population], device)
        for population in ("source", "independent")
    }

    arm_schedule, replay_schedule = matched_arm_schedule(
        block["schedule_rng"],
        steps=int(config["continuation_steps"]),
        arm_per_batch=int(config["arm_examples_per_batch"]),
        replay_per_batch=int(config["replay_examples_per_batch"]),
        arm_examples=int(config["source_trajectories"]),
        replay_examples=len(base_video),
    )

    errors: dict[str, dict[str, float]] = {}
    arm_seconds: dict[str, float] = {}
    final_losses: dict[str, float] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        arm_video, arm_target = block["arms"][arm]
        arm_model = ConvLSTMStatePredictor(
            future_frames=spec.future_frames,
            hidden_channels=int(config["hidden_channels"]),
            image_size=spec.image_size,
        ).to(device)
        arm_model.load_state_dict(base_state)
        arm_optimizer = torch.optim.AdamW(arm_model.parameters(), lr=float(config["learning_rate"]))
        arm_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        batches = []
        for arm_indices, replay_indices in zip(arm_schedule, replay_schedule, strict=True):
            arm_index = torch.from_numpy(arm_indices)
            replay_index = torch.from_numpy(replay_indices)
            batches.append(
                (
                    torch.cat((arm_video[arm_index], base_video[replay_index])),
                    torch.cat((arm_target[arm_index], base_target[replay_index])),
                )
            )
        losses = train_steps(arm_model, arm_optimizer, batches, device)
        final_losses[arm] = float(np.mean(losses[-10:]))
        errors[arm] = {
            population: center_error(arm_model, *block[population], device)
            for population in POPULATIONS
        }
        arm_seconds[arm] = time.perf_counter() - arm_started

    finite = all(math.isfinite(value) for row in errors.values() for value in row.values())
    return {
        "seed": seed,
        "errors": errors,
        "base_errors": base_errors,
        "contrasts": with_exact_diagnostics(errors) if finite else None,
        "finite": finite,
        "build_seconds": build_seconds,
        "arm_seconds": arm_seconds,
        "total_seconds": time.perf_counter() - started,
        "base_final_loss": float(np.mean(base_losses[-20:])),
        "arm_final_losses": final_losses,
        "nearby_band": list(block["nearby_band"]),
        "nearby_distance_median": float(np.median(block["nearby_distances"])),
        "replacement_distance_median": float(np.median(block["replacement_distances"])),
    }


def with_exact_diagnostics(errors: dict[str, dict[str, float]]) -> dict[str, float]:
    contrasts = reuse_contrasts(errors)
    contrasts["E_exact_gain"] = errors["B"]["exact"] - errors["E"]["exact"]
    contrasts["E_exact_minus_heldout"] = errors["E"]["source"] - errors["E"]["exact"]
    return contrasts


def required_blocks(standard_deviation: float, excess: float, alpha: float, power: float) -> int:
    normal = NormalDist()
    z_value = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return math.ceil((z_value * standard_deviation / excess) ** 2)


def blinded_report(config: dict[str, Any], runs: list[dict[str, Any]], device) -> dict[str, Any]:
    valid = [run for run in runs if run["finite"]]
    planning = config["planning"]
    contrast_names = sorted(valid[0]["contrasts"]) if valid else []
    dispersion = {
        name: blinded_summary(
            [run["contrasts"][name] for run in valid],
            confidence=float(planning["sd_upper_limit_confidence"]),
        )
        for name in contrast_names
    }
    for summary in dispersion.values():
        summary["required_blocks_at_upper_limit"] = required_blocks(
            float(summary["standard_deviation_upper_limit"]),
            float(planning["detectable_excess"]),
            float(planning["alpha"]),
            float(planning["power"]),
        )
    exact_positive = [run["contrasts"]["E_exact_gain"] > 0 for run in valid]
    base_balance = [
        run["base_errors"]["source"] - run["base_errors"]["independent"] for run in valid
    ]
    return {
        "config": config,
        "device": str(device),
        "blinded": True,
        "blocks_scheduled": len(runs),
        "blocks_valid": len(valid),
        "positive_control_passed": bool(valid and np.mean(exact_positive) >= 0.75),
        "dispersion": dispersion,
        "pretreatment_balance": {
            "base_source_minus_independent_mean": float(np.mean(base_balance)),
            "base_source_minus_independent_sd": float(np.std(base_balance, ddof=1)),
            "base_independent_error_mean": float(
                np.mean([run["base_errors"]["independent"] for run in valid])
            ),
        },
        "design_descriptors": {
            "nearby_band_mean": [
                float(np.mean([run["nearby_band"][0] for run in valid])),
                float(np.mean([run["nearby_band"][1] for run in valid])),
            ],
            "nearby_state_distance_median": float(
                np.median([run["nearby_distance_median"] for run in valid])
            ),
            "replacement_state_distance_median": float(
                np.median([run["replacement_distance_median"] for run in valid])
            ),
        },
        "runtime": {
            "seconds_per_block_mean": float(np.mean([run["total_seconds"] for run in runs])),
            "seconds_per_block_max": float(np.max([run["total_seconds"] for run in runs])),
            "total_seconds": float(np.sum([run["total_seconds"] for run in runs])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/exposure_pilot_v2.json"))
    parser.add_argument("--out", type=Path, default=Path("results/exposure_pilot_v2_blinded.json"))
    parser.add_argument(
        "--unblind",
        type=Path,
        default=None,
        help="Also write raw block errors to this path. Do not use before the threshold freeze.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = EventSpec(
        observed_frames=int(config["observed_frames"]),
        future_frames=int(config["future_frames"]),
        frame_dt=float(config["frame_dt"]),
        image_size=int(config["image_size"]),
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    start = int(config["block_seed_start"])
    runs = []
    for offset in range(int(config["blocks"])):
        run = run_block(config, start + offset, spec, device)
        runs.append(run)
        print(f"block {offset + 1}/{config['blocks']} finished in {run['total_seconds']:.1f}s")
    report = blinded_report(config, runs, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.unblind is not None:
        args.unblind.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "config"}, indent=2))


if __name__ == "__main__":
    main()
