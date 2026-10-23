"""Run the matched ConvLSTM trajectory-reuse experiment (pilot or confirmation).

By default the report is effect-blinded: it releases contrast dispersion, runtime, pre-treatment
balance, design descriptors, and a binary exact-render positive control. Raw block errors are
released only with ``--unblind`` together with a freeze manifest whose configuration hash matches
and which records collaborator sign-off.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from freeze import FreezeError, changed_since, git_state, run_identity  # noqa: E402
from freeze import validate as validate_freeze  # noqa: E402

from trajectory_lineage.convlstm import ConvLSTMStatePredictor
from trajectory_lineage.detectors import state_trajectory_distance
from trajectory_lineage.exposure import (
    AppearanceRegistry,
    Event,
    EventSpec,
    blinded_summary,
    block_streams,
    independent_distance_band,
    matched_arm_schedule,
    nearest_distances,
    render_event,
    reuse_contrasts,
    sample_closer_neighbor,
    sample_event,
    select_reservoir_neighbors,
)

ARMS = ("B", "C", "D", "D2", "E")
POPULATIONS = ("source", "exact", "independent")


def spec_from(config: dict[str, Any]) -> EventSpec:
    return EventSpec(
        observed_frames=int(config["observed_frames"]),
        future_frames=int(config["future_frames"]),
        frame_dt=float(config["frame_dt"]),
        image_size=int(config["image_size"]),
        dims=int(config.get("dims", 1)),
    )


def to_tensors(videos: list[np.ndarray], events: list[Event], spec: EventSpec):
    video = torch.from_numpy(np.stack(videos).astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3)
    target = torch.from_numpy(
        np.stack([event.trajectory.positions[spec.observed_frames :] for event in events]).astype(
            np.float32
        )
    )
    return video, target


def build_block(config: dict[str, Any], rng: np.random.Generator, spec: EventSpec) -> dict:
    """Construct every training and evaluation population for one independent block."""

    seeds = AppearanceRegistry(rng)
    render_seeds: dict[str, list[int]] = {}
    base_events = [sample_event(rng, spec) for _ in range(int(config["base_train_examples"]))]
    render_seeds["base"] = [seeds.draw() for _ in base_events]
    base_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(base_events, render_seeds["base"], strict=True)
    ]

    sources = [sample_event(rng, spec) for _ in range(int(config["source_trajectories"]))]
    render_seeds["exact"] = [seeds.draw() for _ in sources]
    exact_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(sources, render_seeds["exact"], strict=True)
    ]
    renders_per_source = int(config["heldout_renders_per_source"])
    heldout_events = [event for event in sources for _ in range(renders_per_source)]
    render_seeds["heldout"] = [seeds.draw() for _ in heldout_events]
    heldout_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(heldout_events, render_seeds["heldout"], strict=True)
    ]
    render_seeds["rerender"] = [seeds.draw() for _ in sources]
    rerender_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(sources, render_seeds["rerender"], strict=True)
    ]

    independent = [sample_event(rng, spec) for _ in range(int(config["independent_eval_examples"]))]
    render_seeds["independent"] = [seeds.draw() for _ in independent]
    independent_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(independent, render_seeds["independent"], strict=True)
    ]

    replacement = [sample_event(rng, spec) for _ in sources]
    render_seeds["replacement"] = [seeds.draw() for _ in replacement]
    replacement_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(replacement, render_seeds["replacement"], strict=True)
    ]

    screen = [sample_event(rng, spec) for _ in range(int(config["screen_examples"]))]
    render_seeds["screen"] = [seeds.draw() for _ in screen]
    screen_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(screen, render_seeds["screen"], strict=True)
    ]

    # D: independent neighbors closer than anything in base training, relative to each source's
    # nearest base-training distance r_i.
    closer_config = config["closer_nearby"]
    base_nearest = nearest_distances(sources, base_events)
    nearby, nearby_distances, box_offsets = [], [], []
    for source, reference in zip(sources, base_nearest, strict=True):
        neighbor, distance, offset = sample_closer_neighbor(
            rng,
            source,
            spec,
            min_distance=float(closer_config["low_fraction"]) * float(reference),
            max_distance=float(closer_config["high_fraction"]) * float(reference),
            box_multiplier=float(closer_config["box_multiplier"]),
        )
        nearby.append(neighbor)
        nearby_distances.append(distance)
        box_offsets.append(offset)
    render_seeds["nearby"] = [seeds.draw() for _ in nearby]
    nearby_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(nearby, render_seeds["nearby"], strict=True)
    ]

    # D2: reservoir-selected independent neighbors in a pairwise quantile band.
    nearby_config = config["nearby"]
    reservoir = [sample_event(rng, spec) for _ in range(int(nearby_config["reservoir_size"]))]
    low, high = independent_distance_band(
        rng,
        reservoir,
        low_quantile=float(nearby_config["low_quantile"]),
        high_quantile=float(nearby_config["high_quantile"]),
    )
    banded, banded_distances = select_reservoir_neighbors(
        rng, sources, reservoir, min_distance=low, max_distance=high
    )
    render_seeds["banded"] = [seeds.draw() for _ in banded]
    banded_videos = [
        render_event(event, spec, seed_value)
        for event, seed_value in zip(banded, render_seeds["banded"], strict=True)
    ]

    replacement_distances = [
        state_trajectory_distance(source.trajectory.positions, other.trajectory.positions)
        for source, other in zip(sources, replacement, strict=True)
    ]
    expected_renders = (
        len(base_events)
        + len(sources) * (2 + renders_per_source)
        + len(independent)
        + len(replacement)
        + len(nearby)
        + len(banded)
        + len(screen)
    )
    if len(seeds) != expected_renders:
        raise RuntimeError("appearance seeds are not unique across the block")

    return {
        "base": to_tensors(base_videos, base_events, spec),
        "source": to_tensors(heldout_videos, heldout_events, spec),
        "exact": to_tensors(exact_videos, sources, spec),
        "independent": to_tensors(independent_videos, independent, spec),
        "arms": {
            "B": to_tensors(replacement_videos, replacement, spec),
            "C": to_tensors(rerender_videos, sources, spec),
            "D": to_tensors(nearby_videos, nearby, spec),
            "D2": to_tensors(banded_videos, banded, spec),
            "E": to_tensors(exact_videos, sources, spec),
        },
        "nearby_band": (low, high),
        "nearby_distances": banded_distances,
        "closer_distances": nearby_distances,
        "base_nearest_distances": base_nearest.tolist(),
        "box_offsets": box_offsets,
        "replacement_distances": replacement_distances,
        "render_seeds": render_seeds,
        "events": {
            "base": base_events,
            "sources": sources,
            "independent": independent,
            "replacement": replacement,
            "nearby": nearby,
            "banded": banded,
        },
        "screen": to_tensors(screen_videos, screen, spec),
        "screen_events": screen,
        "source_family_of_render": [
            index for index in range(len(sources)) for _ in range(renders_per_source)
        ],
    }


def constant_velocity_error(events: list[Event], spec: EventSpec, cap: float) -> float:
    """Bounded error of constant-velocity extrapolation from the last two observed states."""

    positions = torch.from_numpy(np.stack([event.trajectory.positions for event in events]))
    last = positions[:, spec.observed_frames - 1]
    velocity = last - positions[:, spec.observed_frames - 2]
    steps = torch.arange(1, spec.future_frames + 1, dtype=positions.dtype).view(1, -1, 1, 1)
    prediction = last[:, None] + steps * velocity[:, None]
    target = positions[:, spec.observed_frames :]
    return float(bounded_errors(prediction, target, cap).mean())


def bounded_errors(prediction: torch.Tensor, target: torch.Tensor, cap: float) -> torch.Tensor:
    """Per-example mean of capped disk-frame errors; nonfinite predictions receive the cap."""

    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    distance = torch.where(torch.isfinite(distance), distance, torch.full_like(distance, cap))
    return distance.clamp(max=cap).mean(dim=(1, 2))


def evaluate_items(model, video, target, device, cap: float) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        prediction = model(video.to(device)).cpu()
    return bounded_errors(prediction, target, cap)


def evaluate(model, video, target, device, cap: float) -> float:
    return float(evaluate_items(model, video, target, device, cap).mean())


def rounded(values: torch.Tensor) -> list[float]:
    return [float(f"{value:.7g}") for value in values.tolist()]


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


def new_model(config: dict[str, Any], spec: EventSpec, device) -> ConvLSTMStatePredictor:
    return ConvLSTMStatePredictor(
        future_frames=spec.future_frames,
        hidden_channels=int(config["hidden_channels"]),
        image_size=spec.image_size,
    ).to(device)


def run_block(
    config: dict[str, Any],
    block_index: int,
    spec: EventSpec,
    device,
    arm_content_override: str | None = None,
) -> dict[str, Any]:
    """Train one base model and its four matched continuation arms.

    ``arm_content_override`` replaces every arm's content with one arm's content; it exists for
    the identical-content integration control and is never used by experiments.
    """

    started = time.perf_counter()
    streams = block_streams(int(config["experiment_seed"]), block_index)
    torch.manual_seed(int(streams["torch"].generate_state(1)[0]))
    block = build_block(config, np.random.default_rng(streams["data"]), spec)
    build_seconds = time.perf_counter() - started
    cap = float(config["error_cap"])

    base_video, base_target = block["base"]
    batch_size = int(config["batch_size"])
    base_rng = np.random.default_rng(streams["base_batches"])
    base_batches = []
    for _ in range(int(config["base_training_steps"])):
        index = torch.from_numpy(base_rng.integers(0, len(base_video), size=batch_size))
        base_batches.append((base_video[index], base_target[index]))

    model = new_model(config, spec, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    base_losses = train_steps(model, optimizer, base_batches, device)
    base_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    base_errors = {
        population: evaluate(model, *block[population], device, cap) for population in POPULATIONS
    }
    screen_error = evaluate(model, *block["screen"], device, cap)
    screen_cv_error = constant_velocity_error(block["screen_events"], spec, cap)
    base_competent = screen_error < screen_cv_error

    arm_schedule, replay_schedule = matched_arm_schedule(
        np.random.default_rng(streams["schedule"]),
        steps=int(config["continuation_steps"]),
        arm_per_batch=int(config["arm_examples_per_batch"]),
        replay_per_batch=int(config["replay_examples_per_batch"]),
        arm_examples=int(config["source_trajectories"]),
        replay_examples=len(base_video),
    )

    errors: dict[str, dict[str, float]] = {}
    items: dict[str, dict[str, list[float]]] = {}
    final_losses: dict[str, float] = {}
    for arm in ARMS:
        content = block["arms"][arm_content_override or arm]
        arm_video, arm_target = content
        arm_model = new_model(config, spec, device)
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
        item_losses = {
            population: evaluate_items(arm_model, *block[population], device, cap)
            for population in POPULATIONS
        }
        errors[arm] = {
            population: float(values.mean()) for population, values in item_losses.items()
        }
        items[arm] = {population: rounded(values) for population, values in item_losses.items()}

    presentations = np.bincount(arm_schedule.ravel(), minlength=int(config["source_trajectories"]))
    return {
        "block_index": block_index,
        "status": "completed",
        "errors": errors,
        "item_losses": items,
        "source_family_of_render": block["source_family_of_render"],
        "base_errors": base_errors,
        "screen_base_error": screen_error,
        "screen_constant_velocity_error": screen_cv_error,
        "base_competent": bool(base_competent),
        "contrasts": reuse_contrasts(errors),
        "build_seconds": build_seconds,
        "total_seconds": time.perf_counter() - started,
        "base_final_loss": float(np.mean(base_losses[-20:])),
        "arm_final_losses": final_losses,
        "presentations_per_source": [int(presentations.min()), int(presentations.max())],
        "nearby_band": list(block["nearby_band"]),
        "nearby_distance_median": float(np.median(block["nearby_distances"])),
        "closer_distance_median": float(np.median(block["closer_distances"])),
        "base_nearest_distance_median": float(np.median(block["base_nearest_distances"])),
        "closer_box_offset_max": float(np.max(block["box_offsets"])),
        "replacement_distance_median": float(np.median(block["replacement_distances"])),
    }


def provenance(device) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain") != "",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
    }


def blinded_report(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed_all = [run for run in runs if run["status"] == "completed"]
    completed = [run for run in completed_all if run.get("base_competent", True)]
    confidence = float(config["sd_upper_limit_confidence"])
    names = sorted(completed[0]["contrasts"]) if completed else []
    dispersion = {}
    if len(completed) >= 2:
        dispersion = {
            name: blinded_summary(
                [run["contrasts"][name] for run in completed], confidence=confidence
            )
            for name in names
        }
    exact_positive = sum(run["contrasts"]["E_exact_gain"] > 0 for run in completed)
    balance = [
        run["base_errors"]["source"] - run["base_errors"]["independent"] for run in completed
    ]
    return {
        "blinded": True,
        "blocks_scheduled": len(runs),
        "blocks_completed": len(completed_all),
        "blocks_base_competent": len(completed),
        "base_competence_failures": [
            run["block_index"] for run in completed_all if not run.get("base_competent", True)
        ],
        "infrastructure_failures": [
            {"block_index": run["block_index"], "error": run["error"]}
            for run in runs
            if run["status"] == "infrastructure_failure"
        ],
        "algorithmic_failures": [
            {"block_index": run["block_index"], "error": run["error"]}
            for run in runs
            if run["status"] == "algorithmic_failure"
        ],
        "positive_control_passed": bool(completed) and exact_positive >= 0.75 * len(runs),
        "dispersion": dispersion,
        "pretreatment_balance": {
            "base_source_minus_independent_mean": float(np.mean(balance)) if balance else None,
            "base_source_minus_independent_sd": float(np.std(balance, ddof=1))
            if len(balance) >= 2
            else None,
        },
        "design_descriptors": {
            "presentations_per_source": sorted(
                {tuple(run["presentations_per_source"]) for run in completed}
            ),
            "nearby_band_mean": [
                float(np.mean([run["nearby_band"][0] for run in completed])),
                float(np.mean([run["nearby_band"][1] for run in completed])),
            ]
            if completed
            else None,
            "nearby_state_distance_median": float(
                np.median([run["nearby_distance_median"] for run in completed])
            )
            if completed
            else None,
            "closer_state_distance_median": float(
                np.median([run["closer_distance_median"] for run in completed])
            )
            if completed
            else None,
            "base_nearest_distance_median": float(
                np.median([run["base_nearest_distance_median"] for run in completed])
            )
            if completed
            else None,
            "closer_box_offset_max": float(max(run["closer_box_offset_max"] for run in completed))
            if completed
            else None,
            "replacement_state_distance_median": float(
                np.median([run["replacement_distance_median"] for run in completed])
            )
            if completed
            else None,
        },
        "runtime": {
            "seconds_per_block_mean": float(np.mean([run["total_seconds"] for run in completed]))
            if completed
            else None,
            "total_seconds": float(np.sum([run.get("total_seconds", 0.0) for run in runs])),
        },
    }


ANALYZER = Path(__file__).with_name("analyze_confirmation.py")


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_freeze(manifest_path: Path, config_path: Path) -> None:
    """Refuse unblinding unless the manifest passes the shared freeze validation."""

    try:
        validate_freeze(
            json.loads(manifest_path.read_text()),
            config_path,
            ANALYZER,
            state=git_state,
            changes=changed_since,
        )
    except FreezeError as error:
        raise SystemExit(f"refusing to unblind: {error}") from error


INFRASTRUCTURE_MARKERS = ("mps", "out of memory", "cuda", "device")


def is_infrastructure(error: BaseException) -> bool:
    if isinstance(error, (OSError, MemoryError)):
        return True
    return isinstance(error, RuntimeError) and any(
        marker in str(error).lower() for marker in INFRASTRUCTURE_MARKERS
    )


def run_with_policy(config, block_index, spec, device) -> dict[str, Any]:
    """Run a block; retry once on infrastructure errors; record every failure by class."""

    attempts = []
    for _ in range(2):
        try:
            run = run_block(config, block_index, spec, device)
            run["attempts"] = attempts + ["completed"]
            return run
        except Exception as error:  # noqa: BLE001 - classified and recorded below
            attempts.append(repr(error))
            if not is_infrastructure(error):
                return {
                    "block_index": block_index,
                    "status": "algorithmic_failure",
                    "error": repr(error),
                    "attempts": attempts,
                }
    return {
        "block_index": block_index,
        "status": "infrastructure_failure",
        "error": attempts[-1],
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--unblind", type=Path, default=None, help="Path for raw block records.")
    parser.add_argument("--freeze-manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.unblind is not None:
        if args.freeze_manifest is None:
            raise SystemExit("--unblind requires --freeze-manifest")
        check_freeze(args.freeze_manifest, args.config)

    config = json.loads(args.config.read_text())
    spec = spec_from(config)
    device_name = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    runs = []
    for block_index in range(int(config["blocks"])):
        run = run_with_policy(config, block_index, spec, device)
        runs.append(run)
        print(f"block {block_index + 1}/{config['blocks']}: {run['status']}", flush=True)

    report = blinded_report(config, runs)
    report["config"] = config
    report["config_sha256"] = config_hash(args.config)
    report["provenance"] = provenance(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.unblind is not None:
        export = {"identity": run_identity(args.freeze_manifest, args.config), "runs": runs}
        args.unblind.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
