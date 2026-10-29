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
import os
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
    capacity_levels,
    capacity_schedule,
    independent_distance_band,
    matched_arm_schedule,
    nearest_distances,
    render_event,
    reuse_contrasts,
    sample_closer_neighbors,
    sample_event,
    select_reservoir_neighbors,
)
from trajectory_lineage.latent import LatentStatePredictor, get_encoder

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
    sampled = sample_closer_neighbors(
        rng,
        sources,
        spec,
        min_distances=[float(closer_config["low_fraction"]) * float(r) for r in base_nearest],
        max_distances=[float(closer_config["high_fraction"]) * float(r) for r in base_nearest],
        box_multiplier=float(closer_config["box_multiplier"]),
        workers=config.get("build_workers"),
    )
    nearby = [neighbor for neighbor, _, _ in sampled]
    nearby_distances = [distance for _, distance, _ in sampled]
    box_offsets = [offset for _, _, offset in sampled]
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


def block_state_hash(block: dict) -> dict[str, str]:
    """SHA-256 of every population's initial states (the generated events), so a block can be
    re-derived from (configuration, seed, index) and checked byte for byte at analysis."""

    from dataclasses import asdict

    digests: dict[str, str] = {}
    populations = dict(block["events"])
    populations["screen"] = block["screen_events"]
    for name, events in populations.items():
        payload = json.dumps(
            [asdict(event.state) for event in events], sort_keys=True, separators=(",", ":")
        )
        digests[name] = hashlib.sha256(payload.encode()).hexdigest()
    digests["all"] = hashlib.sha256(
        "".join(digests[k] for k in sorted(digests)).encode()
    ).hexdigest()
    return digests


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


def predictor_config(config: dict[str, Any]) -> dict[str, Any]:
    """The predictor block: ``{"type": "convlstm"}`` (default) or a latent world model
    ``{"type": "latent", "encoder": ..., "depth": ..., "heads": ...}`` with encoder
    ``dinov2-small``, ``vjepa2-vitl`` or ``fake-<dim>``.
    For both types ``hidden_channels`` (and ``capacity_levels``) set the capacity: ConvLSTM hidden
    channels, or the latent transformer's width."""

    return dict({"type": "convlstm"}, **(config.get("predictor") or {}))


def encode_block(block: dict, config: dict[str, Any], device) -> dict:
    """For latent predictors: replace every population's video tensor by frozen-encoder tokens.

    The encoder never trains, so the block's training trajectories never touch it; tokens are
    computed once per block for all populations and arms (arm E and the exact population share
    one)."""

    pc = predictor_config(config)
    if pc["type"] != "latent":
        return block
    encoder = get_encoder(pc["encoder"], device)
    cache: dict[int, torch.Tensor] = {}

    def tokens_for(video: torch.Tensor) -> torch.Tensor:
        key = id(video)
        if key not in cache:
            cache[key] = encoder.encode(video)
        return cache[key]

    out = dict(block)
    for population in ("base", "source", "exact", "independent", "screen"):
        video, target = block[population]
        out[population] = (tokens_for(video), target)
    out["arms"] = {
        arm: (tokens_for(video), target) for arm, (video, target) in block["arms"].items()
    }
    return out


def new_model(config: dict[str, Any], spec: EventSpec, device, hidden_channels: int | None = None):
    pc = predictor_config(config)
    if pc["type"] == "latent":
        encoder = get_encoder(pc["encoder"], device)
        steps = (
            spec.observed_frames // 2
            if pc["encoder"].startswith("vjepa2")
            else spec.observed_frames
        )
        return LatentStatePredictor(
            token_dim=encoder.token_dim,
            steps=steps,
            tokens_per_step=encoder.tokens_per_step,
            future_frames=spec.future_frames,
            width=int(
                hidden_channels or config["hidden_channels"]
            ),  # capacity levels = transformer width
            depth=int(pc.get("depth", 4)),
            heads=int(pc.get("heads", 8)),
        ).to(device)
    return ConvLSTMStatePredictor(
        future_frames=spec.future_frames,
        hidden_channels=int(hidden_channels or config["hidden_channels"]),
        image_size=spec.image_size,
    ).to(device)


def base_steps_for(config: dict[str, Any], hidden_channels: int) -> int:
    """Base training steps for a capacity level: a registered per-level override, else default."""

    overrides = config.get("base_training_steps_by_capacity") or {}
    return int(overrides.get(str(hidden_channels), config["base_training_steps"]))


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
    hidden = capacity_schedule(config)[block_index]
    streams = block_streams(int(config["experiment_seed"]), block_index)
    torch.manual_seed(int(streams["torch"].generate_state(1)[0]))
    block = build_block(config, np.random.default_rng(streams["data"]), spec)
    build_seconds = time.perf_counter() - started
    encode_started = time.perf_counter()
    block = encode_block(block, config, device)
    encode_seconds = time.perf_counter() - encode_started
    cap = float(config["error_cap"])

    base_video, base_target = block["base"]
    batch_size = int(config["batch_size"])
    base_rng = np.random.default_rng(streams["base_batches"])
    base_indices = [
        torch.from_numpy(base_rng.integers(0, len(base_video), size=batch_size))
        for _ in range(base_steps_for(config, hidden))
    ]
    # gathered lazily: identical batches and order, without materializing ~600 copies (gigabytes)
    base_batches = ((base_video[index], base_target[index]) for index in base_indices)

    model = new_model(config, spec, device, hidden)
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
        arm_model = new_model(config, spec, device, hidden)
        arm_model.load_state_dict(base_state)
        arm_optimizer = torch.optim.AdamW(arm_model.parameters(), lr=float(config["learning_rate"]))
        arm_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        batches = (
            (
                torch.cat((arm_video[torch.from_numpy(a)], base_video[torch.from_numpy(r)])),
                torch.cat((arm_target[torch.from_numpy(a)], base_target[torch.from_numpy(r)])),
            )
            for a, r in zip(arm_schedule, replay_schedule, strict=True)
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
        "hidden_channels": hidden,
        "base_training_steps": base_steps_for(config, hidden),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "state_sha256": block_state_hash(block),
        "predictor": predictor_config(config),
        "encode_seconds": encode_seconds,
        "errors": errors,
        "item_losses": items,
        "source_family_of_render": block["source_family_of_render"],
        "base_errors": base_errors,
        "screen_base_error": screen_error,
        "screen_constant_velocity_error": screen_cv_error,
        "base_competent": bool(base_competent),
        "contrasts": reuse_contrasts(errors),
        "build_seconds": build_seconds,
        "learning_rate": float(config["learning_rate"]),
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
    exact_positive = sum(
        run.get("E_exact_gain_positive", run["contrasts"].get("E_exact_gain", 0.0) > 0)
        for run in completed
    )
    balance = [
        run["base_errors"]["source"] - run["base_errors"]["independent"] for run in completed
    ]
    report = {
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

    levels = capacity_levels(config)
    if len(levels) > 1:
        schedule = capacity_schedule(config)
        by_level: dict[str, Any] = {}
        for level in levels:
            level_runs = [run for run in runs if schedule[run["block_index"]] == level]
            level_all = [run for run in level_runs if run["status"] == "completed"]
            level_ok = [run for run in level_all if run.get("base_competent", True)]
            by_level[str(level)] = {
                "blocks_scheduled": len(level_runs),
                "blocks_completed": len(level_all),
                "blocks_base_competent": len(level_ok),
                "base_competence_failures": [
                    run["block_index"] for run in level_all if not run.get("base_competent", True)
                ],
                "model_parameters": level_ok[0]["model_parameters"] if level_ok else None,
                "dispersion": {
                    name: blinded_summary(
                        [run["contrasts"][name] for run in level_ok], confidence=confidence
                    )
                    for name in names
                }
                if len(level_ok) >= 2
                else {},
                "seconds_per_block_mean": float(np.mean([run["total_seconds"] for run in level_ok]))
                if level_ok
                else None,
            }
        report["by_capacity"] = by_level

    if not config.get("blinded_release_positive_control", True):
        # The flag is derived from signs of arm E gains; withheld when the release must be
        # dispersion-only (registered 2D confirmation).
        report.pop("positive_control_passed")
    return report


ANALYZER = Path(__file__).with_name("analyze_confirmation.py")


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyzer_for(config: dict[str, Any]) -> Path:
    """The analyzer bound by the freeze manifest: the one the configuration names."""

    return Path(__file__).resolve().parents[1] / config.get(
        "analyzer_script", "scripts/analyze_confirmation.py"
    )


def check_freeze(manifest_path: Path, config_path: Path) -> None:
    """Refuse unblinding unless the manifest passes the shared freeze validation."""

    try:
        validate_freeze(
            json.loads(manifest_path.read_text()),
            config_path,
            analyzer_for(json.loads(config_path.read_text())),
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


def configure_backend(device) -> None:
    """Speed knobs that do not change the registered computation's meaning: TF32 matmuls and cuDNN
    autotuning on CUDA. Training stays in float32; only frozen encoders run under bf16 autocast."""

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _worker_run(
    config_path: str, block_index: int, device_name: str, sampler_workers: int
) -> dict[str, Any]:
    """Process-pool entry: one block, in its own process with its own CUDA context and encoder."""

    import os

    os.environ["TRAJECTORY_LINEAGE_WORKERS"] = str(max(1, sampler_workers))
    config = json.loads(Path(config_path).read_text())
    device = torch.device(device_name)
    configure_backend(device)
    return run_with_policy(config, block_index, spec_from(config), device)


def run_blocks_parallel(
    config_path: Path, indices: list[int], device_name: str, workers: int, on_done
) -> None:
    """Run blocks concurrently in ``workers`` spawned processes sharing one device.

    Blocks are independent (per-block seeds), so results do not depend on which worker runs which
    block or on completion order; ``on_done(run)`` is called as each block finishes. The arm-D
    sampler inside each worker gets an equal share of the CPU cores."""

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    sampler_workers = max(1, (os.cpu_count() or 2) // max(1, workers))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(_worker_run, str(config_path), index, device_name, sampler_workers): index
            for index in indices
        }
        for future in as_completed(futures):
            on_done(future.result())


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


BLIND_DROP = ("errors", "item_losses", "source_family_of_render")


def load_blinding_key(path: Path, names: list[str]) -> dict[str, float]:
    """Per-contrast random offsets. Created once per run; must never be opened by the owner."""

    if path.exists():
        key = json.loads(path.read_text())
        missing = [name for name in names if name not in key]
        if missing:
            raise SystemExit(f"blinding key {path} lacks offsets for {missing}")
        return {name: float(key[name]) for name in key}
    rng = np.random.default_rng()
    key = {name: float(rng.normal(0.0, 10.0)) for name in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    return key


def blind_record(run: dict[str, Any], key: dict[str, float]) -> dict[str, Any]:
    """A block record safe to store: no component errors, contrasts shifted by the key.

    Dispersions (standard deviations) are invariant to the shift, the positive-control flag is
    kept as a boolean, and pre-treatment base errors stay (they carry no reuse effect)."""

    if run.get("status") != "completed" or run.get("blinded"):
        return run
    record = {k: v for k, v in run.items() if k not in BLIND_DROP}
    missing = [name for name in run["contrasts"] if name not in key]
    if missing:
        raise SystemExit(f"blinding key lacks offsets for {missing}")
    record["contrasts"] = {
        name: float(value) + key[name] for name, value in run["contrasts"].items()
    }
    record["E_exact_gain_positive"] = bool(run["contrasts"]["E_exact_gain"] > 0)
    record["blinded"] = True
    return record


def load_checkpoint(path: Path, expected_config_hash: str) -> list[dict[str, Any]]:
    """Raw block records written so far; refuses a file produced under another configuration."""

    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("config_sha256") != expected_config_hash:
            raise SystemExit(f"checkpoint {path} was written under a different configuration")
        runs.append(record["run"])
    indices = [run["block_index"] for run in runs]
    if len(set(indices)) != len(indices):
        raise SystemExit(f"checkpoint {path} records a block twice")
    return runs


def append_checkpoint(path: Path, config_sha256: str, run: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps({"config_sha256": config_sha256, "run": run}, sort_keys=True) + "\n"
        )
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument("--unblind", type=Path, default=None, help="Path for raw block records.")
    parser.add_argument("--freeze-manifest", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JSONL of raw block records, appended per block and resumed on restart. Raw records "
        "carry effects, so this is allowed only for approved confirmatory runs "
        "(--freeze-manifest).",
    )
    parser.add_argument(
        "--blinded-checkpoint",
        type=Path,
        default=None,
        help="JSONL of BLINDED block records (contrasts shifted by a random key, no component "
        "errors), appended per block and resumed on restart. Safe for effect-blinded pilots.",
    )
    parser.add_argument(
        "--blinding-key",
        type=Path,
        default=None,
        help="Where the random offsets live (default: next to the blinded checkpoint). "
        "Never open it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Blocks trained concurrently in separate processes on the same device "
        "(GPU runs: 2-4). "
        "Results do not depend on this value.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.unblind is not None or config.get("require_approval_to_run", False):
        if args.freeze_manifest is None:
            raise SystemExit("this run requires --freeze-manifest (approved, confirmatory)")
        check_freeze(args.freeze_manifest, args.config)
    if args.checkpoint is not None and args.freeze_manifest is None:
        raise SystemExit("--checkpoint writes raw block records; it requires --freeze-manifest")
    if args.blinded_checkpoint is not None and args.unblind is not None:
        raise SystemExit(
            "--blinded-checkpoint keeps no raw records; it cannot be combined with --unblind"
        )
    if args.blinded_checkpoint is not None and args.checkpoint is not None:
        raise SystemExit(
            "use either --checkpoint (confirmatory, raw) or --blinded-checkpoint (pilot)"
        )
    blinding_key: dict[str, float] | None = None
    if args.blinded_checkpoint is not None:
        key_path = args.blinding_key or args.blinded_checkpoint.with_name(
            "DO_NOT_OPEN_blinding_key.json"
        )
        zero = {arm: dict.fromkeys(POPULATIONS, 0.0) for arm in ARMS}
        blinding_key = load_blinding_key(key_path, sorted(reuse_contrasts(zero)))

    spec = spec_from(config)
    device_name = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_name)
    checkpoint = args.checkpoint or args.blinded_checkpoint
    runs = load_checkpoint(checkpoint, config_hash(args.config)) if checkpoint else []
    done = {run["block_index"] for run in runs}
    if done:
        print(f"resuming: {len(done)} blocks already recorded in {checkpoint}", flush=True)
    started = time.perf_counter()
    configure_backend(device)
    pending = [index for index in range(int(config["blocks"])) if index not in done]
    finished = {"n": 0}

    def record(run: dict[str, Any]) -> None:
        if blinding_key is not None:
            run = blind_record(run, blinding_key)
        runs.append(run)
        if checkpoint is not None:
            append_checkpoint(checkpoint, config_hash(args.config), run)
        finished["n"] += 1
        elapsed = time.perf_counter() - started
        rate = elapsed / finished["n"]
        remaining = (len(pending) - finished["n"]) * rate / max(1, args.workers)
        print(
            f"block {run['block_index'] + 1}/{config['blocks']}: {run['status']}"
            f" (hidden {run.get('hidden_channels', '?')}, {run.get('total_seconds', 0):.0f} s;"
            f" {finished['n']}/{len(pending)} this session, {elapsed / 60:.1f} min elapsed,"
            f" about {remaining / 60:.0f} min left)",
            flush=True,
        )

    if args.workers > 1 and len(pending) > 1:
        run_blocks_parallel(args.config, pending, device_name, args.workers, record)
    else:
        for block_index in pending:
            record(run_with_policy(config, block_index, spec, device))
    runs.sort(key=lambda run: run["block_index"])

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
