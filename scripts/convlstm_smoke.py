"""Run the registered ConvLSTM visual-state feasibility experiment."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trajectory_lineage.convlstm import ConvLSTMStatePredictor
from trajectory_lineage.simulator import InitialState, render, simulate


def sample_example(
    rng: np.random.Generator,
    *,
    frames: int,
    observed_frames: int,
    frame_dt: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one visible collision with contact after the observation prefix."""

    earliest_contact = observed_frames * frame_dt
    latest_contact = (frames - 4) * frame_dt
    for _ in range(10_000):
        state = InitialState(
            x_left=float(rng.uniform(0.10, 0.25)),
            x_right=float(rng.uniform(0.65, 0.82)),
            vx_left=float(rng.uniform(0.45, 0.72)),
            vx_right=float(rng.uniform(-0.16, 0.02)),
            y=float(rng.uniform(0.30, 0.70)),
            radius=float(rng.uniform(0.04, 0.065)),
        )
        collision_time = state.collision_time()
        if collision_time is None or not earliest_contact <= collision_time <= latest_contact:
            continue
        try:
            trajectory = simulate(state, frames=frames, dt=frame_dt)
        except ValueError:
            continue
        pixels = render(
            trajectory,
            appearance_seed=int(rng.integers(0, 2**31 - 1)),
            size=image_size,
            radius_pixels=max(2, round(state.radius * image_size)),
        )
        return pixels[:observed_frames], trajectory.positions
    raise RuntimeError("could not sample a valid collision")


def make_dataset(
    config: dict[str, Any], *, examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    observed_frames = int(config["observed_frames"])
    total_frames = observed_frames + int(config["future_frames"])
    videos: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for _ in range(examples):
        video, trajectory = sample_example(
            rng,
            frames=total_frames,
            observed_frames=observed_frames,
            frame_dt=float(config["frame_dt"]),
            image_size=int(config["image_size"]),
        )
        videos.append(video)
        states.append(trajectory)
    video_tensor = torch.from_numpy(np.stack(videos).astype(np.float32) / 255.0)
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3)
    state_tensor = torch.from_numpy(np.stack(states).astype(np.float32))
    return video_tensor, state_tensor


def center_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(predicted - target, dim=-1).mean())


def persistence_baseline(states: torch.Tensor, observed_frames: int) -> torch.Tensor:
    future_frames = states.shape[1] - observed_frames
    return states[:, observed_frames - 1, None].expand(-1, future_frames, -1, -1)


def constant_velocity_baseline(states: torch.Tensor, observed_frames: int) -> torch.Tensor:
    future_frames = states.shape[1] - observed_frames
    last = states[:, observed_frames - 1]
    velocity_per_frame = states[:, observed_frames - 1] - states[:, observed_frames - 2]
    steps = torch.arange(1, future_frames + 1, dtype=states.dtype).view(1, -1, 1, 1)
    return last[:, None] + steps * velocity_per_frame[:, None]


def run_seed(
    config: dict[str, Any], seed: int, device: torch.device
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_video, train_states = make_dataset(
        config,
        examples=int(config["train_examples"]),
        seed=seed,
    )
    test_video, test_states = make_dataset(
        config,
        examples=int(config["test_examples"]),
        seed=seed + 1_000_000,
    )

    observed_frames = int(config["observed_frames"])
    target = train_states[:, observed_frames:].to(device)
    model = ConvLSTMStatePredictor(
        future_frames=int(config["future_frames"]),
        hidden_channels=int(config["hidden_channels"]),
        image_size=int(config["image_size"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(seed + 2_000_000)
    batch_size = int(config["batch_size"])
    losses: list[float] = []
    started = time.perf_counter()

    model.train()
    train_video = train_video.to(device)
    for _ in range(int(config["training_steps"])):
        indices = torch.from_numpy(rng.integers(0, len(train_video), size=batch_size)).to(device)
        prediction = model(train_video[indices])
        loss = torch.square(prediction - target[indices]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        prediction = model(test_video.to(device)).cpu()
    test_target = test_states[:, observed_frames:]
    model_error = center_error(prediction, test_target)
    persistence_error = center_error(
        persistence_baseline(test_states, observed_frames),
        test_target,
    )
    constant_velocity_error = center_error(
        constant_velocity_baseline(test_states, observed_frames),
        test_target,
    )

    return {
        "seed": seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - started,
        "initial_loss": losses[0],
        "final_loss_mean_20": float(np.mean(losses[-20:])),
        "model_center_error": model_error,
        "persistence_center_error": persistence_error,
        "constant_velocity_center_error": constant_velocity_error,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    started = time.perf_counter()
    runs = [run_seed(config, int(seed), device) for seed in config["seeds"]]

    model_errors = np.array([row["model_center_error"] for row in runs])
    persistence_errors = np.array([row["persistence_center_error"] for row in runs])
    constant_errors = np.array([row["constant_velocity_center_error"] for row in runs])
    gates = config["gates"]
    passed = (
        (not gates["all_seed_errors_finite"] or np.isfinite(model_errors).all())
        and (
            not gates["model_mean_error_below_persistence"]
            or model_errors.mean() < persistence_errors.mean()
        )
        and (
            not gates["model_mean_error_below_constant_velocity"]
            or model_errors.mean() < constant_errors.mean()
        )
    )

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))),
        }

    return {
        "config": config,
        "device": str(device),
        "total_seconds": time.perf_counter() - started,
        "aggregate": {
            "model_center_error": summary(model_errors),
            "persistence_center_error": summary(persistence_errors),
            "constant_velocity_center_error": summary(constant_errors),
            "gate_passed": bool(passed),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/convlstm_smoke.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/convlstm_smoke.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
