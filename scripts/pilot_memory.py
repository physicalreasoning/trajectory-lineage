"""Run the registered nearest-trajectory memory positive control."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_lineage import (
    InitialState,
    NearestTrajectoryPredictor,
    bounded_center_error,
    evaluation_reuse_effect,
    simulate,
)


def sample_trajectory(
    rng: np.random.Generator,
    *,
    frames: int,
    dt: float,
    observed_frames: int,
    dims: int = 1,
) -> np.ndarray:
    """Sample an accepted collision with contact after observation and before the final frame."""

    if dims == 2:
        from trajectory_lineage.exposure import EventSpec, sample_event

        spec = EventSpec(
            observed_frames=observed_frames,
            future_frames=frames - observed_frames,
            frame_dt=dt,
            dims=2,
        )
        return sample_event(rng, spec).trajectory.positions
    earliest = observed_frames * dt
    latest = (frames - 4) * dt
    for _ in range(10_000):
        state = InitialState(
            x_left=float(rng.uniform(0.12, 0.28)),
            x_right=float(rng.uniform(0.58, 0.76)),
            vx_left=float(rng.uniform(0.42, 0.72)),
            vx_right=float(rng.uniform(-0.14, 0.08)),
            y=0.5,
            radius=0.045,
        )
        collision_time = state.collision_time()
        if collision_time is None or not earliest <= collision_time <= latest:
            continue
        try:
            return simulate(state, frames=frames, dt=dt).positions
        except ValueError:
            continue
    raise RuntimeError("could not sample an accepted trajectory")


def sample_bank(
    rng: np.random.Generator,
    count: int,
    *,
    frames: int,
    dt: float,
    observed_frames: int,
    dims: int = 1,
) -> np.ndarray:
    return np.stack(
        [
            sample_trajectory(
                rng,
                frames=frames,
                dt=dt,
                observed_frames=observed_frames,
                dims=dims,
            )
            for _ in range(count)
        ]
    )


def evaluate(predictor: NearestTrajectoryPredictor, examples: np.ndarray, observed: int) -> float:
    predicted, _ = predictor.predict(examples[:, :observed])
    target = examples[:, observed:]
    return float(
        np.mean(
            [
                bounded_center_error(one_prediction, one_target)
                for one_prediction, one_target in zip(predicted, target, strict=True)
            ]
        )
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(config["seed"])
    blocks = int(config["blocks"])
    bank_size = int(config["bank_size"])
    population_size = int(config["population_evaluation_size"])
    observed = int(config["observed_frames"])
    future = int(config["future_frames"])
    frames = observed + future
    dt = float(config["frame_dt"])
    reuse_fraction = float(config["reuse_fraction"])
    dims = int(config.get("dims", 1))
    block_results: list[dict[str, float | int]] = []

    started = time.perf_counter()
    for block in range(blocks):
        source_bank = sample_bank(
            rng,
            bank_size,
            frames=frames,
            dt=dt,
            observed_frames=observed,
            dims=dims,
        )
        background_bank = sample_bank(
            rng,
            bank_size,
            frames=frames,
            dt=dt,
            observed_frames=observed,
            dims=dims,
        )
        independent_population = sample_bank(
            rng,
            population_size,
            frames=frames,
            dt=dt,
            observed_frames=observed,
            dims=dims,
        )

        source_predictor = NearestTrajectoryPredictor().fit(
            source_bank[:, :observed], source_bank[:, observed:]
        )
        background_predictor = NearestTrajectoryPredictor().fit(
            background_bank[:, :observed], background_bank[:, observed:]
        )

        loss_source_on_source = evaluate(source_predictor, source_bank, observed)
        loss_background_on_source = evaluate(background_predictor, source_bank, observed)
        loss_source_on_population = evaluate(source_predictor, independent_population, observed)
        loss_background_on_population = evaluate(
            background_predictor, independent_population, observed
        )
        source_gain = loss_background_on_source - loss_source_on_source
        population_gain = loss_background_on_population - loss_source_on_population
        effect = reuse_fraction * (source_gain - population_gain)

        block_results.append(
            {
                "block": block,
                "loss_source_on_source": loss_source_on_source,
                "loss_background_on_source": loss_background_on_source,
                "loss_source_on_population": loss_source_on_population,
                "loss_background_on_population": loss_background_on_population,
                "source_gain": source_gain,
                "population_gain": population_gain,
                "evaluation_reuse_effect": effect,
            }
        )

    effects = np.array([row["evaluation_reuse_effect"] for row in block_results])
    source_gains = np.array([row["source_gain"] for row in block_results])
    population_gains = np.array([row["population_gain"] for row in block_results])
    mean_effect = evaluation_reuse_effect(
        source_gains,
        population_gains,
        reuse_fraction=reuse_fraction,
    )
    standard_deviation = float(effects.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(blocks)
    normal_lower = mean_effect - 1.96 * standard_error
    gate = config["gate"]
    passed = (
        mean_effect > float(gate["mean_effect_above"])
        and normal_lower > float(gate["normal_95_lower_bound_above"])
        and max(row["loss_source_on_source"] for row in block_results)
        < float(gate["all_source_policy_source_bank_losses_below"])
    )

    return {
        "config": config,
        "seconds": time.perf_counter() - started,
        "aggregate": {
            "blocks": blocks,
            "mean_source_gain": float(source_gains.mean()),
            "mean_population_gain": float(population_gains.mean()),
            "mean_evaluation_reuse_effect": mean_effect,
            "effect_standard_deviation": standard_deviation,
            "effect_standard_error": standard_error,
            "normal_95_lower_bound": normal_lower,
            "gate_passed": passed,
        },
        "blocks": block_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/memory_positive_control.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/memory_positive_control.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
