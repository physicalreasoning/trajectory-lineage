"""Run classical overlap-detector baselines on controlled trajectory families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_lineage.detectors import (
    centroid_motion_descriptor,
    descriptor_distance,
    exact_video_hash,
    silhouette_descriptor,
    state_trajectory_distance,
)
from trajectory_lineage.simulator import InitialState, render, simulate


def sample_trajectory(
    rng: np.random.Generator,
    *,
    frames: int,
    frame_dt: float,
):
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
        if collision_time is None or not 0.34 <= collision_time <= 0.72:
            continue
        try:
            return state, simulate(state, frames=frames, dt=frame_dt)
        except ValueError:
            continue
    raise RuntimeError("could not sample a valid trajectory")


def fraction(values: list[bool]) -> float:
    return float(np.mean(values))


def run(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(config["seed"])
    thresholds = config["thresholds"]
    records: list[dict[str, float | bool | int]] = []

    for family in range(int(config["families"])):
        source_state, source = sample_trajectory(
            rng,
            frames=int(config["frames"]),
            frame_dt=float(config["frame_dt"]),
        )
        independent_state, independent = sample_trajectory(
            rng,
            frames=int(config["frames"]),
            frame_dt=float(config["frame_dt"]),
        )
        radius_pixels = max(2, round(source_state.radius * int(config["image_size"])))
        source_render = render(
            source,
            appearance_seed=int(rng.integers(0, 2**31 - 1)),
            size=int(config["image_size"]),
            radius_pixels=radius_pixels,
        )
        rerender = render(
            source,
            appearance_seed=int(rng.integers(0, 2**31 - 1)),
            size=int(config["image_size"]),
            radius_pixels=radius_pixels,
        )
        independent_render = render(
            independent,
            appearance_seed=int(rng.integers(0, 2**31 - 1)),
            size=int(config["image_size"]),
            radius_pixels=max(
                2,
                round(independent_state.radius * int(config["image_size"])),
            ),
        )

        source_silhouette = silhouette_descriptor(source_render)
        source_motion = centroid_motion_descriptor(source_render)
        same_silhouette_distance = descriptor_distance(
            source_silhouette,
            silhouette_descriptor(rerender),
        )
        independent_silhouette_distance = descriptor_distance(
            source_silhouette,
            silhouette_descriptor(independent_render),
        )
        same_motion_distance = descriptor_distance(
            source_motion,
            centroid_motion_descriptor(rerender),
        )
        independent_motion_distance = descriptor_distance(
            source_motion,
            centroid_motion_descriptor(independent_render),
        )
        same_oracle_distance = state_trajectory_distance(source.positions, source.positions)
        independent_oracle_distance = state_trajectory_distance(
            source.positions,
            independent.positions,
        )

        records.append(
            {
                "family": family,
                "same_exact_match": exact_video_hash(source_render) == exact_video_hash(rerender),
                "independent_exact_match": exact_video_hash(source_render)
                == exact_video_hash(independent_render),
                "same_silhouette_distance": same_silhouette_distance,
                "independent_silhouette_distance": independent_silhouette_distance,
                "same_motion_distance": same_motion_distance,
                "independent_motion_distance": independent_motion_distance,
                "same_oracle_distance": same_oracle_distance,
                "independent_oracle_distance": independent_oracle_distance,
                "same_silhouette_match": same_silhouette_distance
                <= thresholds["silhouette_distance"],
                "independent_silhouette_match": independent_silhouette_distance
                <= thresholds["silhouette_distance"],
                "same_motion_match": same_motion_distance <= thresholds["centroid_motion_distance"],
                "independent_motion_match": independent_motion_distance
                <= thresholds["centroid_motion_distance"],
                "same_oracle_match": same_oracle_distance <= thresholds["oracle_state_distance"],
                "independent_oracle_match": independent_oracle_distance
                <= thresholds["oracle_state_distance"],
            }
        )

    aggregate = {
        "families": len(records),
        "exact_hash_same_trajectory_recall": fraction(
            [bool(row["same_exact_match"]) for row in records]
        ),
        "silhouette_same_trajectory_recall": fraction(
            [bool(row["same_silhouette_match"]) for row in records]
        ),
        "motion_same_trajectory_recall": fraction(
            [bool(row["same_motion_match"]) for row in records]
        ),
        "oracle_same_trajectory_recall": fraction(
            [bool(row["same_oracle_match"]) for row in records]
        ),
        "exact_hash_independent_false_positive": fraction(
            [bool(row["independent_exact_match"]) for row in records]
        ),
        "silhouette_independent_false_positive": fraction(
            [bool(row["independent_silhouette_match"]) for row in records]
        ),
        "motion_independent_false_positive": fraction(
            [bool(row["independent_motion_match"]) for row in records]
        ),
        "oracle_independent_false_positive": fraction(
            [bool(row["independent_oracle_match"]) for row in records]
        ),
    }
    gates = config["gates"]
    aggregate["gate_passed"] = bool(
        aggregate["exact_hash_same_trajectory_recall"]
        <= gates["exact_hash_same_trajectory_recall_at_most"]
        and aggregate["silhouette_same_trajectory_recall"]
        >= gates["silhouette_same_trajectory_recall_at_least"]
        and aggregate["motion_same_trajectory_recall"]
        >= gates["motion_same_trajectory_recall_at_least"]
        and aggregate["oracle_same_trajectory_recall"] == gates["oracle_same_trajectory_recall"]
        and max(
            aggregate["exact_hash_independent_false_positive"],
            aggregate["silhouette_independent_false_positive"],
            aggregate["motion_independent_false_positive"],
            aggregate["oracle_independent_false_positive"],
        )
        <= gates["independent_false_positive_at_most"]
    )
    return {"config": config, "aggregate": aggregate, "families": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/detector_smoke.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/detector_smoke.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
