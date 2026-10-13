"""Run pre-specified state-trajectory baselines on PhyWorld collision data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from trajectory_lineage.phyworld import file_sha256

FloatArray = NDArray[np.float64]


def load_group(path: Path, group: str) -> tuple[FloatArray, FloatArray]:
    with h5py.File(path, "r") as handle:
        initial = np.asarray(handle["init_streams"][group][:], dtype=np.float64)
        positions = np.asarray(handle["position_streams"][group][:], dtype=np.float64)
    return initial, positions


def observable_features(initial: FloatArray, positions: FloatArray, observed: int) -> FloatArray:
    """Use visible radii and observed positions; do not use source velocity parameters."""

    radii = initial[:, :2]
    prefix = positions[:, :observed].reshape(len(positions), -1)
    return np.concatenate([radii, prefix], axis=1)


def constant_velocity(positions: FloatArray, observed: int) -> FloatArray:
    last = positions[:, observed - 1]
    velocity_per_frame = positions[:, observed - 1] - positions[:, observed - 2]
    steps = np.arange(1, positions.shape[1] - observed + 1, dtype=np.float64)
    return last[:, None] + steps[None, :, None, None] * velocity_per_frame[:, None]


def nearest_neighbor(
    train_features: FloatArray,
    train_future: FloatArray,
    test_features: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    squared = np.square(test[:, None, :] - train[None, :, :]).mean(axis=-1)
    nearest = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(test)), nearest])
    return train_future[nearest], distances


def per_example_error(predicted: FloatArray, target: FloatArray, world_scale: float) -> FloatArray:
    errors = np.linalg.norm(predicted - target, axis=-1) / world_scale
    return errors.mean(axis=(1, 2))


def visible_only_errors(
    predicted: FloatArray,
    target: FloatArray,
    radii: FloatArray,
    world_scale: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Average each example over frames where both disks are fully visible."""

    visible = (
        (target[..., 0] - radii[:, None, :] >= 0.0)
        & (target[..., 0] + radii[:, None, :] <= world_scale)
        & (target[..., 1] - radii[:, None, :] >= 0.0)
        & (target[..., 1] + radii[:, None, :] <= world_scale)
    ).all(axis=2)
    included = visible.any(axis=1)
    errors = np.linalg.norm(predicted - target, axis=-1) / world_scale
    per_example = np.array(
        [errors[index, mask].mean() for index, mask in enumerate(visible) if mask.any()]
    )
    return per_example, included


def paired_bootstrap(
    difference: FloatArray,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_split(
    *,
    name: str,
    train_initial: FloatArray,
    train_positions: FloatArray,
    test_initial: FloatArray,
    test_positions: FloatArray,
    observed: int,
    world_scale: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    train_features = observable_features(train_initial, train_positions, observed)
    test_features = observable_features(test_initial, test_positions, observed)
    target = test_positions[:, observed:]
    constant = constant_velocity(test_positions, observed)
    nearest, nearest_distance = nearest_neighbor(
        train_features,
        train_positions[:, observed:],
        test_features,
    )
    constant_error = per_example_error(constant, target, world_scale)
    nearest_error = per_example_error(nearest, target, world_scale)
    difference = nearest_error - constant_error
    visible_constant, included_constant = visible_only_errors(
        constant,
        target,
        test_initial[:, :2],
        world_scale,
    )
    visible_nearest, included_nearest = visible_only_errors(
        nearest,
        target,
        test_initial[:, :2],
        world_scale,
    )
    if not np.array_equal(included_constant, included_nearest):
        raise RuntimeError("visible-frame inclusion differs between predictors")
    visible_difference = visible_nearest - visible_constant
    return {
        "name": name,
        "train_examples": len(train_positions),
        "test_examples": len(test_positions),
        "constant_velocity": {
            "mean_error": float(constant_error.mean()),
            "median_error": float(np.median(constant_error)),
            "p90_error": float(np.quantile(constant_error, 0.9)),
        },
        "nearest_trajectory": {
            "mean_error": float(nearest_error.mean()),
            "median_error": float(np.median(nearest_error)),
            "p90_error": float(np.quantile(nearest_error, 0.9)),
            "mean_feature_distance": float(nearest_distance.mean()),
        },
        "nearest_minus_constant": {
            "mean": float(difference.mean()),
            "bootstrap_95_interval": paired_bootstrap(
                difference,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "fraction_nearest_better": float(np.mean(difference < 0.0)),
        },
        "visible_only_sensitivity": {
            "included_examples": int(included_constant.sum()),
            "excluded_examples": int((~included_constant).sum()),
            "constant_velocity_mean_error": float(visible_constant.mean()),
            "nearest_trajectory_mean_error": float(visible_nearest.mean()),
            "nearest_minus_constant_mean": float(visible_difference.mean()),
            "bootstrap_95_interval": paired_bootstrap(
                visible_difference,
                seed=bootstrap_seed + 10_000,
                resamples=bootstrap_resamples,
            ),
        },
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["dataset_path"])
    actual_sha = file_sha256(path)
    if actual_sha != config["dataset_sha256"]:
        raise ValueError(f"dataset checksum mismatch: {actual_sha}")

    observed = int(config["observed_frames"])
    future = int(config["future_frames"])
    world_scale = float(config["source_world_scale"])
    bootstrap_seed = int(config["bootstrap_seed"])
    bootstrap_resamples = int(config["bootstrap_resamples"])
    rng = np.random.default_rng(config["split_seed"])

    within_initial, within_positions = load_group(path, config["within_group"])
    if within_positions.shape[1] != observed + future:
        raise ValueError("observed and future frame counts do not match the dataset")
    order = rng.permutation(len(within_positions))
    train_count = int(len(order) * float(config["within_group_train_fraction"]))
    train_indices, test_indices = order[:train_count], order[train_count:]

    results = [
        evaluate_split(
            name=f"within_{config['within_group']}",
            train_initial=within_initial[train_indices],
            train_positions=within_positions[train_indices],
            test_initial=within_initial[test_indices],
            test_positions=within_positions[test_indices],
            observed=observed,
            world_scale=world_scale,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
    ]

    cross_initial, cross_positions = load_group(path, config["cross_group_train"])
    for offset, group in enumerate(config["cross_group_tests"], start=1):
        test_initial, test_positions = load_group(path, group)
        results.append(
            evaluate_split(
                name=f"{config['cross_group_train']}_to_{group}",
                train_initial=cross_initial,
                train_positions=cross_positions,
                test_initial=test_initial,
                test_positions=test_positions,
                observed=observed,
                world_scale=world_scale,
                bootstrap_seed=bootstrap_seed + offset,
                bootstrap_resamples=bootstrap_resamples,
            )
        )

    return {
        "config": config,
        "dataset_sha256": actual_sha,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/phyworld_state_baselines.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/phyworld_state_baselines.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
