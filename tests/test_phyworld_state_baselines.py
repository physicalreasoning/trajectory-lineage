import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "phyworld_state_baselines.py"
_SPEC = importlib.util.spec_from_file_location("phyworld_state_baselines", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_constant_velocity_extends_last_observed_step() -> None:
    positions = np.zeros((1, 6, 2, 2), dtype=np.float64)
    positions[0, :, 0, 0] = np.arange(6)
    positions[0, :, 1, 0] = 10 - np.arange(6)

    predicted = _MODULE.constant_velocity(positions, observed=3)

    assert np.array_equal(predicted, positions[:, 3:])


def test_nearest_neighbor_returns_matching_future() -> None:
    train_features = np.array([[0.0, 0.0], [1.0, 1.0]])
    train_future = np.array([[[[0.0, 0.0]]], [[[1.0, 1.0]]]])
    predicted, distances = _MODULE.nearest_neighbor(
        train_features,
        train_future,
        np.array([[0.9, 1.1]]),
    )

    assert np.array_equal(predicted, train_future[[1]])
    assert distances[0] > 0.0


def test_per_example_error_normalizes_by_world_scale() -> None:
    target = np.zeros((1, 2, 2, 2), dtype=np.float64)
    predicted = target.copy()
    predicted[..., 0] = 1.0

    error = _MODULE.per_example_error(predicted, target, world_scale=10.0)

    assert error[0] == 0.1


def test_visible_only_errors_exclude_examples_without_visible_frames() -> None:
    target = np.array(
        [
            [[[2.0, 5.0], [8.0, 5.0]], [[-1.0, 5.0], [11.0, 5.0]]],
            [[[-1.0, 5.0], [11.0, 5.0]], [[-2.0, 5.0], [12.0, 5.0]]],
        ]
    )
    predicted = target.copy()
    predicted[0, 0, :, 0] += 1.0
    radii = np.full((2, 2), 0.5)

    errors, included = _MODULE.visible_only_errors(
        predicted,
        target,
        radii,
        world_scale=10.0,
    )

    assert included.tolist() == [True, False]
    assert errors.tolist() == [0.1]
