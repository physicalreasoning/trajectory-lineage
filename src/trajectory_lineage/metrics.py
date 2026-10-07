"""Bounded metrics whose failure handling is part of the measurement contract."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bounded_center_error(
    predicted: FloatArray,
    target: FloatArray,
    *,
    invalid_frames: NDArray[np.bool_] | None = None,
    cap: float = 1.0,
) -> float:
    """Return mean disk-center error, capped per object-frame.

    Inputs have shape ``[frames, disks, xy]`` in arena-width units. An invalid frame receives
    the full cap for every disk instead of disappearing from the denominator.
    """

    if predicted.shape != target.shape:
        raise ValueError("predicted and target shapes differ")
    if predicted.ndim != 3 or predicted.shape[-1] != 2:
        raise ValueError("positions must have shape [frames, disks, 2]")
    if cap <= 0:
        raise ValueError("cap must be positive")

    errors = np.linalg.norm(predicted - target, axis=-1)
    errors = np.minimum(errors, cap)

    if invalid_frames is not None:
        invalid = np.asarray(invalid_frames, dtype=bool)
        if invalid.shape != (predicted.shape[0],):
            raise ValueError("invalid_frames must have one value per frame")
        errors[invalid, :] = cap

    return float(errors.mean())


def evaluation_reuse_effect(
    source_gain: FloatArray,
    population_gain: FloatArray,
    *,
    reuse_fraction: float = 0.5,
) -> float:
    """Change in measured gain when source-family events replace population events.

    ``source_gain`` and ``population_gain`` are paired block-level policy gains. The function
    returns the across-block mean of ``reuse_fraction * (source_gain - population_gain)``.
    """

    source = np.asarray(source_gain, dtype=np.float64)
    population = np.asarray(population_gain, dtype=np.float64)
    if source.shape != population.shape or source.ndim != 1:
        raise ValueError("gains must be paired one-dimensional arrays")
    if not 0.0 <= reuse_fraction <= 1.0:
        raise ValueError("reuse_fraction must be between 0 and 1")
    return float(np.mean(reuse_fraction * (source - population)))
