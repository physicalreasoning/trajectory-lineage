import numpy as np
import pytest

from trajectory_lineage import bounded_center_error, evaluation_reuse_effect


def test_center_error_is_bounded_and_invalid_frames_stay_in_denominator() -> None:
    target = np.zeros((2, 2, 2), dtype=np.float64)
    predicted = target.copy()
    predicted[0, 0, 0] = 3.0

    measured = bounded_center_error(
        predicted,
        target,
        invalid_frames=np.array([False, True]),
    )

    assert measured == pytest.approx(0.75)


def test_evaluation_reuse_effect_is_paired_by_block() -> None:
    source = np.array([0.10, 0.08, 0.12])
    population = np.array([0.02, 0.01, 0.03])
    assert evaluation_reuse_effect(source, population, reuse_fraction=0.5) == pytest.approx(0.04)


def test_evaluation_reuse_effect_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="paired"):
        evaluation_reuse_effect(np.array([0.1]), np.array([0.1, 0.2]))
