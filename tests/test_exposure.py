import math

import numpy as np
import pytest

from trajectory_lineage.detectors import state_trajectory_distance
from trajectory_lineage.exposure import (
    EventSpec,
    blinded_summary,
    chi_square_quantile,
    independent_distance_band,
    matched_arm_schedule,
    render_event,
    reuse_contrasts,
    sample_event,
    sample_nearby_event,
    select_reservoir_neighbors,
    standard_deviation_upper_limit,
)


def test_reservoir_neighbors_are_unique_and_inside_band() -> None:
    rng = np.random.default_rng(11)
    spec = EventSpec()
    sources = [sample_event(rng, spec) for _ in range(8)]
    reservoir = [sample_event(rng, spec) for _ in range(4000)]
    low, high = independent_distance_band(
        rng, reservoir, low_quantile=0.001, high_quantile=0.01, pairs=20_000
    )

    neighbors, distances = select_reservoir_neighbors(
        rng, sources, reservoir, min_distance=low, max_distance=high
    )

    assert 0 < low < high
    assert len({id(event) for event in neighbors}) == len(sources)
    assert all(low <= distance <= high for distance in distances)
    for source, neighbor, distance in zip(sources, neighbors, distances, strict=True):
        assert distance == pytest.approx(
            state_trajectory_distance(source.trajectory.positions, neighbor.trajectory.positions)
        )


def test_nearby_event_is_new_simulation_inside_band() -> None:
    rng = np.random.default_rng(3)
    spec = EventSpec()
    source = sample_event(rng, spec)

    neighbor, distance = sample_nearby_event(
        rng,
        source,
        spec,
        perturbation=0.08,
        min_distance=0.01,
        max_distance=0.03,
    )

    assert 0.01 <= distance <= 0.03
    assert distance == state_trajectory_distance(
        source.trajectory.positions, neighbor.trajectory.positions
    )
    assert neighbor.state != source.state


def test_rerender_changes_pixels_but_not_trajectory() -> None:
    rng = np.random.default_rng(5)
    spec = EventSpec()
    event = sample_event(rng, spec)

    first = render_event(event, spec, appearance_seed=1)
    second = render_event(event, spec, appearance_seed=2)

    assert first.shape == (spec.observed_frames, spec.image_size, spec.image_size, 3)
    assert not np.array_equal(first, second)


def test_matched_schedule_balances_arm_presentations() -> None:
    arm, replay = matched_arm_schedule(
        np.random.default_rng(0),
        steps=100,
        arm_per_batch=8,
        replay_per_batch=24,
        arm_examples=64,
        replay_examples=512,
    )

    counts = np.bincount(arm.ravel(), minlength=64)
    assert arm.shape == (100, 8)
    assert replay.shape == (100, 24)
    assert counts.sum() == 800
    assert counts.max() - counts.min() <= 1


def test_reuse_contrasts_use_b_as_control() -> None:
    errors = {
        "B": {"source": 0.040, "independent": 0.030},
        "C": {"source": 0.030, "independent": 0.031},
        "D": {"source": 0.035, "independent": 0.030},
        "E": {"source": 0.020, "independent": 0.032},
    }

    contrasts = reuse_contrasts(errors)

    assert contrasts["C_source_gain"] == pytest.approx(0.010)
    assert contrasts["C_independent_gain"] == pytest.approx(-0.001)
    assert contrasts["C_reuse_gap"] == pytest.approx(0.011)
    assert contrasts["C_minus_D_source"] == pytest.approx(0.005)
    assert contrasts["C_minus_D_gap"] == pytest.approx(0.011 - 0.005)


def test_chi_square_approximation_is_close_to_reference() -> None:
    # Reference: chi-square 0.20 quantile with 7 degrees of freedom is 3.822.
    assert chi_square_quantile(0.2, 7) == pytest.approx(3.822, abs=0.02)


def test_blinded_summary_hides_location() -> None:
    shifted = blinded_summary([1.0, 2.0, 3.0, 4.0])
    centered = blinded_summary([-1.5, -0.5, 0.5, 1.5])

    assert shifted == centered
    assert "mean" not in shifted
    assert shifted["standard_deviation_upper_limit"] > shifted["standard_deviation"]


def test_upper_limit_requires_two_blocks() -> None:
    with pytest.raises(ValueError):
        standard_deviation_upper_limit(1.0, 1)
    assert math.isfinite(standard_deviation_upper_limit(1.0, 8))


def test_block_streams_are_disjoint_across_blocks_and_roles() -> None:
    from trajectory_lineage.exposure import block_streams

    first = block_streams(7, 0)
    second = block_streams(7, 1)
    draws = {
        (block, role): tuple(np.random.default_rng(stream).integers(0, 2**63 - 1, size=4))
        for block, streams in ((0, first), (1, second))
        for role, stream in streams.items()
    }
    assert len(set(draws.values())) == len(draws)
    again = block_streams(7, 0)
    assert np.random.default_rng(again["data"]).integers(0, 1000) == np.random.default_rng(
        first["data"]
    ).integers(0, 1000)


def test_appearance_registry_never_repeats() -> None:
    from trajectory_lineage.exposure import AppearanceRegistry

    class Repeating:
        def __init__(self) -> None:
            self.values = iter([5, 5, 5, 9])

        def integers(self, low, high):
            return next(self.values)

    registry = AppearanceRegistry(Repeating())
    assert [registry.draw(), registry.draw()] == [5, 9]
    assert len(registry) == 2


def _constant_event(offset: float):
    from trajectory_lineage.exposure import Event
    from trajectory_lineage.simulator import InitialState, Trajectory

    positions = np.zeros((4, 2, 2))
    positions[..., 0] = 0.5 + offset
    positions[..., 1] = 0.5
    trajectory = Trajectory(
        frame_times=np.arange(4) * 0.04,
        positions=positions,
        velocities=np.zeros_like(positions),
        collision_time=None,
    )
    return Event(state=InitialState(0.1, 0.8, 0.5, -0.1), trajectory=trajectory)


def test_neighbor_matching_succeeds_where_greedy_can_fail() -> None:
    sources = [_constant_event(0.0), _constant_event(0.3)]
    shared = _constant_event(0.15)
    only_first = _constant_event(-0.15)
    reservoir = [shared, only_first]
    for seed in range(40):
        neighbors, distances = select_reservoir_neighbors(
            np.random.default_rng(seed), sources, reservoir, min_distance=0.05, max_distance=0.2
        )
        assert neighbors[0] is only_first
        assert neighbors[1] is shared
        assert all(0.05 <= distance <= 0.2 for distance in distances)


def test_statistics_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        standard_deviation_upper_limit(-1.0, 8)
    with pytest.raises(ValueError):
        standard_deviation_upper_limit(float("nan"), 8)
    with pytest.raises(ValueError):
        blinded_summary([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        chi_square_quantile(0.001, 1)


def test_closer_neighbor_lies_inside_relative_band() -> None:
    from trajectory_lineage.exposure import nearest_distances, sample_closer_neighbor

    rng = np.random.default_rng(21)
    spec = EventSpec()
    base = [sample_event(rng, spec) for _ in range(256)]
    source = sample_event(rng, spec)
    reference = float(nearest_distances([source], base)[0])
    neighbor, distance, offset = sample_closer_neighbor(
        rng, source, spec, min_distance=0.25 * reference, max_distance=0.5 * reference
    )
    assert 0.25 * reference <= distance <= 0.5 * reference
    assert distance < float(nearest_distances([source], base)[0])
    assert 0 <= offset <= 1
    assert neighbor.state != source.state
