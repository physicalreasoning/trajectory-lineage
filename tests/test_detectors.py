import numpy as np

from trajectory_lineage.detectors import (
    centroid_motion_descriptor,
    descriptor_distance,
    exact_video_hash,
    silhouette_descriptor,
    state_trajectory_distance,
)
from trajectory_lineage.simulator import InitialState, render, simulate


def trajectories():
    source = simulate(
        InitialState(x_left=0.2, x_right=0.7, vx_left=0.55, vx_right=-0.10),
        frames=24,
        dt=0.04,
    )
    independent = simulate(
        InitialState(x_left=0.15, x_right=0.75, vx_left=0.65, vx_right=-0.04),
        frames=24,
        dt=0.04,
    )
    return source, independent


def test_same_trajectory_rerenders_have_distinct_hashes_and_equal_motion() -> None:
    source, _ = trajectories()
    first = render(source, appearance_seed=7)
    second = render(source, appearance_seed=19)

    assert exact_video_hash(first) != exact_video_hash(second)
    assert descriptor_distance(silhouette_descriptor(first), silhouette_descriptor(second)) == 0.0
    assert (
        descriptor_distance(
            centroid_motion_descriptor(first),
            centroid_motion_descriptor(second),
        )
        == 0.0
    )


def test_motion_and_oracle_distance_separate_independent_trajectory() -> None:
    source, independent = trajectories()
    source_video = render(source, appearance_seed=7)
    independent_video = render(independent, appearance_seed=23)

    motion_distance = descriptor_distance(
        centroid_motion_descriptor(source_video),
        centroid_motion_descriptor(independent_video),
    )
    oracle_distance = state_trajectory_distance(source.positions, independent.positions)

    assert motion_distance > 0.0
    assert oracle_distance > 0.0


def test_state_distance_is_invariant_to_global_object_swap() -> None:
    source, _ = trajectories()

    assert state_trajectory_distance(source.positions, source.positions[:, ::-1]) == 0.0


def test_silhouette_descriptor_rejects_incompatible_size() -> None:
    video = np.zeros((2, 30, 30, 3), dtype=np.uint8)

    try:
        silhouette_descriptor(video, cells=8)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected descriptor size validation")
