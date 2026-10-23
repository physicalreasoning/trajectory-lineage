import numpy as np
import pytest

from trajectory_lineage.exposure import EventSpec, sample_event, state_ranges
from trajectory_lineage.simulator2d import InitialState2D, simulate_2d


def _events(n=300, seed=0):
    spec = EventSpec(dims=2)
    rng = np.random.default_rng(seed)
    return spec, [sample_event(rng, spec) for _ in range(n)]


def test_conserves_momentum_and_energy():
    _, events = _events()
    for event in events:
        v = event.trajectory.velocities
        assert np.allclose(v[0].sum(0), v[-1].sum(0))
        assert np.isclose((v[0] ** 2).sum(), (v[-1] ** 2).sum())


def test_single_contact_inside_window_and_no_overlap():
    spec, events = _events()
    for event in events:
        t = event.trajectory.collision_time
        assert spec.observed_frames * spec.frame_dt <= t <= (spec.frames - 4) * spec.frame_dt
        gap = np.linalg.norm(
            event.trajectory.positions[:, 0] - event.trajectory.positions[:, 1], axis=-1
        )
        assert gap.min() >= 2 * event.state.radius - 1e-9


def test_identity_convention_disk0_starts_left():
    _, events = _events()
    assert all(e.trajectory.positions[0, 0, 0] < e.trajectory.positions[0, 1, 0] for e in events)


def test_glancing_collisions_are_represented():
    _, events = _events(n=600)
    impacts = np.array([e.state.impact for e in events])
    assert impacts.min() < -0.6 and impacts.max() > 0.6


def test_contact_time_is_verified_analytically():
    state = InitialState2D(0.5, 0.5, 0.5, 0.5, 0.0, -0.2, 0.0, 0.0, 0.05)
    assert state.collision_time() == pytest.approx(0.5, abs=1e-6)
    traj = simulate_2d(state, frames=24, dt=0.04)
    # head-on, equal masses: velocities exchange
    assert np.allclose(traj.velocities[-1, 0], [-0.2, 0.0]) and np.allclose(
        traj.velocities[-1, 1], [0.5, 0.0]
    )


def test_one_dimensional_ranges_unchanged():
    assert list(state_ranges(EventSpec())) == [
        "x_left",
        "x_right",
        "vx_left",
        "vx_right",
        "y",
        "radius",
    ]
    with pytest.raises(ValueError):
        state_ranges(EventSpec(dims=3))
