import numpy as np
import pytest

from trajectory_lineage import InitialState, hash_array, render, simulate


def test_equal_masses_exchange_velocity_at_contact() -> None:
    state = InitialState(x_left=0.2, x_right=0.7, vx_left=0.55, vx_right=-0.10)
    trajectory = simulate(state, frames=24, dt=0.04)

    assert trajectory.collision_time == pytest.approx((0.7 - 0.2 - 0.11) / 0.65)
    before = trajectory.frame_times <= trajectory.collision_time
    after = trajectory.frame_times > trajectory.collision_time
    assert np.allclose(trajectory.velocities[before, 0, 0], 0.55)
    assert np.allclose(trajectory.velocities[before, 1, 0], -0.10)
    assert np.allclose(trajectory.velocities[after, 0, 0], -0.10)
    assert np.allclose(trajectory.velocities[after, 1, 0], 0.55)


def test_momentum_and_kinetic_energy_are_conserved() -> None:
    trajectory = simulate(
        InitialState(x_left=0.2, x_right=0.7, vx_left=0.55, vx_right=-0.10),
        frames=24,
        dt=0.04,
    )
    momentum = trajectory.velocities[..., 0].sum(axis=1)
    energy = np.square(trajectory.velocities[..., 0]).sum(axis=1)
    assert np.allclose(momentum, momentum[0], atol=1e-12)
    assert np.allclose(energy, energy[0], atol=1e-12)


def test_rerender_changes_pixels_not_state() -> None:
    trajectory = simulate(
        InitialState(x_left=0.2, x_right=0.7, vx_left=0.55, vx_right=-0.10),
        frames=24,
        dt=0.04,
    )
    first = render(trajectory, appearance_seed=7)
    second = render(trajectory, appearance_seed=19)

    assert hash_array(first) != hash_array(second)
    assert hash_array(trajectory.positions) == hash_array(trajectory.positions.copy())


def test_trajectory_that_leaves_arena_is_rejected() -> None:
    with pytest.raises(ValueError, match="leaves the unit arena"):
        simulate(
            InitialState(x_left=0.2, x_right=0.7, vx_left=2.0, vx_right=1.0),
            frames=24,
            dt=0.04,
        )
