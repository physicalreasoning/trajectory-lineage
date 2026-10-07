"""Small, analytic collision generator for lineage-audit smoke tests.

This is intentionally not the research simulator yet. It establishes the contracts that
matter before model training: deterministic states, continuous collision timing, independent
rendering, and hashes that distinguish a physical trajectory from its pixel realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class InitialState:
    """Initial state for two equal-radius disks moving on one horizontal line."""

    x_left: float
    x_right: float
    vx_left: float
    vx_right: float
    y: float = 0.5
    radius: float = 0.055

    def collision_time(self) -> float | None:
        """Return the first contact time, or ``None`` when the disks never approach."""

        closing_speed = self.vx_left - self.vx_right
        gap = self.x_right - self.x_left - 2.0 * self.radius
        if closing_speed <= 0.0 or gap < 0.0:
            return None
        return gap / closing_speed


@dataclass(frozen=True)
class Trajectory:
    """A deterministic state trajectory.

    ``positions[t, disk, xy]`` and ``velocities[t, disk, xy]`` are sampled at the same frame
    times. Equal masses exchange their horizontal velocities at contact.
    """

    frame_times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    collision_time: float | None


def simulate(state: InitialState, *, frames: int = 24, dt: float = 0.04) -> Trajectory:
    """Simulate an equal-mass elastic collision at exact continuous collision time."""

    if frames < 2:
        raise ValueError("frames must be at least 2")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if state.x_left >= state.x_right:
        raise ValueError("left disk must begin left of right disk")

    times = np.arange(frames, dtype=np.float64) * dt
    positions = np.zeros((frames, 2, 2), dtype=np.float64)
    velocities = np.zeros_like(positions)
    collision_time = state.collision_time()

    for index, time in enumerate(times):
        collided = collision_time is not None and time > collision_time
        if not collided:
            left_x = state.x_left + state.vx_left * time
            right_x = state.x_right + state.vx_right * time
            left_vx, right_vx = state.vx_left, state.vx_right
        else:
            left_contact = state.x_left + state.vx_left * collision_time
            right_contact = state.x_right + state.vx_right * collision_time
            elapsed = time - collision_time
            left_x = left_contact + state.vx_right * elapsed
            right_x = right_contact + state.vx_left * elapsed
            left_vx, right_vx = state.vx_right, state.vx_left

        positions[index] = ((left_x, state.y), (right_x, state.y))
        velocities[index] = ((left_vx, 0.0), (right_vx, 0.0))

    if np.any(positions[..., 0] < state.radius) or np.any(positions[..., 0] > 1.0 - state.radius):
        raise ValueError("trajectory leaves the unit arena")

    return Trajectory(
        frame_times=times,
        positions=positions,
        velocities=velocities,
        collision_time=collision_time,
    )


def render(
    trajectory: Trajectory,
    *,
    appearance_seed: int,
    size: int = 64,
    radius_pixels: int = 4,
) -> UInt8Array:
    """Render states without changing them.

    Appearance seeds choose colors and a low-contrast background. The physical state array is
    never regenerated, which makes a same-trajectory rerender explicit and testable.
    """

    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    if radius_pixels < 1:
        raise ValueError("radius_pixels must be positive")

    rng = np.random.default_rng(appearance_seed)
    background = rng.integers(224, 248, size=3, dtype=np.uint8)
    colors = rng.integers(30, 205, size=(2, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    frames = np.empty((len(trajectory.frame_times), size, size, 3), dtype=np.uint8)

    for frame_index, frame_positions in enumerate(trajectory.positions):
        image = np.broadcast_to(background, (size, size, 3)).copy()
        for disk_index, (x_pos, y_pos) in enumerate(frame_positions):
            center_x = int(round(x_pos * (size - 1)))
            center_y = int(round(y_pos * (size - 1)))
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_pixels**2
            image[mask] = colors[disk_index]
        frames[frame_index] = image

    return frames
