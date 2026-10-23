"""Two-dimensional equal-mass elastic disk collisions for the registered 2D reuse study.

States are parameterized by collision geometry rather than by initial positions. Every
accepted event therefore contains exactly one visible contact, and the impact parameter
``impact`` controls how glancing it is: 0 is head-on, values near +/-1 graze. This gives 2D
trajectories that can be "nearby" in many ways (contact time, place, direction, obliquity),
which is what makes the same-event versus nearby-event comparison meaningful.

Identity convention: disk 0 is the disk that starts on the left. Positions keep this index
for the whole trajectory, and errors are computed per disk with no per-frame rematching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trajectory_lineage.simulator import Trajectory

FloatArray = np.ndarray


@dataclass(frozen=True)
class InitialState2D:
    """Collision-parameterized state of two equal-radius, equal-mass disks.

    ``contact_time``: time of first contact (s). ``contact_x``/``contact_y``: contact point.
    ``vx0, vy0`` and ``vx1, vy1``: pre-contact velocities of disk 0 and disk 1.
    ``impact``: signed impact parameter as a fraction of the contact distance ``2 * radius``.
    """

    contact_time: float
    contact_x: float
    contact_y: float
    vx0: float
    vy0: float
    vx1: float
    vy1: float
    impact: float
    radius: float = 0.055

    def contact_normal(self) -> np.ndarray | None:
        """Unit normal from disk 0 to disk 1 at contact, or ``None`` if they never approach."""

        rel = np.array([self.vx0 - self.vx1, self.vy0 - self.vy1], dtype=np.float64)
        speed = float(np.linalg.norm(rel))
        if speed < 1e-9 or not -1.0 < self.impact < 1.0:
            return None
        u = rel / speed
        angle = math.asin(self.impact)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # rotate the approach direction by the impact angle: u . n = cos(angle) > 0
        return np.array([cos_a * u[0] - sin_a * u[1], sin_a * u[0] + cos_a * u[1]])

    def initial_positions(self) -> np.ndarray | None:
        normal = self.contact_normal()
        if normal is None:
            return None
        center = np.array([self.contact_x, self.contact_y])
        v = np.array([[self.vx0, self.vy0], [self.vx1, self.vy1]])
        at_contact = np.stack([center - self.radius * normal, center + self.radius * normal])
        return at_contact - v * self.contact_time

    def collision_time(self) -> float | None:
        """First contact time, verified analytically from the reconstructed initial state."""

        start = self.initial_positions()
        if start is None or self.contact_time < 0:
            return None
        dp = start[1] - start[0]
        dv = np.array([self.vx1 - self.vx0, self.vy1 - self.vy0])
        a = float(dv @ dv)
        b = 2.0 * float(dp @ dv)
        c = float(dp @ dp) - (2.0 * self.radius) ** 2
        if c < 0 or a < 1e-12:
            return None  # overlapping at t = 0, or no relative motion
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / (2 * a)
        if t < 0 or abs(t - self.contact_time) > 1e-6:
            return None
        return t


def simulate_2d(state: InitialState2D, *, frames: int = 24, dt: float = 0.04) -> Trajectory:
    """Simulate one exact elastic contact (velocity components along the normal exchange)."""

    if frames < 2 or dt <= 0:
        raise ValueError("frames must be >= 2 and dt positive")
    collision_time = state.collision_time()
    start = state.initial_positions()
    if collision_time is None or start is None:
        raise ValueError("state has no valid first contact")
    if start[0, 0] >= start[1, 0]:
        raise ValueError("disk 0 must start left of disk 1 (identity convention)")
    normal = state.contact_normal()
    v = np.array([[state.vx0, state.vy0], [state.vx1, state.vy1]], dtype=np.float64)
    exchange = float((v[0] - v[1]) @ normal)
    v_after = v.copy()
    v_after[0] -= exchange * normal
    v_after[1] += exchange * normal
    at_contact = start + v * collision_time

    times = np.arange(frames, dtype=np.float64) * dt
    positions = np.zeros((frames, 2, 2))
    velocities = np.zeros((frames, 2, 2))
    for i, t in enumerate(times):
        if t <= collision_time:
            positions[i] = start + v * t
            velocities[i] = v
        else:
            positions[i] = at_contact + v_after * (t - collision_time)
            velocities[i] = v_after
    r = state.radius
    if np.any(positions < r) or np.any(positions > 1.0 - r):
        raise ValueError("trajectory leaves the unit arena")
    return Trajectory(
        frame_times=times, positions=positions, velocities=velocities, collision_time=collision_time
    )
