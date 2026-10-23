"""Event construction and blinded summaries for the matched trajectory-reuse pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from trajectory_lineage.detectors import state_trajectory_distance
from trajectory_lineage.simulator import InitialState, Trajectory, render, simulate
from trajectory_lineage.simulator2d import InitialState2D, simulate_2d

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]

STATE_RANGES: dict[str, tuple[float, float]] = {
    "x_left": (0.10, 0.25),
    "x_right": (0.65, 0.82),
    "vx_left": (0.45, 0.72),
    "vx_right": (-0.16, 0.02),
    "y": (0.30, 0.70),
    "radius": (0.04, 0.065),
}

# 2D study: collision-parameterized states (see simulator2d.py). Contact times lie inside the
# window that ``_valid_trajectory`` requires (after the observed prefix, >= 4 frames before end).
STATE_RANGES_2D: dict[str, tuple[float, float]] = {
    "contact_time": (0.34, 0.76),
    "contact_x": (0.40, 0.60),
    "contact_y": (0.35, 0.65),
    "vx0": (0.35, 0.65),
    "vy0": (-0.25, 0.25),
    "vx1": (-0.30, 0.00),
    "vy1": (-0.25, 0.25),
    "impact": (-0.85, 0.85),
    "radius": (0.04, 0.065),
}


@dataclass(frozen=True)
class EventSpec:
    """Shared temporal and rendering settings for every event in one block."""

    observed_frames: int = 8
    future_frames: int = 16
    frame_dt: float = 0.04
    image_size: int = 64
    dims: int = 1

    @property
    def frames(self) -> int:
        return self.observed_frames + self.future_frames


def state_ranges(spec: EventSpec) -> dict[str, tuple[float, float]]:
    """Initial-state parameter ranges for the configured physics (1D pilot or 2D study)."""

    if spec.dims == 1:
        return STATE_RANGES
    if spec.dims == 2:
        return STATE_RANGES_2D
    raise ValueError("dims must be 1 or 2")


def _make_state(spec: EventSpec, values: dict[str, float]) -> InitialState | InitialState2D:
    return (InitialState2D if spec.dims == 2 else InitialState)(**values)


@dataclass(frozen=True)
class Event:
    """One physical trajectory and its initial condition."""

    state: InitialState | InitialState2D
    trajectory: Trajectory


STREAM_ROLES = ("data", "schedule", "base_batches", "torch")


def block_streams(experiment_seed: int, block_index: int) -> dict[str, np.random.SeedSequence]:
    """Return independent, role-separated seed sequences for one training block.

    Streams are derived from ``(experiment_seed, block_index, role)`` through NumPy's
    ``SeedSequence`` spawning, so no stream is shared across blocks or roles.
    """

    if block_index < 0:
        raise ValueError("block_index must be non-negative")
    root = np.random.SeedSequence(entropy=experiment_seed, spawn_key=(block_index,))
    return dict(zip(STREAM_ROLES, root.spawn(len(STREAM_ROLES)), strict=True))


class AppearanceRegistry:
    """Draw appearance seeds that are unique across every render in one block."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._used: set[int] = set()

    def draw(self) -> int:
        seed_value = int(self._rng.integers(0, 2**31 - 1))
        while seed_value in self._used:
            seed_value = int(self._rng.integers(0, 2**31 - 1))
        self._used.add(seed_value)
        return seed_value

    def __len__(self) -> int:
        return len(self._used)


def _valid_trajectory(state: InitialState | InitialState2D, spec: EventSpec) -> Trajectory | None:
    collision_time = state.collision_time()
    earliest = spec.observed_frames * spec.frame_dt
    latest = (spec.frames - 4) * spec.frame_dt
    if collision_time is None or not earliest <= collision_time <= latest:
        return None
    try:
        sim = simulate_2d if isinstance(state, InitialState2D) else simulate
        return sim(state, frames=spec.frames, dt=spec.frame_dt)
    except ValueError:
        return None


def sample_event(rng: np.random.Generator, spec: EventSpec) -> Event:
    """Sample an independent visible collision whose contact follows the observed prefix."""

    for _ in range(10_000):
        state = _make_state(
            spec,
            {name: float(rng.uniform(low, high)) for name, (low, high) in state_ranges(spec).items()},
        )
        trajectory = _valid_trajectory(state, spec)
        if trajectory is not None:
            return Event(state=state, trajectory=trajectory)
    raise RuntimeError("could not sample a valid collision")


def sample_nearby_event(
    rng: np.random.Generator,
    source: Event,
    spec: EventSpec,
    *,
    perturbation: float,
    min_distance: float,
    max_distance: float,
) -> tuple[Event, float]:
    """Independently perturb a source initial condition into a declared distance band.

    Each initial-state parameter is perturbed by a uniform draw scaled to its sampling range.
    The accepted event is a new simulation, not a transformed copy of the source trajectory.
    """

    if not 0 <= min_distance < max_distance:
        raise ValueError("distance band must satisfy 0 <= min < max")
    for _ in range(20_000):
        values = {}
        for name, (low, high) in state_ranges(spec).items():
            width = high - low
            proposed = (
                getattr(source.state, name) + rng.uniform(-perturbation, perturbation) * width
            )
            values[name] = float(np.clip(proposed, low, high))
        state = replace(source.state, **values)
        trajectory = _valid_trajectory(state, spec)
        if trajectory is None:
            continue
        distance = state_trajectory_distance(source.trajectory.positions, trajectory.positions)
        if min_distance <= distance <= max_distance:
            return Event(state=state, trajectory=trajectory), distance
    raise RuntimeError("could not sample a nearby event inside the distance band")


def _pairwise_state_distances(source: FloatArray, candidates: FloatArray) -> FloatArray:
    """RMS state distance from one trajectory to many, minimized over a global object swap."""

    direct = np.sqrt(np.square(candidates - source[None]).mean(axis=(1, 2, 3)))
    swapped = np.sqrt(np.square(candidates[:, :, ::-1] - source[None]).mean(axis=(1, 2, 3)))
    return np.minimum(direct, swapped)


def independent_distance_band(
    rng: np.random.Generator,
    reservoir: list[Event],
    *,
    low_quantile: float,
    high_quantile: float,
    pairs: int = 50_000,
) -> tuple[float, float]:
    """Estimate a state-distance band from random independent reservoir pairs."""

    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    positions = np.stack([event.trajectory.positions for event in reservoir])
    left = rng.integers(0, len(reservoir), size=pairs)
    right = rng.integers(0, len(reservoir), size=pairs)
    keep = left != right
    differences = positions[left[keep]] - positions[right[keep]]
    swapped = positions[left[keep]] - positions[right[keep]][:, :, ::-1]
    distances = np.minimum(
        np.sqrt(np.square(differences).mean(axis=(1, 2, 3))),
        np.sqrt(np.square(swapped).mean(axis=(1, 2, 3))),
    )
    return float(np.quantile(distances, low_quantile)), float(np.quantile(distances, high_quantile))


def select_reservoir_neighbors(
    rng: np.random.Generator,
    sources: list[Event],
    reservoir: list[Event],
    *,
    min_distance: float,
    max_distance: float,
) -> tuple[list[Event], list[float]]:
    """Assign one independently simulated reservoir event per source inside a distance band.

    Candidates are generated independently of the sources and then selected by similarity, so
    neighbors are similarity-selected independent events rather than derived copies. A maximum
    bipartite matching (augmenting paths over randomly ordered candidates) guarantees that each
    reservoir event is used at most once whenever a complete assignment exists.
    """

    positions = np.stack([event.trajectory.positions for event in reservoir])
    eligible: list[NDArray[np.int64]] = []
    distance_rows: list[FloatArray] = []
    for source in sources:
        candidate_distances = _pairwise_state_distances(source.trajectory.positions, positions)
        indices = np.flatnonzero(
            (candidate_distances >= min_distance) & (candidate_distances <= max_distance)
        )
        if indices.size == 0:
            raise RuntimeError("no reservoir event inside the declared distance band")
        eligible.append(rng.permutation(indices))
        distance_rows.append(candidate_distances)

    owner: dict[int, int] = {}

    def assign(source_index: int, visited: set[int]) -> bool:
        for candidate in eligible[source_index]:
            candidate = int(candidate)
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in owner or assign(owner[candidate], visited):
                owner[candidate] = source_index
                return True
        return False

    for source_index in rng.permutation(len(sources)):
        if not assign(int(source_index), set()):
            raise RuntimeError("no complete one-to-one neighbor assignment exists")

    chosen = {source_index: candidate for candidate, source_index in owner.items()}
    neighbors = [reservoir[chosen[index]] for index in range(len(sources))]
    distances = [float(distance_rows[index][chosen[index]]) for index in range(len(sources))]
    return neighbors, distances


def nearest_distances(sources: list[Event], pool: list[Event]) -> FloatArray:
    """Distance from each source to its nearest event in ``pool``."""

    positions = np.stack([event.trajectory.positions for event in pool])
    return np.array(
        [
            _pairwise_state_distances(source.trajectory.positions, positions).min()
            for source in sources
        ]
    )


def _parameter_sensitivities(source: Event, spec: EventSpec, step: float = 0.01) -> FloatArray:
    """Finite-difference state distance per unit normalized offset for each parameter."""

    ranges = state_ranges(spec)
    names = list(ranges)
    sensitivities = np.full(len(names), np.nan)
    for index, name in enumerate(names):
        low, high = ranges[name]
        for direction in (1.0, -1.0):
            value = getattr(source.state, name) + direction * step * (high - low)
            if not low <= value <= high:
                continue
            trajectory = _valid_trajectory(replace(source.state, **{name: float(value)}), spec)
            if trajectory is None:
                continue
            distance = state_trajectory_distance(source.trajectory.positions, trajectory.positions)
            sensitivities[index] = distance / step
            break
    return sensitivities


def closer_neighbor_box(
    source: Event, spec: EventSpec, *, max_distance: float, box_multiplier: float
) -> FloatArray:
    """Proposal half-widths for ``sample_closer_neighbor``, in parameter units.

    Each half-width is ``box_multiplier * max_distance / sensitivity`` times the parameter range,
    capped at the full range. A capped parameter is therefore proposed over its entire range from
    any source value, so insensitive parameters (for example the radius) are never truncated.
    """

    ranges = state_ranges(spec)
    names = list(ranges)
    widths = np.array([ranges[name][1] - ranges[name][0] for name in names])
    sensitivities = _parameter_sensitivities(source, spec)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = box_multiplier * max_distance / sensitivities
    fractions = np.where(np.isfinite(fractions), np.minimum(fractions, 1.0), 1.0)
    return fractions * widths


def sample_closer_neighbor(
    rng: np.random.Generator,
    source: Event,
    spec: EventSpec,
    *,
    min_distance: float,
    max_distance: float,
    box_multiplier: float = 6.0,
    max_proposals: int = 5_000_000,
) -> tuple[Event, float, float]:
    """Draw an independent event from the box-conditioned band distribution around ``source``.

    Target distribution (declared, not assumed): the uniform initial-state prior restricted to the
    proposal box from ``closer_neighbor_box`` and conditioned on the RMS state distance lying in
    ``[min_distance, max_distance]``. Proposals are uniform over the box intersected with the
    ranges (out-of-range proposals are rejected, not clipped) and accepted by distance band, so
    accepted events follow exactly that distribution. It equals the unrestricted conditional
    (reservoir-selection target) only if the box contains every band member; that is checked
    empirically by ``scripts/validate_closer_sampler.py`` rather than assumed. The third return
    value is the largest normalized box offset of the accepted state.
    """

    if not 0 <= min_distance < max_distance:
        raise ValueError("distance band must satisfy 0 <= min < max")
    ranges = state_ranges(spec)
    names = list(ranges)
    lows = np.array([ranges[name][0] for name in names])
    highs = np.array([ranges[name][1] for name in names])
    center = np.array([getattr(source.state, name) for name in names])
    half_widths = closer_neighbor_box(
        source, spec, max_distance=max_distance, box_multiplier=box_multiplier
    )
    for _ in range(max_proposals):
        offset = rng.uniform(-1.0, 1.0, size=len(names))
        proposal = center + offset * half_widths
        if np.any(proposal < lows) or np.any(proposal > highs):
            continue
        state = _make_state(spec, dict(zip(names, map(float, proposal), strict=True)))
        trajectory = _valid_trajectory(state, spec)
        if trajectory is None:
            continue
        distance = state_trajectory_distance(source.trajectory.positions, trajectory.positions)
        if min_distance <= distance <= max_distance:
            return Event(state=state, trajectory=trajectory), distance, float(np.abs(offset).max())
    raise RuntimeError("no proposal inside the distance band")


def render_event(event: Event, spec: EventSpec, appearance_seed: int) -> UInt8Array:
    """Render the observed prefix of one event under one appearance seed."""

    pixels = render(
        event.trajectory,
        appearance_seed=appearance_seed,
        size=spec.image_size,
        radius_pixels=max(2, round(event.state.radius * spec.image_size)),
    )
    return pixels[: spec.observed_frames]


def matched_arm_schedule(
    rng: np.random.Generator,
    *,
    steps: int,
    arm_per_batch: int,
    replay_per_batch: int,
    arm_examples: int,
    replay_examples: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return one batch-index schedule shared by every continuation arm.

    Arm indices cycle through balanced permutations so each arm-specific example receives the
    same presentation count, up to one presentation when the total is not divisible.
    """

    if min(steps, arm_per_batch, arm_examples, replay_examples) <= 0 or replay_per_batch < 0:
        raise ValueError("schedule sizes must be positive")
    total = steps * arm_per_batch
    permutations = math.ceil(total / arm_examples)
    arm_indices = np.concatenate([rng.permutation(arm_examples) for _ in range(permutations)])
    arm_schedule = arm_indices[:total].reshape(steps, arm_per_batch)
    replay_schedule = rng.integers(0, replay_examples, size=(steps, replay_per_batch))
    return arm_schedule.astype(np.int64), replay_schedule.astype(np.int64)


def reuse_contrasts(errors: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute block-level contrasts from arm-by-population mean errors.

    Positive gain means lower error than control arm B.
    """

    def gain(arm: str, population: str) -> float:
        return errors["B"][population] - errors[arm][population]

    contrasts: dict[str, float] = {}
    for arm in [name for name in ("C", "D", "D2", "E") if name in errors]:
        contrasts[f"{arm}_source_gain"] = gain(arm, "source")
        contrasts[f"{arm}_independent_gain"] = gain(arm, "independent")
        contrasts[f"{arm}_reuse_gap"] = gain(arm, "source") - gain(arm, "independent")
    contrasts["C_independent_harm"] = -contrasts["C_independent_gain"]
    contrasts["C_minus_D_source"] = errors["D"]["source"] - errors["C"]["source"]
    contrasts["C_minus_D_gap"] = contrasts["C_reuse_gap"] - contrasts["D_reuse_gap"]
    if "D2" in errors:
        contrasts["C_minus_D2_gap"] = contrasts["C_reuse_gap"] - contrasts["D2_reuse_gap"]
    for arm in ("B", "C"):
        contrasts[f"{arm}_apparent_improvement"] = (
            errors[arm]["independent"] - errors[arm]["source"]
        )
    if "exact" in errors["E"]:
        contrasts["E_exact_gain"] = errors["B"]["exact"] - errors["E"]["exact"]
        contrasts["E_transfer_loss"] = errors["E"]["source"] - errors["E"]["exact"]
    return contrasts


def chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Wilson–Hilferty approximation to a chi-square quantile."""

    if not 0 < probability < 1 or degrees_of_freedom <= 0:
        raise ValueError("invalid chi-square quantile request")
    z_value = NormalDist().inv_cdf(probability)
    k = float(degrees_of_freedom)
    base = 1 - 2 / (9 * k) + z_value * math.sqrt(2 / (9 * k))
    if base <= 0:
        raise ValueError("Wilson-Hilferty approximation is invalid for this request")
    return k * base**3


def standard_deviation_upper_limit(
    standard_deviation: float, blocks: int, confidence: float = 0.8
) -> float:
    """One-sided upper confidence limit for a normal standard deviation."""

    if blocks < 2:
        raise ValueError("at least two blocks are required")
    if not math.isfinite(standard_deviation) or standard_deviation < 0:
        raise ValueError("standard deviation must be finite and non-negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    degrees = blocks - 1
    return standard_deviation * math.sqrt(degrees / chi_square_quantile(1 - confidence, degrees))


def blinded_summary(values: list[float], *, confidence: float = 0.8) -> dict[str, float | int]:
    """Summarize dispersion without reporting the location of a contrast."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one block-level contrast per block")
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("need at least two finite values")
    standard_deviation = float(array.std(ddof=1))
    upper = standard_deviation_upper_limit(standard_deviation, int(array.size), confidence)

    def released(value: float) -> float:
        # Rounding removes floating-point residue that could otherwise depend on location.
        return float(f"{value:.9g}")

    return {
        "blocks": int(array.size),
        "standard_deviation": released(standard_deviation),
        "standard_deviation_upper_limit": released(upper),
        "upper_limit_confidence": confidence,
    }
