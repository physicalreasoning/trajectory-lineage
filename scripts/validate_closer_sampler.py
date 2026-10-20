"""Archive empirical checks of the closer-neighbor sampler's box-containment assumption.

1. Box enlargement: accepted neighbors under multipliers 6 and 9 should have matching
   per-parameter offset distributions if the multiplier-6 box already contains the band region.
2. Reservoir agreement: on a band that a 40,000-event reservoir can reach, box sampling and
   reservoir selection should give matching accepted-distance distributions.
3. Capped parameters: fraction of proposals whose half-width is the full parameter range.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trajectory_lineage.exposure import (
    STATE_RANGES,
    EventSpec,
    _pairwise_state_distances,
    closer_neighbor_box,
    nearest_distances,
    sample_closer_neighbor,
    sample_event,
)


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(np.sort(left), values, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def ks_critical(left: int, right: int, alpha: float = 0.05) -> float:
    coefficient = {0.05: 1.358, 0.01: 1.628}[alpha]
    return coefficient * float(np.sqrt((left + right) / (left * right)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=30)
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/closer_sampler_validation.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    spec = EventSpec()
    names = list(STATE_RANGES)
    widths = np.array([STATE_RANGES[name][1] - STATE_RANGES[name][0] for name in names])
    rng = np.random.default_rng(np.random.SeedSequence(20261016))

    base = [sample_event(rng, spec) for _ in range(512)]
    offsets: dict[float, list[np.ndarray]] = {6.0: [], 9.0: []}
    ratios: dict[float, list[float]] = {6.0: [], 9.0: []}
    capped = []
    for _ in range(args.sources):
        source = sample_event(rng, spec)
        reference = float(nearest_distances([source], base)[0])
        center = np.array([getattr(source.state, name) for name in names])
        half = closer_neighbor_box(source, spec, max_distance=0.5 * reference, box_multiplier=6.0)
        capped.append(np.isclose(half, widths))
        for multiplier in offsets:
            for _ in range(args.per_source):
                neighbor, distance, _ = sample_closer_neighbor(
                    rng,
                    source,
                    spec,
                    min_distance=0.25 * reference,
                    max_distance=0.5 * reference,
                    box_multiplier=multiplier,
                )
                state = np.array([getattr(neighbor.state, name) for name in names])
                offsets[multiplier].append(np.abs(state - center) / widths)
                ratios[multiplier].append(distance / reference)
    enlargement = {}
    for index, name in enumerate(names):
        six = np.array([row[index] for row in offsets[6.0]])
        nine = np.array([row[index] for row in offsets[9.0]])
        enlargement[name] = {
            "mean_offset_fraction": [float(six.mean()), float(nine.mean())],
            "q95_offset_fraction": [float(np.quantile(six, 0.95)), float(np.quantile(nine, 0.95))],
            "ks": ks_statistic(six, nine),
            "ks_critical_0.05": ks_critical(len(six), len(nine)),
        }

    reservoir = [sample_event(rng, spec) for _ in range(40_000)]
    positions = np.stack([event.trajectory.positions for event in reservoir])
    low, high = 0.012, 0.016
    from_reservoir, from_box = [], []
    for _ in range(120):
        source = sample_event(rng, spec)
        distances = _pairwise_state_distances(source.trajectory.positions, positions)
        eligible = distances[(distances >= low) & (distances <= high)]
        if eligible.size:
            from_reservoir.append(float(rng.choice(eligible)))
        _, distance, _ = sample_closer_neighbor(
            rng, source, spec, min_distance=low, max_distance=high
        )
        from_box.append(distance)
    reservoir_check = {
        "band": [low, high],
        "reservoir_mean_sd": [float(np.mean(from_reservoir)), float(np.std(from_reservoir))],
        "box_mean_sd": [float(np.mean(from_box)), float(np.std(from_box))],
        "ks": ks_statistic(np.array(from_reservoir), np.array(from_box)),
        "ks_critical_0.05": ks_critical(len(from_reservoir), len(from_box)),
    }
    result = {
        "box_enlargement_6_vs_9": enlargement,
        "distance_over_reference_mean": [float(np.mean(ratios[6.0])), float(np.mean(ratios[9.0]))],
        "reservoir_agreement": reservoir_check,
        "capped_fraction_by_parameter": dict(
            zip(names, np.mean(capped, axis=0).astype(float).tolist(), strict=True)
        ),
        "samples_per_multiplier": len(offsets[6.0]),
        "seconds": time.perf_counter() - started,
        "interpretation": "Containment is supported when no enlargement KS statistic exceeds its "
        "critical value and mean offsets agree; it is an empirical check, not a proof.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
