"""Exploratory: how much exposure-induced evaluation distortion survives duplicate filtering.

Specification (committed before any confirmation result was inspected):

* Blocks: the first ``--blocks`` (default 32) completed, competence-screened confirmation blocks.
* Each block is reconstructed exactly from its SeedSequence streams. Gallery = the C arm's
  training clips (512 base events + 64 source rerenders); queries = the 256 held-out source
  renders and 512 independent events. Clips are full 24-frame renders at 224 px with the same
  appearance seeds as the 64 px model inputs.
* Detectors: SSCD (time-aligned, background-normalized; passes the copy gate) and centroid tracks
  with mirror and time-offset search. Query score = maximum over the gallery.
* Thresholds: 1% and 5% false removal of independent queries, calibrated on separate blocks
  generated from a different experiment seed (data only; no training).
* One mask per block and detector is applied identically to every arm's per-item losses.
* Per block: retained source and independent fractions r_S and r_I; residual gap G_tau from
  retained items; prevalence-weighted distortion D_tau = pi_tau * G_tau with
  pi_tau = pi r_S / (pi r_S + (1 - pi) r_I) at pi = 0.10 (zero contribution when no source item
  survives); unfiltered D_0 = pi * G; difficulty shift = mean B loss of retained minus all
  independent items.
* Baselines: random removal of the same number of items in each population (matched counts)
  and provenance exclusion (all source items removed, D = 0).
* Uncertainty: 2,000 block-bootstrap draws. Results are exploratory, not confirmatory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import detector_benchmark as bench  # noqa: E402
from exposure_experiment import build_block, spec_from  # noqa: E402

from trajectory_lineage.detectors import centroid_motion_descriptor  # noqa: E402
from trajectory_lineage.exposure import block_streams, sample_event  # noqa: E402

PREVALENCE = 0.10
RENDER = {"image_size": 224, "clip_frames": 24, "time_shift_frames": 3}
FRAMES = [0, 3, 6, 9, 12, 15, 18, 21]


def clips(events, seeds) -> np.ndarray:
    rng = np.random.default_rng(0)  # unused for untransformed renders
    return np.stack(
        [
            bench.render_clip(event, int(seed), None, rng, RENDER)
            for event, seed in zip(events, seeds, strict=True)
        ]
    )


def describe(clip_array: np.ndarray, encoders) -> dict[str, np.ndarray]:
    frames = clip_array[:, FRAMES].reshape(-1, *clip_array.shape[2:])
    sscd = bench.frame_embeddings(encoders.sscd(), frames, encoders.device)
    tracks = np.full((len(clip_array), clip_array.shape[1] * 4), np.nan, dtype=np.float32)
    for row, clip in enumerate(clip_array):
        try:
            tracks[row] = centroid_motion_descriptor(clip).ravel()
        except ValueError:
            pass
    return {"sscd": sscd.reshape(len(clip_array), len(FRAMES), -1), "tracks": tracks}


def describe_chunked(events, seeds, encoders, chunk: int = 48) -> dict[str, np.ndarray]:
    parts = []
    for start in range(0, len(events), chunk):
        parts.append(
            describe(clips(events[start : start + chunk], seeds[start : start + chunk]), encoders)
        )
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def block_scores(block: dict, background: dict, encoders) -> dict[str, dict[str, np.ndarray]]:
    events, seeds = block["events"], block["render_seeds"]
    gallery = describe_chunked(
        events["base"] + events["sources"], seeds["base"] + seeds["rerender"], encoders
    )
    renders = int(len(seeds["heldout"]) / len(events["sources"]))
    heldout_events = [event for event in events["sources"] for _ in range(renders)]
    source = describe_chunked(heldout_events, seeds["heldout"], encoders)
    independent = describe_chunked(events["independent"], seeds["independent"], encoders)
    scores = {}
    for name, query in (("source", source), ("independent", independent)):
        sscd = bench.cosine_aligned_matrix(query["sscd"], gallery["sscd"])
        normalized = bench.background_normalized(
            sscd, bench.cosine_aligned_matrix(query["sscd"], background["sscd"]), 10
        )
        tracks = bench.invariant_track_matrix(query["tracks"], gallery["tracks"], 3, 224)
        scores[name] = {
            "sscd_aligned_norm": normalized.max(dim=1).values.double().numpy(),
            "centroid_invariant": tracks.max(dim=1).values.double().numpy(),
        }
    return scores


def residual(run: dict, keep_source: np.ndarray, keep_independent: np.ndarray) -> dict:
    items = run["item_losses"]
    r_s, r_i = float(keep_source.mean()), float(keep_independent.mean())
    result = {"r_S": r_s, "r_I": r_i}
    if r_i == 0:
        result.update({"G": None, "pi_tau": None, "D": None})
        return result

    def mean(arm: str, population: str, keep: np.ndarray) -> float:
        return float(np.asarray(items[arm][population])[keep].mean())

    independent_gain = mean("B", "independent", keep_independent) - mean(
        "C", "independent", keep_independent
    )
    if r_s == 0:
        result.update({"G": None, "pi_tau": 0.0, "D": 0.0})
    else:
        source_gain = mean("B", "source", keep_source) - mean("C", "source", keep_source)
        gap = source_gain - independent_gain
        pi_tau = PREVALENCE * r_s / (PREVALENCE * r_s + (1 - PREVALENCE) * r_i)
        result.update({"G": gap, "pi_tau": pi_tau, "D": pi_tau * gap})
    b_independent = np.asarray(items["B"]["independent"])
    result["difficulty_shift"] = float(
        b_independent[keep_independent].mean() - b_independent.mean()
    )
    return result


def bootstrap(values: list[float], rng: np.random.Generator, draws: int = 2000) -> list[float]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if array.size == 0:
        return [float("nan"), float("nan")]
    means = [array[rng.integers(0, len(array), len(array))].mean() for _ in range(draws)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def calibrate(config: dict, spec, background: dict, encoders, seed: int, count: int) -> dict:
    pooled: dict[str, list[np.ndarray]] = {"sscd_aligned_norm": [], "centroid_invariant": []}
    for index in range(count):
        streams = block_streams(seed, index)
        block = build_block(config, np.random.default_rng(streams["data"]), spec)
        scores = block_scores(block, background, encoders)["independent"]
        for name in pooled:
            pooled[name].append(scores[name])
    return {
        name: {
            str(alpha): float(np.quantile(np.concatenate(values), 1 - alpha, method="higher"))
            for alpha in (0.01, 0.05)
        }
        for name, values in pooled.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/exposure_confirmation.json")
    )
    parser.add_argument(
        "--records", type=Path, default=Path("results/exposure_confirmation_raw.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/exposure_filtering.json"))
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=2026101699)
    parser.add_argument("--calibration-blocks", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    spec = spec_from(config)
    export = json.loads(args.records.read_text())
    runs = export["runs"] if isinstance(export, dict) else export
    eligible = [
        run
        for run in sorted(runs, key=lambda run: run["block_index"])
        if run["status"] == "completed"
        and run["screen_base_error"] < run["screen_constant_velocity_error"]
    ][: args.blocks]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoders = bench.LearnedEncoders(device)
    background_rng = np.random.default_rng(np.random.SeedSequence(args.calibration_seed + 1))
    background_events = [sample_event(background_rng, spec) for _ in range(500)]
    background_seeds = [int(background_rng.integers(0, 2**31 - 1)) for _ in background_events]
    background = describe_chunked(background_events, background_seeds, encoders)
    thresholds = calibrate(
        config, spec, background, encoders, args.calibration_seed, args.calibration_blocks
    )
    rng = np.random.default_rng(20261016)
    per_block = []
    for run in eligible:
        streams = block_streams(int(config["experiment_seed"]), int(run["block_index"]))
        block = build_block(config, np.random.default_rng(streams["data"]), spec)
        scores = block_scores(block, background, encoders)
        n_source = len(scores["source"]["sscd_aligned_norm"])
        n_independent = len(scores["independent"]["sscd_aligned_norm"])
        everything = residual(
            run, np.ones(n_source, dtype=bool), np.ones(n_independent, dtype=bool)
        )
        record: dict[str, Any] = {"block_index": run["block_index"], "unfiltered": everything}
        record["unfiltered"]["D"] = PREVALENCE * everything["G"]
        for name, by_alpha in thresholds.items():
            for alpha, tau in by_alpha.items():
                keep_s = scores["source"][name] <= tau
                keep_i = scores["independent"][name] <= tau
                random_s = np.ones(n_source, dtype=bool)
                random_i = np.ones(n_independent, dtype=bool)
                random_s[rng.choice(n_source, int((~keep_s).sum()), replace=False)] = False
                random_i[rng.choice(n_independent, int((~keep_i).sum()), replace=False)] = False
                record[f"{name}@{alpha}"] = residual(run, keep_s, keep_i)
                record[f"random_matched_{name}@{alpha}"] = residual(run, random_s, random_i)
        record["provenance_exclusion"] = residual(
            run, np.zeros(n_source, dtype=bool), np.ones(n_independent, dtype=bool)
        )
        per_block.append(record)
        print(f"filtering block {run['block_index']} done", flush=True)

    conditions = [key for key in per_block[0] if key != "block_index"]
    summary = {}
    for condition in conditions:
        rows = [record[condition] for record in per_block]
        summary[condition] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows if row.get(metric) is not None]))
                if any(row.get(metric) is not None for row in rows)
                else None,
                "bootstrap_95": bootstrap([row.get(metric) for row in rows], rng),
            }
            for metric in ("r_S", "r_I", "G", "pi_tau", "D", "difficulty_shift")
        }
    result = {
        "status": "exploratory",
        "specification": __doc__,
        "blocks": [record["block_index"] for record in per_block],
        "prevalence": PREVALENCE,
        "thresholds": thresholds,
        "summary": summary,
        "per_block": per_block,
        "seconds": time.perf_counter() - started,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key]["D"] for key in summary}, indent=2))


if __name__ == "__main__":
    main()
