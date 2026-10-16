"""Analyze saved per-query detector scores: calibration, uncertainty, gates, and constrained recall.

Inputs come from ``scripts/detector_benchmark.py`` (``*_scores.json.gz``). Event families are
shared across transformations within a role, so family bootstrap resamples indices jointly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROLES = ("positive", "hard", "easy")


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return [center - half, center + half]


def finite(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, -1e30)


def threshold(scores: np.ndarray, alpha: float) -> float:
    return float(np.quantile(finite(scores), 1 - alpha, method="higher"))


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive, negative = finite(positive), finite(negative)
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    n_pos = len(positive)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def arrays(split: dict, transform: str, role: str, detector: str, key: str = "score") -> np.ndarray:
    return np.asarray(split["scores"][transform][role][detector][key], dtype=np.float64)


def analyze(data: dict[str, Any], alphas: list[float], draws: int, seed: int) -> dict[str, Any]:
    calibration, test = data["splits"]["calibration"], data["splits"]["test"]
    transforms = list(test["scores"])
    detectors = list(test["scores"][transforms[0]]["positive"])
    rng = np.random.default_rng(seed)
    n_cal_easy = len(arrays(calibration, transforms[0], "easy", detectors[0]))
    n_test = {role: len(arrays(test, transforms[0], role, detectors[0])) for role in ROLES}
    cal_easy_draws = [rng.integers(0, n_cal_easy, n_cal_easy) for _ in range(draws)]
    test_draws = [
        {role: rng.integers(0, n_test[role], n_test[role]) for role in ROLES} for _ in range(draws)
    ]
    output: dict[str, Any] = {
        "independent_families": {
            "calibration_easy": n_cal_easy,
            **{f"test_{role}": count for role, count in n_test.items()},
        },
        "detectors": {},
    }
    for detector in detectors:
        cal_easy = {t: arrays(calibration, t, "easy", detector) for t in transforms}
        cal_hard = {t: arrays(calibration, t, "hard", detector) for t in transforms}
        pooled_easy = np.concatenate([cal_easy[t] for t in transforms])
        pooled_hard = np.concatenate([cal_hard[t] for t in transforms])
        scores = {t: {role: arrays(test, t, role, detector) for role in ROLES} for t in transforms}
        ranks = {t: arrays(test, t, "positive", detector, "rank") for t in transforms}
        entry: dict[str, Any] = {"operating_points": {}, "transforms": {}}
        for alpha in alphas:
            tau = threshold(pooled_easy, alpha)
            tau_constrained = max(tau, threshold(pooled_hard, alpha))
            boot_tau = [
                threshold(np.concatenate([cal_easy[t][index] for t in transforms]), alpha)
                for index in cal_easy_draws
            ]
            entry["operating_points"][str(alpha)] = {
                "threshold": tau,
                "threshold_with_hard_constraint": tau_constrained,
            }
            for t in transforms:
                row = entry["transforms"].setdefault(
                    t,
                    {
                        "auroc_vs_easy": auroc(scores[t]["positive"], scores[t]["easy"]),
                        "auroc_vs_hard": auroc(scores[t]["positive"], scores[t]["hard"]),
                        "median_rank": float(np.median(ranks[t])),
                        "strict_top1": float(np.mean(ranks[t] <= 1.0)),
                    },
                )
                point: dict[str, Any] = {}
                for role in ROLES:
                    removed = finite(scores[t][role]) > tau
                    point[role] = {
                        "rate": float(removed.mean()),
                        "count": int(removed.sum()),
                        "wilson": wilson(int(removed.sum()), len(removed)),
                    }
                    boot = [
                        float((finite(scores[t][role][draw[role]]) > boot_threshold).mean())
                        for draw, boot_threshold in zip(test_draws, boot_tau, strict=True)
                    ]
                    point[role]["bootstrap_95"] = [
                        float(np.quantile(boot, 0.025)),
                        float(np.quantile(boot, 0.975)),
                    ]
                constrained = {
                    role: float((finite(scores[t][role]) > tau_constrained).mean())
                    for role in ROLES
                }
                point["with_hard_constraint"] = constrained
                row[str(alpha)] = point
        copy = entry["transforms"].get("copy_reencode")
        if copy is not None:
            entry["copy_control_passed"] = bool(
                copy["auroc_vs_easy"] >= 0.95 and copy["0.05"]["positive"]["rate"] >= 0.80
            )
        output["detectors"][detector] = entry
    if "oracle_state" in detectors:
        exact = {}
        for t in transforms:
            exact[t] = {
                role: float((finite(arrays(test, t, role, "oracle_state")) > -1e-6).mean())
                for role in ROLES
            }
        output["state_distance_zero_threshold"] = exact
    return output


def markdown(result: dict[str, Any], alpha: str = "0.01") -> str:
    detectors = result["detectors"]
    transforms = list(next(iter(detectors.values()))["transforms"])
    lines = [
        f"Recall / easy-negative false removal at the pooled {alpha} operating point (test split)",
        "",
        "| Detector | Copy gate | " + " | ".join(transforms) + " |",
        "|---|---|" + "---|" * len(transforms),
    ]
    for name, entry in detectors.items():
        cells = []
        for t in transforms:
            point = entry["transforms"][t][alpha]
            cells.append(f"{point['positive']['rate']:.2f} / {point['easy']['rate']:.3f}")
        gate = "pass" if entry.get("copy_control_passed") else "fail"
        lines.append(f"| {name} | {gate} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path, default=Path("results/detector_benchmark_v3_scores.json.gz")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/detector_benchmark_v3_analysis.json")
    )
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    result = analyze(load(args.scores), [0.01, 0.05], args.draws, seed=20261016)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    table = markdown(result)
    args.out.with_suffix(".md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
