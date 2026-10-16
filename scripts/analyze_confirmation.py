"""Frozen confirmatory analysis for the matched trajectory-reuse experiment.

Implements the decision contract in ``experiments/exposure_confirmation.json``. The SHA-256 of this
file is bound into the freeze manifest, so it cannot change after sign-off without failing the
runner's unblinding check.

Outcomes are reported on separate axes so that a failed safeguard is never relabeled as a
prevalence effect:

* gap magnitude (consequential, prevalence-dependent, negligible, reverse, inconclusive);
* safeguards (source gain supported, independent harm bounded, consequential harm);
* robustness: decisive claims must survive least-favorable imputation of failed blocks.
* identity specificity (same trajectory versus closer independent neighbor), Holm-adjusted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from freeze import FreezeError  # noqa: E402
from freeze import validate as validate_freeze  # noqa: E402
from plan_confirmation import t_cdf, t_quantile  # noqa: E402

ARMS = ("B", "C", "D", "D2", "E")
POPULATIONS = ("source", "exact", "independent")
STATUSES = ("completed", "algorithmic_failure", "infrastructure_failure")

FIELDS = {
    "S": "C_source_gain",
    "I": "C_independent_gain",
    "H": "C_independent_harm",
    "G": "C_reuse_gap",
    "Q": "C_minus_D_gap",
    "Q2": "C_minus_D2_gap",
    "A_C": "C_apparent_improvement",
    "A_B": "B_apparent_improvement",
    "E_exact_gain": "E_exact_gain",
    "E_transfer_loss": "E_transfer_loss",
}


class RecordError(Exception):
    """Raised when block records do not form exactly the registered experiment."""


def validate_records(runs: list[dict[str, Any]], scheduled: int) -> None:
    """Require exactly one terminal, internally consistent record per registered block."""

    indices = [run.get("block_index") for run in runs]
    if sorted(indices) != list(range(scheduled)):
        raise RecordError(
            f"expected block indices 0..{scheduled - 1} exactly once; got {len(runs)} records"
        )
    for run in runs:
        if run.get("status") not in STATUSES:
            raise RecordError(f"block {run.get('block_index')} has unknown status")
        if run["status"] != "completed":
            continue
        errors = run["errors"]
        values = [errors[arm][population] for arm in ARMS for population in POPULATIONS]
        values += list(run["contrasts"].values())
        if not all(math.isfinite(float(value)) for value in values):
            raise RecordError(f"block {run['block_index']} has nonfinite values")
        if any(not 0 <= errors[arm][population] <= 1 for arm in ARMS for population in POPULATIONS):
            raise RecordError(f"block {run['block_index']} has errors outside the bounded metric")
        gap = (errors["B"]["source"] - errors["C"]["source"]) - (
            errors["B"]["independent"] - errors["C"]["independent"]
        )
        if abs(gap - run["contrasts"]["C_reuse_gap"]) > 1e-9:
            raise RecordError(f"block {run['block_index']} contrasts disagree with its errors")


def fieller(numerator: np.ndarray, denominator: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fieller confidence set for mean(numerator) / mean(denominator) over blocks."""

    n = len(numerator)
    covariance = np.cov(numerator, denominator, ddof=1) / n
    h, g = float(numerator.mean()), float(denominator.mean())
    t2 = t_quantile(1 - alpha, n - 1) ** 2
    a = g * g - t2 * covariance[1, 1]
    b = -2 * (h * g - t2 * covariance[0, 1])
    c = h * h - t2 * covariance[0, 0]
    discriminant = b * b - 4 * a * c
    if a > 0:
        root = math.sqrt(max(discriminant, 0.0))
        return {"type": "bounded", "interval": [(-b - root) / (2 * a), (-b + root) / (2 * a)]}
    if discriminant > 0:
        root = math.sqrt(discriminant)
        low, high = sorted([(-b - root) / (2 * a), (-b + root) / (2 * a)])
        return {"type": "two rays", "excluded_interval": [low, high]}
    return {"type": "entire real line"}


def interval(values: np.ndarray, alpha_one_sided: float) -> dict[str, float]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    crit = t_quantile(1 - alpha_one_sided, n - 1)
    return {"n": n, "mean": mean, "sd": sd, "lower": mean - crit * se, "upper": mean + crit * se}


def one_sided_p(values: np.ndarray, margin: float) -> float:
    """p-value for H0: mean <= margin against mean > margin."""

    n = len(values)
    se = float(values.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        return 0.0 if values.mean() > margin else 1.0
    return 1.0 - t_cdf((float(values.mean()) - margin) / se, n - 1)


def holm(p_values: dict[str, float], alpha: float) -> dict[str, bool]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {}
    rejecting = True
    for rank, (name, value) in enumerate(ordered):
        rejecting = rejecting and value <= alpha / (len(ordered) - rank)
        decisions[name] = rejecting
    return decisions


def break_even(gap: dict[str, float], margin: float) -> dict[str, Any]:
    """Prevalence at which reuse alone moves C's advantage over B by ``margin``."""

    lower, upper = gap["lower"], gap["upper"]
    if upper <= 0:
        return {"status": "no positive crossing"}
    low_end = margin / upper
    high_end = math.inf if lower <= 0 else margin / lower
    return {
        "status": "finite" if lower > 0 else "unbounded above",
        "interval": [low_end, high_end],
        "within_unit_interval": [min(low_end, 1.0), min(high_end, 1.0)] if low_end <= 1 else None,
        "point": margin / gap["mean"] if gap["mean"] > 0 else None,
    }


def classify_gap(gap: dict[str, float], contract: dict[str, Any]) -> str:
    consequential = float(contract["consequential_threshold"])
    negligible = float(contract["negligible_bound"])
    if gap["lower"] > consequential:
        return "consequential"
    if gap["lower"] > negligible:
        return "prevalence_dependent"
    if gap["lower"] > -negligible and gap["upper"] < negligible:
        return "negligible"
    if gap["upper"] < -negligible:
        return "reverse"
    return "inconclusive"


def arm_summaries(runs: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    return {
        arm: {
            population: interval(np.array([run["errors"][arm][population] for run in runs]), alpha)
            for population in POPULATIONS
        }
        for arm in ARMS
    }


def analyze_blocks(
    runs: list[dict[str, Any]],
    contract: dict[str, Any],
    rng: np.random.Generator,
    failed: int = 0,
) -> dict[str, Any]:
    alpha = float(contract["alpha_one_sided"])
    values = {
        name: np.array([run["contrasts"][field] for run in runs])
        for name, field in FIELDS.items()
        if all(field in run["contrasts"] for run in runs)
    }
    summaries = {name: interval(array, alpha) for name, array in values.items()}
    gap = summaries["G"]
    magnitude = classify_gap(gap, contract)
    harm_margin = float(contract["harm_margin_h"])
    source_supported = summaries["S"]["lower"] > 0
    harm_bounded = summaries["H"]["upper"] < harm_margin
    assay_sensitive = float(np.sum(values["E_exact_gain"] > 0)) / (len(runs) + failed) >= 0.75

    if magnitude in ("consequential", "prevalence_dependent"):
        if source_supported and harm_bounded:
            claim = f"{magnitude} reuse advantage"
        else:
            claim = "differential gap without safeguards"
    elif magnitude == "negligible":
        claim = "negligible differential gap" if assay_sensitive else "assay insensitive"
    elif magnitude == "reverse":
        claim = "reverse gap"
    else:
        claim = "inconclusive"

    p_values = {
        "identity_specific": one_sided_p(values["Q"], float(contract["consequential_threshold"])),
        "detectable": one_sided_p(values["G"], 0.0),
        "consequential_harm": one_sided_p(values["H"], harm_margin),
    }
    margin = float(contract["ranking_margin_m"])
    order = np.argsort(values["G"])
    trim = int(0.1 * len(order))
    trimmed = values["G"][order[trim : len(order) - trim]] if trim else values["G"]
    leave_one_out = [
        float(np.delete(values["G"], index).mean()) for index in range(len(values["G"]))
    ]
    return {
        "blocks": len(runs),
        "arm_population_errors": arm_summaries(runs, alpha),
        "contrasts": summaries,
        "gap_magnitude": magnitude,
        "source_gain_supported": bool(source_supported),
        "harm_bounded": bool(harm_bounded),
        "assay_sensitive": bool(assay_sensitive),
        "primary_claim": claim,
        "secondary_p_values": p_values,
        "secondary_holm_rejections": holm(p_values, alpha),
        "break_even_prevalence": break_even(gap, margin),
        "harm_to_gap_crossover_fieller": fieller(values["H"], values["G"], alpha),
        "sensitivity": {
            "trimmed_mean_G_10pct": float(trimmed.mean()),
            "leave_one_block_out_G_range": [min(leave_one_out), max(leave_one_out)],
        },
    }


def worst_case_records(
    runs: list[dict[str, Any]], failed: int, direction: str
) -> list[dict[str, Any]]:
    """Append imputed records for failed blocks using least favorable observed contrasts."""

    if failed == 0:
        return runs
    contrasts = {field: [run["contrasts"][field] for run in runs] for field in FIELDS.values()}
    worst = {
        field: (min(values) if direction == "low" else max(values))
        for field, values in contrasts.items()
    }
    worst["C_independent_harm"] = max(contrasts["C_independent_harm"])
    worst["C_independent_gain"] = -worst["C_independent_harm"]
    worst["E_exact_gain"] = min(contrasts["E_exact_gain"])
    imputed = [
        {"block_index": -1, "status": "imputed", "contrasts": dict(worst)} for _ in range(failed)
    ]
    for record in imputed:
        record["errors"] = runs[0]["errors"]
    return runs + imputed


def analyze(
    runs: list[dict[str, Any]], config: dict[str, Any], *, check_records: bool = True
) -> dict[str, Any]:
    contract = config["decision_contract"]
    scheduled = int(config["blocks"])
    if check_records:
        validate_records(runs, scheduled)
    completed = [run for run in runs if run["status"] == "completed"]
    failures = [run for run in runs if run["status"] != "completed"]
    screened = [run for run in completed if run.get("base_competent", False)]
    rng = np.random.default_rng(int(config["experiment_seed"]))
    max_failure = float(contract.get("max_failure_fraction", 0.05))
    minimum = int(contract.get("minimum_screened_blocks", 2))
    result: dict[str, Any] = {
        "scheduled": scheduled,
        "completed": len(completed),
        "failures": {
            "algorithmic": [
                run["block_index"] for run in failures if "algorithmic" in run["status"]
            ],
            "infrastructure": [
                run["block_index"] for run in failures if "infrastructure" in run["status"]
            ],
        },
        "screened": len(screened),
        "screen_failures": [run["block_index"] for run in completed if run not in screened],
    }
    if len(screened) >= 2:
        result["primary"] = analyze_blocks(screened, contract, rng, failed=len(failures))
    if len(completed) >= 2:
        result["all_completed_blocks"] = analyze_blocks(
            completed, contract, rng, failed=len(failures)
        )
    if len(failures) > max_failure * scheduled:
        result["primary_claim"] = "inconclusive (execution failures exceed the declared limit)"
        return result
    if len(screened) < minimum:
        result["primary_claim"] = "inconclusive (too few screened blocks)"
        return result
    claim = result["primary"]["primary_claim"]
    if failures:
        robust = {
            analyze_blocks(
                worst_case_records(screened, len(failures), direction),
                contract,
                rng,
            )["primary_claim"]
            for direction in ("low", "high")
        }
        result["failure_robustness"] = sorted(robust)
        if robust != {claim}:
            claim = "inconclusive (failure-sensitive)"
    result["primary_claim"] = claim
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records", type=Path, required=True, help="Raw block records (unblinded)."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/exposure_confirmation.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_freeze(json.loads(args.freeze_manifest.read_text()), args.config, Path(__file__))
    except FreezeError as error:
        raise SystemExit(f"refusing to analyze: {error}") from error
    result = analyze(json.loads(args.records.read_text()), json.loads(args.config.read_text()))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("primary_claim", "scheduled", "screened")}))


if __name__ == "__main__":
    main()
