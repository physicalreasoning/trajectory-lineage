"""Frozen confirmatory analysis for the matched trajectory-reuse experiment.

Implements the decision contract in ``experiments/exposure_confirmation.json``. The SHA-256 of this
file is bound into the freeze manifest, so it cannot change after sign-off without failing the
runner's unblinding check.

Outcomes are reported on separate axes so that a failed safeguard is never relabeled as a
prevalence effect:

* gap magnitude (consequential, prevalence-dependent, negligible, reverse, inconclusive);
* safeguards (source gain supported, independent harm bounded, consequential harm);
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

from plan_confirmation import t_cdf, t_quantile  # noqa: E402

CONTRASTS = ("S", "I", "H", "G", "Q", "Q2", "A_C", "E_exact_gain")
FIELDS = {
    "S": "C_source_gain",
    "I": "C_independent_gain",
    "H": "C_independent_harm",
    "G": "C_reuse_gap",
    "Q": "C_minus_D_gap",
    "Q2": "C_minus_D2_gap",
    "A_C": "C_apparent_improvement",
    "E_exact_gain": "E_exact_gain",
}


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


def ratio_interval(
    numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator, draws: int = 2000
) -> dict[str, Any]:
    """Block-bootstrap percentile interval for mean(H)/mean(G) where defined."""

    n = len(numerator)
    ratios, undefined = [], 0
    for _ in range(draws):
        index = rng.integers(0, n, size=n)
        g_mean = denominator[index].mean()
        h_mean = numerator[index].mean()
        if g_mean <= 0 or h_mean <= 0:
            undefined += 1
            continue
        ratios.append(h_mean / g_mean)
    if not ratios:
        return {"status": "undefined in every resample", "undefined_fraction": 1.0}
    return {
        "status": "estimated",
        "percentile_95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
        "undefined_fraction": undefined / draws,
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


def analyze_blocks(
    runs: list[dict[str, Any]], contract: dict[str, Any], rng: np.random.Generator
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
    assay_sensitive = float(np.mean(values["E_exact_gain"] > 0)) >= 0.75

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
        "contrasts": summaries,
        "gap_magnitude": magnitude,
        "source_gain_supported": bool(source_supported),
        "harm_bounded": bool(harm_bounded),
        "assay_sensitive": bool(assay_sensitive),
        "primary_claim": claim,
        "secondary_p_values": p_values,
        "secondary_holm_rejections": holm(p_values, alpha),
        "break_even_prevalence": break_even(gap, margin),
        "harm_to_gap_crossover": ratio_interval(values["H"], values["G"], rng),
        "sensitivity": {
            "trimmed_mean_G_10pct": float(trimmed.mean()),
            "leave_one_block_out_G_range": [min(leave_one_out), max(leave_one_out)],
        },
    }


def analyze(runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    contract = config["decision_contract"]
    scheduled = len(runs)
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
    if len(failures) > max_failure * scheduled:
        result["primary_claim"] = "inconclusive (execution failures exceed the declared limit)"
        return result
    if len(screened) < minimum:
        result["primary_claim"] = "inconclusive (too few screened blocks)"
        return result
    result["primary"] = analyze_blocks(screened, contract, rng)
    result["primary_claim"] = result["primary"]["primary_claim"]
    if len(completed) >= 2:
        result["intent_to_treat"] = analyze_blocks(completed, contract, rng)
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
    args = parser.parse_args()
    result = analyze(json.loads(args.records.read_text()), json.loads(args.config.read_text()))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("primary_claim", "scheduled", "screened")}))


if __name__ == "__main__":
    main()
