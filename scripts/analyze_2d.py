"""Analyzer for the registered 2D trajectory-reuse study.

Reuses the frozen 1D machinery (record validation, contrast recomputation, t intervals, Holm,
break-even prevalence, Fieller crossover) from ``analyze_confirmation.py`` and replaces only the
decision logic, which the 2D registration revises in four ways:

1. Support uses two-sided 95% intervals; absence (negligible) uses two-sided 90% intervals, as
   the plan's analysis rules require.
2. "Prevalence-dependent" requires the WHOLE 95% interval of the gap G to lie in
   (0.002, 0.010]. An interval that starts above 0.002 but reaches past 0.010 is reported as
   "detectable, magnitude unresolved" instead of being called prevalence-dependent.
3. The positive control is reachable: assay sensitivity requires the 95% lower bound of arm E's
   exact-render gain to exceed zero (the 1D rule's "E can detect 0.010" is not required).
4. Magnitude, safeguards (source gain, harm) and event identity (Q) are separate axes; the
   primary claim combines them by a fixed table. The confirmatory event-identity label comes from
   the Holm-adjusted secondary test, never from a nominal interval.
5. Any unresolved block failure makes every claim inconclusive (complete-block rule).
6. Predictor capacity is a registered block factor (``capacity_levels``). The primary claim uses
   the blocks at the registered primary level only. The capacity secondary tests whether the
   reuse gap G grows with capacity: a Welch comparison of G between the largest and the smallest
   level, with its one-sided p-value in the Holm family of secondaries ("capacity_increase").
   Per-level estimates are descriptive; "capacity_independent" needs the 90% interval of the
   difference inside the negligible bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_confirmation import (  # noqa: E402
    FIELDS,
    RecordError,
    arm_summaries,
    break_even,
    check_identity,
    fieller,
    holm,
    interval,
    one_sided_p,
    recompute_contrasts,
    t_cdf,
    t_quantile,
    validate_records,
)
from freeze import FreezeError  # noqa: E402
from freeze import validate as validate_freeze  # noqa: E402

from trajectory_lineage.exposure import capacity_levels, capacity_schedule  # noqa: E402

MAGNITUDES = (
    "consequential",
    "prevalence_dependent",
    "detectable_magnitude_unresolved",
    "negligible",
    "reverse",
    "inconclusive",
)


def classify_gap_2d(
    gap95: dict[str, float], gap90: dict[str, float], contract: dict[str, Any]
) -> str:
    consequential = float(contract["consequential_threshold"])
    negligible = float(contract["negligible_bound"])
    if gap95["lower"] > consequential:
        return "consequential"
    if gap95["lower"] > negligible:
        return (
            "prevalence_dependent"
            if gap95["upper"] <= consequential
            else "detectable_magnitude_unresolved"
        )
    if gap90["lower"] > -negligible and gap90["upper"] < negligible:
        return "negligible"
    if gap95["upper"] < -negligible:
        return "reverse"
    return "inconclusive"


def primary_claim(magnitude: str, source_ok: bool, harm_ok: bool, assay_ok: bool) -> str:
    if magnitude in ("consequential", "prevalence_dependent"):
        return (
            f"{magnitude} reuse advantage"
            if (source_ok and harm_ok)
            else "differential gap without safeguards"
        )
    if magnitude == "detectable_magnitude_unresolved":
        return (
            "detectable reuse advantage, magnitude unresolved"
            if (source_ok and harm_ok)
            else "differential gap without safeguards"
        )
    if magnitude == "negligible":
        return "negligible differential gap" if assay_ok else "positive control not detected"
    if magnitude == "reverse":
        return "reverse gap"
    return "inconclusive"


def welch(a: np.ndarray, b: np.ndarray, alpha_one_sided: float) -> dict[str, float]:
    """Difference of means a - b with a Welch-Satterthwaite t interval and a one-sided p (a > b)."""

    na, nb = len(a), len(b)
    va, vb = float(a.var(ddof=1)) / na, float(b.var(ddof=1)) / nb
    se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / ((va**2 / (na - 1)) + (vb**2 / (nb - 1))) if se > 0 else float(na + nb - 2)
    diff = float(a.mean() - b.mean())
    crit = t_quantile(1 - alpha_one_sided, int(max(1, round(df))))
    p = 1.0 if se == 0 and diff <= 0 else (0.0 if se == 0 else 1.0 - t_cdf(diff / se, int(max(1, round(df)))))
    return {"n_a": na, "n_b": nb, "mean": diff, "se": se, "df": df, "lower": diff - crit * se, "upper": diff + crit * se, "p_one_sided": p}


def analyze_blocks_2d(
    runs: list[dict[str, Any]], contract: dict[str, Any], extra_p_values: dict[str, float] | None = None
) -> dict[str, Any]:
    alpha_support = float(contract["alpha_one_sided"])  # 0.025 -> two-sided 95%
    alpha_absence = float(contract["alpha_absence_one_sided"])  # 0.05 -> two-sided 90%
    values = {
        name: np.array([run["contrasts"][field] for run in runs])
        for name, field in FIELDS.items()
        if all(field in run["contrasts"] for run in runs)
    }
    s95 = {name: interval(array, alpha_support) for name, array in values.items()}
    s90 = {name: interval(array, alpha_absence) for name, array in values.items()}
    magnitude = classify_gap_2d(s95["G"], s90["G"], contract)
    harm_margin = float(contract["harm_margin_h"])
    source_ok = s95["S"]["lower"] > 0
    harm_ok = s95["H"]["upper"] < harm_margin
    assay_ok = s95["E_exact_gain"]["lower"] > 0
    threshold = float(contract["consequential_threshold"])
    p_values = {
        "identity_specific": one_sided_p(values["Q"], threshold),
        "detectable": one_sided_p(values["G"], 0.0),
        "consequential_harm": one_sided_p(values["H"], harm_margin),
        **(extra_p_values or {}),
    }
    holm_decisions = holm(p_values, alpha_support)
    identity = (
        "event_specific"
        if holm_decisions["identity_specific"]
        else "below_threshold"
        if s95["Q"]["upper"] < threshold
        else "unresolved"
    )
    return {
        "blocks": len(runs),
        "arm_population_errors": arm_summaries(runs, alpha_support),
        "contrasts_95": s95,
        "gap_90": s90["G"],
        "axes": {
            "magnitude": magnitude,
            "source_gain_supported": bool(source_ok),
            "harm_bounded": bool(harm_ok),
            "positive_control_detected": bool(assay_ok),
            "event_identity": identity,
        },
        "E_exact_gain_positive_fraction": float(np.mean(values["E_exact_gain"] > 0)),
        "primary_claim": primary_claim(magnitude, source_ok, harm_ok, assay_ok),
        "secondary_p_values": p_values,
        "secondary_holm_rejections": holm_decisions,
        "break_even_prevalence": break_even(s95["G"], float(contract["ranking_margin_m"])),
        "harm_to_gap_crossover_fieller": fieller(values["H"], values["G"], alpha_support),
    }


def analyze_capacity(
    screened: list[dict[str, Any]], contract: dict[str, Any], levels: list[int]
) -> dict[str, Any]:
    """Per-level reuse-gap estimates and the trend between the largest and smallest level."""

    alpha_support = float(contract["alpha_one_sided"])
    alpha_absence = float(contract["alpha_absence_one_sided"])
    negligible = float(contract["negligible_bound"])
    minimum_by_level = {int(k): int(v) for k, v in (contract.get("minimum_screened_blocks_by_capacity") or {}).items()}
    gaps = {level: np.array([run["contrasts"][FIELDS["G"]] for run in screened if run["hidden_channels"] == level]) for level in levels}
    per_level: dict[str, Any] = {}
    for level in levels:
        g = gaps[level]
        entry: dict[str, Any] = {"screened": int(len(g)), "model_parameters": next((run.get("model_parameters") for run in screened if run["hidden_channels"] == level), None)}
        if len(g) >= 2:
            g95, g90 = interval(g, alpha_support), interval(g, alpha_absence)
            e = np.array([run["contrasts"][FIELDS["E_exact_gain"]] for run in screened if run["hidden_channels"] == level])
            entry.update(G_95=g95, G_90=g90, magnitude_descriptive=classify_gap_2d(g95, g90, contract), E_lower_95=interval(e, alpha_support)["lower"])
        per_level[str(level)] = entry
    low, high = min(levels), max(levels)
    out: dict[str, Any] = {"levels": levels, "per_level": per_level, "trend": None, "trend_p_one_sided": None, "label": "unresolved"}
    too_few = [level for level in (low, high) if len(gaps[level]) < max(2, minimum_by_level.get(level, 2))]
    if too_few:
        out["label"] = f"unresolved (too few screened blocks at capacity {too_few})"
        return out
    trend95 = welch(gaps[high], gaps[low], alpha_support)
    trend90 = welch(gaps[high], gaps[low], alpha_absence)
    out.update(trend={"contrast": f"G({high}) - G({low})", "ci95": trend95, "ci90": {k: trend90[k] for k in ("lower", "upper")}}, trend_p_one_sided=trend95["p_one_sided"])
    if trend90["lower"] > -negligible and trend90["upper"] < negligible:
        out["label"] = "capacity_independent"
    return out


DECISIVE = {
    "consequential reuse advantage",
    "prevalence_dependent reuse advantage",
    "detectable reuse advantage, magnitude unresolved",
    "negligible differential gap",
    "reverse gap",
}


def analyze_2d(
    runs: list[dict[str, Any]], config: dict[str, Any], *, check_records: bool = True
) -> dict[str, Any]:
    contract = config["decision_contract"]
    scheduled = int(config["blocks"])
    if check_records:
        runs = validate_records(runs, scheduled, config)
    else:  # fast path for simulations: still never trust stored contrasts or screen flags
        runs = [
            dict(
                run,
                contrasts=recompute_contrasts(run["errors"]),
                base_competent=bool(
                    run["screen_base_error"] < run["screen_constant_velocity_error"]
                ),
            )
            if run["status"] == "completed"
            else dict(run)
            for run in runs
        ]
    completed = [run for run in runs if run["status"] == "completed"]
    failures = [run for run in runs if run["status"] != "completed"]
    levels = capacity_levels(config)
    schedule = capacity_schedule(config)
    multi = len(levels) > 1
    for run in completed:  # a record's capacity must be the registered one for its block index
        recorded = run.get("hidden_channels", levels[0] if not multi else None)
        if recorded != schedule[run["block_index"]]:
            raise RecordError(f"block {run['block_index']} records hidden_channels={recorded}; registered {schedule[run['block_index']]}")
        run["hidden_channels"] = recorded
    primary_level = int(contract.get("primary_capacity", levels[0]))
    screened_all = [run for run in completed if run.get("base_competent", False)]
    screened = [run for run in screened_all if run["hidden_channels"] == primary_level]
    completed_primary = [run for run in completed if run["hidden_channels"] == primary_level]
    minimum = int(contract["minimum_screened_blocks"])
    result: dict[str, Any] = {
        "scheduled": scheduled,
        "completed": len(completed),
        "failures": [run["block_index"] for run in failures],
        "screened": len(screened),
        "primary_capacity": primary_level,
    }
    capacity = analyze_capacity(screened_all, contract, levels) if multi else None
    extra = {"capacity_increase": capacity["trend_p_one_sided"]} if capacity and capacity["trend_p_one_sided"] is not None else None
    if len(screened) >= 2:
        result["primary"] = analyze_blocks_2d(screened, contract, extra)
    if len(completed_primary) >= 2:
        result["all_completed_blocks_sensitivity"] = analyze_blocks_2d(completed_primary, contract, extra)
    if capacity is not None:
        if "primary" in result and extra and result["primary"]["secondary_holm_rejections"].get("capacity_increase"):
            capacity["label"] = "capacity_dependent"
        result["capacity"] = capacity
    reason = (
        "unresolved block failures"
        if failures
        else "too few screened blocks"
        if len(screened) < minimum
        else None
    )
    if reason is None:
        result["primary_claim"] = result["primary"]["primary_claim"]
        return result
    # Complete-block and minimum-screened rules: estimates stay reported, no decision survives.
    for key in ("primary", "all_completed_blocks_sensitivity"):
        if key in result:
            section = result[key]
            section["decisions_suppressed"] = reason
            section["primary_claim"] = f"inconclusive ({reason})"
            section["secondary_holm_rejections"] = dict.fromkeys(
                section["secondary_holm_rejections"], False
            )
            section["axes"]["event_identity"] = "unresolved"
    if "capacity" in result:
        result["capacity"]["label"] = f"unresolved ({reason})"
    result["primary_claim"] = f"inconclusive ({reason})"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/exposure2d_confirmation.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_freeze(json.loads(args.freeze_manifest.read_text()), args.config, Path(__file__))
    except FreezeError as error:
        raise SystemExit(f"refusing to analyze: {error}") from error
    runs = check_identity(json.loads(args.records.read_text()), args.freeze_manifest, args.config)
    result = analyze_2d(runs, json.loads(args.config.read_text()))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("primary_claim", "scheduled", "screened")}))


if __name__ == "__main__":
    main()
