"""Finite-sample power for the confirmatory decision contract, from an effect-blinded pilot.

The pilot releases only contrast dispersion. Because G = S + H, the blinded SDs of S, H, and G
determine the covariance of (S, H): Cov(S, H) = (Var G - Var S - Var H) / 2. Joint power for the
intersection-union rule is then simulated under coherent planning alternatives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (Lentz)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1 - x) / b


def t_cdf(value: float, df: int) -> float:
    x = df / (df + value * value)
    tail = 0.5 * _incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def t_quantile(probability: float, df: int) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must lie between zero and one")
    if df < 1:
        raise ValueError("degrees of freedom must be at least one")
    low, high = -1.0, 1.0
    while t_cdf(low, df) > probability:
        low *= 2
    while t_cdf(high, df) < probability:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        if t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def simulate(
    *,
    blocks: int,
    mean_s: float,
    mean_h: float,
    covariance: np.ndarray,
    contract: dict,
    rng: np.random.Generator,
    simulations: int,
) -> dict[str, float]:
    draws = rng.multivariate_normal([mean_s, mean_h], covariance, size=(simulations, blocks))
    s, h = draws[..., 0], draws[..., 1]
    g = s + h
    crit = t_quantile(1 - float(contract["alpha_one_sided"]), blocks - 1)

    def bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = values.mean(axis=1)
        half = crit * values.std(axis=1, ddof=1) / math.sqrt(blocks)
        return center - half, center + half

    g_low, g_high = bounds(g)
    s_low, _ = bounds(s)
    _, h_high = bounds(h)
    consequential = (
        (g_low > float(contract["consequential_threshold"]))
        & (s_low > 0)
        & (h_high < float(contract["harm_margin_h"]))
    )
    bound = float(contract["negligible_bound"])
    negligible = (g_low > -bound) & (g_high < bound)
    return {
        "consequential": float(consequential.mean()),
        "gap_gate": float((g_low > float(contract["consequential_threshold"])).mean()),
        "source_gate": float((s_low > 0).mean()),
        "harm_gate": float((h_high < float(contract["harm_margin_h"])).mean()),
        "negligible": float(negligible.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        type=Path,
        nargs="+",
        default=[
            Path("results/exposure_pilot_v4_blinded.json"),
            Path("results/exposure_pilot_v5_blinded.json"),
        ],
        help="One or more effect-blinded pilots; S, H, and G variances are pooled across them.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/exposure_confirmation.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/confirmation_power.json"))
    parser.add_argument("--simulations", type=int, default=20_000)
    args = parser.parse_args()
    pilots = [json.loads(path.read_text()) for path in args.pilot]
    config = json.loads(args.config.read_text())
    contract = config["decision_contract"]
    blocks = int(config.get("planning_analyzed_blocks", config["blocks"]))

    def pooled_variance(name: str) -> tuple[float, int]:
        total, degrees = 0.0, 0
        for pilot in pilots:
            entry = pilot["dispersion"][name]
            df = int(entry["blocks"]) - 1
            total += df * float(entry["standard_deviation"]) ** 2
            degrees += df
        return total / degrees, degrees

    (var_s, df), (var_h, _), (var_g, _) = (
        pooled_variance("C_source_gain"),
        pooled_variance("C_independent_harm"),
        pooled_variance("C_reuse_gap"),
    )
    covariance_sh = (var_g - var_s - var_h) / 2
    correlation = covariance_sh / math.sqrt(var_s * var_h)
    confidence = float(pilots[0]["dispersion"]["C_reuse_gap"]["upper_limit_confidence"])
    chi_square = (
        df
        * (1 - 2 / (9 * df) + NormalDist().inv_cdf(1 - confidence) * math.sqrt(2 / (9 * df))) ** 3
    )
    inflation = math.sqrt(df / chi_square)
    base_cov = np.array([[var_s, covariance_sh], [covariance_sh, var_h]]) * inflation**2

    rng = np.random.default_rng(20261016)
    rows = []
    threshold = float(contract["consequential_threshold"])
    for sd_multiplier in (1.0, 1.5, 2.0):
        covariance = base_cov * sd_multiplier**2
        for mean_h in (0.0, 0.0005):
            mean_g = threshold + 0.005
            power = simulate(
                blocks=blocks,
                mean_s=mean_g - mean_h,
                mean_h=mean_h,
                covariance=covariance,
                contract=contract,
                rng=rng,
                simulations=args.simulations,
            )
            null_power = simulate(
                blocks=blocks,
                mean_s=0.0,
                mean_h=0.0,
                covariance=covariance,
                contract=contract,
                rng=rng,
                simulations=args.simulations,
            )
            at_threshold = simulate(
                blocks=blocks,
                mean_s=threshold,
                mean_h=0.0,
                covariance=covariance,
                contract=contract,
                rng=rng,
                simulations=args.simulations,
            )
            rows.append(
                {
                    "sd_multiplier_over_upper_limit": sd_multiplier,
                    "alternative": {"G": mean_g, "S": mean_g - mean_h, "H": mean_h},
                    "power_at_alternative": power,
                    "negligible_power_at_zero": null_power["negligible"],
                    "false_consequential_rate_at_threshold": at_threshold["consequential"],
                }
            )
    result = {
        "blocks": blocks,
        "pilots": [str(path) for path in args.pilot],
        "pilot_config_sha256": [pilot.get("config_sha256") for pilot in pilots],
        "pooled_degrees_of_freedom": df,
        "pooled_sd": {"S": math.sqrt(var_s), "H": math.sqrt(var_h), "G": math.sqrt(var_g)},
        "derived_sh_correlation": correlation,
        "upper_limit_inflation": inflation,
        "t_critical": t_quantile(1 - float(contract["alpha_one_sided"]), blocks - 1),
        "simulations": args.simulations,
        "rows": rows,
        "assumptions": "Block contrasts bivariate normal with pilot-derived covariance inflated "
        "to the 80% upper SD limit, then by 1.0, 1.5, and 2.0. Conditional power, not assurance.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
