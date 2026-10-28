"""Simulated dry run and operating characteristics of the frozen 2D decision contract.

Synthetic block records are generated under declared true effects, using block-level
dispersions released by the effect-blinded 2D pilot. Every synthetic dataset passes through the
same record validator and analyzer that will analyze the real study (``analyze_2d.analyze_2d``),
so the registration documents exactly what the frozen code does under each truth:

* a mock decision table (one simulated dataset per scenario), and
* operating characteristics: the probability of each primary claim and axis outcome.

Covariance: because G = S + H, the blinded SDs of S, H and G identify Cov(S, H). Other contrasts
(Q, Q2, E_exact_gain) are drawn independently with their pilot SDs; the analyzer recomputes all
contrasts from component errors, so the records are internally consistent by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_2d import analyze_2d  # noqa: E402
from analyze_confirmation import expected_lengths  # noqa: E402

from trajectory_lineage.exposure import capacity_levels, capacity_schedule, scaled_capacity_config  # noqa: E402

SCENARIOS = {
    # name: true means of S (source gain), H (harm), Q (C-D gap), E (exact gain)
    "no_reuse_effect": dict(S=0.0, H=0.0, Q=0.0, E=0.004),
    "tiny_gap_0.001": dict(S=0.001, H=0.0, Q=0.0, E=0.004),
    "prevalence_dependent_0.006": dict(S=0.006, H=0.0, Q=0.0, E=0.004),
    "at_consequential_threshold_0.010": dict(S=0.010, H=0.0, Q=0.0, E=0.004),
    "consequential_0.015": dict(S=0.015, H=0.0, Q=0.0, E=0.004),
    "consequential_0.015_with_harm_0.0005": dict(S=0.0145, H=0.0005, Q=0.0, E=0.004),
    "consequential_but_harmful_0.002": dict(S=0.013, H=0.002, Q=0.0, E=0.004),
    "event_specific_Q_0.015": dict(S=0.015, H=0.0, Q=0.015, E=0.004),
    "positive_control_dead": dict(S=0.0, H=0.0, Q=0.0, E=0.0),
    "weak_positive_control_E_0.001": dict(S=0.0, H=0.0, Q=0.0, E=0.001),
    "reverse_-0.005": dict(S=-0.005, H=0.0, Q=0.0, E=0.004),
    # capacity scenarios (multipliers apply to S, Q and E at the named hidden-channel levels; the
    # smallest level keeps the base truth). In a single-level study they reduce to the base truth.
    "capacity_dependent_x2_x3": dict(
        S=0.006, H=0.0, Q=0.0, E=0.004, capacity_multipliers={"64": 2.0, "256": 3.0}
    ),
    "capacity_flat_consequential": dict(
        S=0.015, H=0.0, Q=0.0, E=0.004, capacity_multipliers={"64": 1.0, "256": 1.0}
    ),
    "capacity_dependent_from_null": dict(
        S=0.0, H=0.0, Q=0.0, E=0.004, capacity_multipliers={"64": 1.0, "256": 1.0}, capacity_add={"64": 0.004, "256": 0.010}
    ),
}


def dispersion_from(path: Path) -> dict[str, float]:
    disp = json.loads(path.read_text())["dispersion"]

    def sd(name: str) -> float:
        return float(disp[name]["standard_deviation_upper_limit"])

    return {
        "S": sd("C_source_gain"),
        "H": sd("C_independent_harm"),
        "G": sd("C_reuse_gap"),
        "Q": sd("C_minus_D_gap"),
        "Q2": sd("C_minus_D2_gap"),
        "E": sd("E_exact_gain"),
    }


def level_truth(truth: dict, level: int, base_level: int) -> dict:
    """Truth at one capacity level: base truth scaled by the level's multiplier plus an additive gap."""

    if level == base_level:
        return truth
    mult = float(truth.get("capacity_multipliers", {}).get(str(level), 1.0))
    add = float(truth.get("capacity_add", {}).get(str(level), 0.0))
    return dict(truth, S=truth["S"] * mult + add, Q=truth["Q"] * mult, E=truth["E"] * mult)


def synthetic_runs(rng, config, truth, sds, n_blocks, competent_rate):
    levels = capacity_levels(config)
    if len(levels) > 1:
        schedule = capacity_schedule(scaled_capacity_config(config, n_blocks))
        runs = []
        for level in levels:
            indices = [b for b in range(n_blocks) if schedule[b] == level]
            level_runs = _synthetic_runs(rng, config, level_truth(truth, level, min(levels)), sds, len(indices), competent_rate)
            for run, b in zip(level_runs, indices, strict=True):
                run["block_index"] = b
                run["hidden_channels"] = level
                run["model_parameters"] = level * 1000
            runs += level_runs
        return sorted(runs, key=lambda run: run["block_index"])
    runs = _synthetic_runs(rng, config, truth, sds, n_blocks, competent_rate)
    for run in runs:
        run["hidden_channels"] = levels[0]
    return runs


def _synthetic_runs(rng, config, truth, sds, n_blocks, competent_rate):
    var_s, var_h, var_g = sds["S"] ** 2, sds["H"] ** 2, sds["G"] ** 2
    cov = (var_g - var_s - var_h) / 2
    cov = float(np.clip(cov, -0.999 * sds["S"] * sds["H"], 0.999 * sds["S"] * sds["H"]))
    sh = rng.multivariate_normal(
        [truth["S"], truth["H"]], [[var_s, cov], [cov, var_h]], size=n_blocks
    )
    q = rng.normal(truth["Q"], sds["Q"], n_blocks)
    q2 = rng.normal(truth["Q"], sds["Q2"], n_blocks)
    e = rng.normal(truth["E"], sds["E"], n_blocks)
    lengths = expected_lengths(config)
    renders = int(config["heldout_renders_per_source"])
    runs = []
    for b in range(n_blocks):
        base = 0.06 + rng.normal(0, 0.004, 3)
        bs, be, bi = base
        s, h = sh[b]
        cs, ci = bs - s, bi + h  # S = B_s - C_s ; H = C_i - B_i
        di = ci + rng.normal(0, 0.001)
        ds = cs + q[b] + (di - ci)  # Q = (D_s - C_s) - (D_i - C_i)
        d2i = ci + rng.normal(0, 0.001)
        d2s = cs + q2[b] + (d2i - ci)
        ee = be - e[b]
        errors = {
            "B": {"source": bs, "exact": be, "independent": bi},
            "C": {"source": cs, "exact": be - 0.5 * s, "independent": ci},
            "D": {"source": ds, "exact": be, "independent": di},
            "D2": {"source": d2s, "exact": be, "independent": d2i},
            "E": {"source": ee + 0.002, "exact": ee, "independent": bi + rng.normal(0, 0.001)},
        }
        errors = {
            a: {p: float(np.clip(v, 0.0, 1.0)) for p, v in pops.items()}
            for a, pops in errors.items()
        }
        items = {a: {p: [v] * lengths[p] for p, v in pops.items()} for a, pops in errors.items()}
        competent = rng.random() < competent_rate
        runs.append(
            {
                "block_index": b,
                "status": "completed",
                "errors": errors,
                "item_losses": items,
                "source_family_of_render": [i // renders for i in range(lengths["source"])],
                "screen_base_error": 0.055 if competent else 0.09,
                "screen_constant_velocity_error": 0.08,
                "base_competent": competent,
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pilot", type=Path, required=True, help="blinded pilot report (dispersion)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=400)
    parser.add_argument("--blocks", type=str, default="")
    parser.add_argument("--competent-rate", type=float, default=0.97)
    parser.add_argument("--sd-scale", type=float, default=1.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sds = {k: v * args.sd_scale for k, v in dispersion_from(args.pilot).items()}
    block_options = (
        [int(x) for x in args.blocks.split(",")] if args.blocks else [int(config["blocks"])]
    )
    rng = np.random.default_rng(20261023)
    out = {
        "planning_sds_80pct_upper": sds,
        "sd_scale": args.sd_scale,
        "competent_rate": args.competent_rate,
        "results": {},
    }
    for n in block_options:
        cfg = scaled_capacity_config(config, n)
        primary_level = int(config["decision_contract"].get("primary_capacity", capacity_levels(config)[0]))
        primary_blocks = int(cfg.get("capacity_blocks", {}).get(str(primary_level), n))
        cfg["decision_contract"] = dict(
            config["decision_contract"], minimum_screened_blocks=int(0.9 * primary_blocks)
        )
        if "minimum_screened_blocks_by_capacity" in config["decision_contract"]:
            cfg["decision_contract"]["minimum_screened_blocks_by_capacity"] = {
                str(k): int(0.9 * int(v)) for k, v in cfg["capacity_blocks"].items()
            }
        per = {}
        for name, truth in SCENARIOS.items():
            claims, axes = Counter(), Counter()
            mock = None
            for _rep in range(args.replications):
                runs = synthetic_runs(rng, cfg, truth, sds, n, args.competent_rate)
                result = analyze_2d(
                    runs, cfg, check_records=True
                )  # full record validation, every dataset
                claims[result["primary_claim"]] += 1
                if "primary" in result:
                    ax = result["primary"]["axes"]
                    axes[f"magnitude={ax['magnitude']}"] += 1
                    axes[f"identity={ax['event_identity']}"] += 1
                    axes[f"harm_bounded={ax['harm_bounded']}"] += 1
                    axes[f"positive_control_detected={ax['positive_control_detected']}"] += 1
                if "capacity" in result:
                    axes[f"capacity={result['capacity']['label']}"] += 1
                if mock is None:
                    p = result["primary"]
                    mock = {
                        "primary_claim": result["primary_claim"],
                        "axes": p["axes"],
                        "G_95": [
                            round(p["contrasts_95"]["G"]["lower"], 5),
                            round(p["contrasts_95"]["G"]["upper"], 5),
                        ],
                        "G_90": [round(p["gap_90"]["lower"], 5), round(p["gap_90"]["upper"], 5)],
                        "H_upper_95": round(p["contrasts_95"]["H"]["upper"], 5),
                        "E_lower_95": round(p["contrasts_95"]["E_exact_gain"]["lower"], 5),
                        "break_even": p["break_even_prevalence"],
                    }
                    if "capacity" in result:
                        cap = result["capacity"]
                        mock["capacity"] = {
                            "label": cap["label"],
                            "trend_95": None
                            if cap["trend"] is None
                            else [round(cap["trend"]["ci95"]["lower"], 5), round(cap["trend"]["ci95"]["upper"], 5)],
                        }
            per[name] = {
                "truth": truth,
                "claim_probabilities": {k: v / args.replications for k, v in claims.most_common()},
                "axis_probabilities": {k: v / args.replications for k, v in sorted(axes.items())},
                "mock_dataset_decision": mock,
            }
            print(n, name, dict(claims.most_common(3)), flush=True)
        out["results"][str(n)] = per
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
