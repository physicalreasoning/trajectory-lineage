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
    "assay_insensitive": dict(S=0.0, H=0.0, Q=0.0, E=0.0),
    "reverse_-0.005": dict(S=-0.005, H=0.0, Q=0.0, E=0.004),
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


def synthetic_runs(rng, config, truth, sds, n_blocks, competent_rate):
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
        cfg = dict(config, blocks=n)
        cfg["decision_contract"] = dict(
            config["decision_contract"], minimum_screened_blocks=int(0.9 * n)
        )
        per = {}
        for name, truth in SCENARIOS.items():
            claims, axes = Counter(), Counter()
            mock = None
            for rep in range(args.replications):
                runs = synthetic_runs(rng, cfg, truth, sds, n, args.competent_rate)
                result = analyze_2d(runs, cfg, check_records=(rep == 0))
                claims[result["primary_claim"]] += 1
                if "primary" in result:
                    ax = result["primary"]["axes"]
                    axes[f"magnitude={ax['magnitude']}"] += 1
                    axes[f"identity={ax['event_identity']}"] += 1
                    axes[f"harm_bounded={ax['harm_bounded']}"] += 1
                    axes[f"assay_sensitive={ax['assay_sensitive']}"] += 1
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
