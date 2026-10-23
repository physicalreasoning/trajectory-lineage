"""Sample-size planning for the 2D confirmation from the effect-blinded 2D pilot.

Vectorized simulation of block-level contrasts under the registered decision rules (two-sided
95% t intervals for support, 90% for absence). Planning SDs are the pilot's 80% upper limits;
Cov(S, H) is identified from G = S + H. The full record-level dry run (dryrun_2d.py) then checks
the chosen N end to end through the real analyzer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plan_confirmation import t_quantile  # noqa: E402


def sds_from(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text())["dispersion"]
    key = {
        "S": "C_source_gain",
        "H": "C_independent_harm",
        "G": "C_reuse_gap",
        "Q": "C_minus_D_gap",
        "E": "E_exact_gain",
    }
    return {k: float(d[v]["standard_deviation_upper_limit"]) for k, v in key.items()}


def simulate(n, sds, S, H, E, sims, rng):
    vs, vh, vg = sds["S"] ** 2, sds["H"] ** 2, sds["G"] ** 2
    cov = float(
        np.clip((vg - vs - vh) / 2, -0.999 * sds["S"] * sds["H"], 0.999 * sds["S"] * sds["H"])
    )
    L = np.linalg.cholesky(np.array([[vs, cov], [cov, vh]]))
    z = rng.standard_normal((sims, n, 2)) @ L.T
    s = S + z[..., 0]
    h = H + z[..., 1]
    g = s + h
    e = E + sds["E"] * rng.standard_normal((sims, n))
    t95, t90 = t_quantile(0.975, n - 1), t_quantile(0.95, n - 1)

    def ci(x, t):
        m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
        return m - t * se, m + t * se

    gl, gu = ci(g, t95)
    gl90, gu90 = ci(g, t90)
    sl, _ = ci(s, t95)
    _, hu = ci(h, t95)
    el, _ = ci(e, t95)
    safe = (sl > 0) & (hu < 0.001)
    return {
        "consequential_claim": float(np.mean((gl > 0.010) & safe)),
        "prevalence_dependent_claim": float(np.mean((gl > 0.002) & (gu <= 0.010) & safe)),
        "any_positive_magnitude": float(np.mean(gl > 0.002)),
        "negligible_claim": float(np.mean((gl90 > -0.002) & (gu90 < 0.002) & (el > 0))),
        "harm_bounded": float(np.mean(hu < 0.001)),
        "assay_sensitive": float(np.mean(el > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=str, default="300,400,600,800,1000,1200,1500")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--attrition", type=float, default=0.03)
    ap.add_argument(
        "--assay-effect",
        type=float,
        default=None,
        help="planning E gain; default = 90%-power detectable effect at the chosen N",
    )
    args = ap.parse_args()
    sds = sds_from(args.pilot)
    rng = np.random.default_rng(2026102302)
    scenarios = {
        "consequential_G0.015_H0": dict(S=0.015, H=0.0),
        "consequential_G0.015_H0.0005": dict(S=0.0145, H=0.0005),
        "at_threshold_G0.010": dict(S=0.010, H=0.0),
        "prevalence_G0.006": dict(S=0.006, H=0.0),
        "null_G0": dict(S=0.0, H=0.0),
        "negligible_boundary_G0.002": dict(S=0.002, H=0.0),
        "harm_at_margin_H0.001": dict(S=0.014, H=0.001),
    }
    out = {"planning_sds_80pct_upper": sds, "attrition": args.attrition, "grid": {}}
    for scheduled in [int(x) for x in args.grid.split(",")]:
        n = int(round(scheduled * (1 - args.attrition)))
        e_mde = (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.90)) * sds["E"] / np.sqrt(n)
        e_plan = args.assay_effect if args.assay_effect is not None else e_mde
        row = {"analyzed_blocks": n, "E_gain_detectable_at_90pct_power": float(e_mde)}
        for name, sc in scenarios.items():
            row[name] = simulate(n, sds, sc["S"], sc["H"], e_plan, args.sims, rng)
        out["grid"][str(scheduled)] = row
        print(
            scheduled,
            n,
            {
                k: (v["consequential_claim"], v["negligible_claim"])
                for k, v in row.items()
                if isinstance(v, dict)
            },
            flush=True,
        )
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
