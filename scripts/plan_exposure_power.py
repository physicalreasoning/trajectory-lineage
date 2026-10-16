"""Plan learned exposure replication from a pilot block-level standard deviation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def required_blocks(
    *,
    standard_deviation: float,
    detectable_excess: float,
    alpha: float,
    power: float,
) -> int:
    if standard_deviation <= 0 or detectable_excess <= 0:
        raise ValueError("standard deviation and detectable excess must be positive")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie between zero and one")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    return math.ceil(((z_alpha + z_power) * standard_deviation / detectable_excess) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/convlstm_smoke.json"),
    )
    parser.add_argument(
        "--pilot-sd",
        type=float,
        nargs="+",
        default=[0.0025, 0.005, 0.0075, 0.01],
    )
    parser.add_argument("--effect-fraction", type=float, default=0.25)
    parser.add_argument("--detectable-excess-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.9)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())["aggregate"]
    constant = result["constant_velocity_center_error"]["mean"]
    model = result["model_center_error"]["mean"]
    headroom = constant - model
    threshold = args.effect_fraction * headroom
    detectable_excess = args.detectable_excess_fraction * threshold
    rows = [
        {
            "pilot_standard_deviation": standard_deviation,
            "required_blocks": required_blocks(
                standard_deviation=standard_deviation,
                detectable_excess=detectable_excess,
                alpha=args.alpha,
                power=args.power,
            ),
        }
        for standard_deviation in args.pilot_sd
    ]
    output = {
        "constant_velocity_error": constant,
        "convlstm_error": model,
        "headroom": headroom,
        "minimum_effect_fraction": args.effect_fraction,
        "minimum_raw_gap": threshold,
        "detectable_excess_fraction": args.detectable_excess_fraction,
        "detectable_excess": detectable_excess,
        "alpha": args.alpha,
        "power": args.power,
        "sensitivity": rows,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
