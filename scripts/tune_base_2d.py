"""Effect-blind base-model tuning for the 2D study.

Trains only base models (no continuation arms are built or scored) and compares their bounded
error on a separate screen population with constant-velocity extrapolation, at several training
checkpoints. Used to fix ``base_training_steps`` (and capacity) before any 2D effect is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exposure_experiment import (  # noqa: E402
    constant_velocity_error,
    evaluate,
    new_model,
    spec_from,
    to_tensors,
    train_steps,
)

from trajectory_lineage.exposure import (  # noqa: E402
    AppearanceRegistry,
    render_event,
    sample_event,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026102399)
    parser.add_argument("--checkpoints", type=str, default="300,600,1200,2000")
    parser.add_argument("--hidden", type=str, default="16,32")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = spec_from(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    rows = []
    for block in range(args.blocks):
        rng = np.random.default_rng([args.seed, block])
        seeds = AppearanceRegistry(rng)
        base_events = [sample_event(rng, spec) for _ in range(int(config["base_train_examples"]))]
        screen_events = [sample_event(rng, spec) for _ in range(int(config["screen_examples"]))]
        base = to_tensors(
            [render_event(e, spec, seeds.draw()) for e in base_events], base_events, spec
        )
        screen = to_tensors(
            [render_event(e, spec, seeds.draw()) for e in screen_events], screen_events, spec
        )
        cv = constant_velocity_error(screen_events, spec, float(config["error_cap"]))
        for hidden in [int(h) for h in args.hidden.split(",")]:
            torch.manual_seed(1000 * block + hidden)
            cfg = dict(config, hidden_channels=hidden)
            model = new_model(cfg, spec, device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
            batch_rng = np.random.default_rng([args.seed, block, hidden])
            done, t0 = 0, time.perf_counter()
            for ck in checkpoints:
                batches = []
                for _ in range(ck - done):
                    idx = torch.from_numpy(
                        batch_rng.integers(0, len(base[0]), int(config["batch_size"]))
                    )
                    batches.append((base[0][idx], base[1][idx]))
                train_steps(model, opt, batches, device)
                done = ck
                err = evaluate(model, *screen, device, float(config["error_cap"]))
                rows.append(
                    dict(
                        block=block,
                        hidden=hidden,
                        steps=ck,
                        screen_error=err,
                        cv_error=cv,
                        ratio=err / cv,
                        competent=bool(err < cv),
                        seconds=time.perf_counter() - t0,
                    )
                )
                print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for hidden in sorted({r["hidden"] for r in rows}):
        for ck in checkpoints:
            sel = [r for r in rows if r["hidden"] == hidden and r["steps"] == ck]
            summary[f"h{hidden}_s{ck}"] = dict(
                competent_fraction=float(np.mean([r["competent"] for r in sel])),
                ratio_median=float(np.median([r["ratio"] for r in sel])),
                ratio_max=float(np.max([r["ratio"] for r in sel])),
                seconds_median=float(np.median([r["seconds"] for r in sel])),
            )
    args.out.write_text(
        json.dumps(
            dict(
                summary=summary,
                rows=rows,
                args=vars(args) | {"config": str(args.config), "out": str(args.out)},
            ),
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
