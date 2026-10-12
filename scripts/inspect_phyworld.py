"""Validate a downloaded PhyWorld HDF5 file and write a structural manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_lineage.phyworld import inspect_phyworld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/external/phyworld/collision_eval.hdf5"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/phyworld_collision_eval_manifest.json"),
    )
    args = parser.parse_args()

    summary = inspect_phyworld(args.path).to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
