"""Registered construction, weighting, and analysis for Experiment 2 (transition-matched Kuka control).

Modules
-------
manifest   episode manifest (id, shard, length, outcome, split) from the cached npz shards; synthetic manifest for dry runs
design     seeded samplers for designs O, A, B, V and the feasibility certificate
weights    frozen evaluation support and overlap weights
analyze    the registered decision contract (paired t, TOST, Holm, missing-run handling)
dryrun     synthetic operating characteristics of the frozen analyzer
freeze     freeze manifest that binds code, config, manifest, selection, support and sign-offs
run_registered  training driver that consumes the frozen selection and writes per-episode scores

Only numpy and scipy are required for everything except ``run_registered`` (torch + the owner's ``wm.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "exp2_registered.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, fixed separators) so hashes are reproducible."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def load_config(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def _jsonable(o: Any) -> Any:
    """Convert numpy scalars and arrays so json can serialize them."""

    import numpy as np

    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=_jsonable) + "\n")
