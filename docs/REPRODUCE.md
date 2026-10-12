# Reproduction

## Prerequisites

- Python 3.11+
- uv

## Install

```bash
uv sync
```

## Tests

```bash
uv run pytest -q
```

All tests should pass. They cover:

- Exact equal-mass velocity exchange;
- Momentum and kinetic-energy conservation;
- Rejection of trajectories leaving the arena;
- Same-state/different-render hashes;
- Relationship precedence; and
- Bounded metric failure handling.

## Lint and format

```bash
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

## Smoke

```bash
uv run python scripts/smoke.py
```

Expected semantic output:

```json
{
  "different_render_hash": true,
  "frames": 24,
  "relationship": "same_trajectory",
  "same_trajectory_hash": true
}
```

Exact hashes are intentionally omitted because they are implementation artifacts. The smoke writes `results/smoke.json` and `results/smoke_manifest.json`.

## Registered memory positive control

```bash
uv run python scripts/pilot_memory.py
```

This runs the configuration frozen in `experiments/memory_positive_control.json` and writes `results/memory_positive_control.json`. It uses a nearest-trajectory memory baseline, not a trained neural world model.

## PhyWorld Dataset Inspection

After downloading the collision evaluation set using `docs/DATA.md`:

```bash
uv run python scripts/inspect_phyworld.py
```

The command validates group alignment, tensor shapes, finite values, and the file checksum. It writes `results/phyworld_collision_eval_manifest.json`.

## PhyWorld State Baselines

After downloading and validating the collision evaluation set:

```bash
uv run python scripts/phyworld_state_baselines.py
```

The command runs the configuration frozen in `experiments/phyworld_state_baselines.json` and writes `results/phyworld_state_baselines.json`.

Expected mean normalized center errors:

| Split | Constant velocity | Nearest trajectory |
|---|---:|---:|
| Within `00000` | 0.2577 | 0.1653 |
| `00000` → `00001` | 0.4729 | 0.2329 |
| `00000` → `00002` | 0.5090 | 0.2604 |

## Scope

The test and smoke commands reproduce software contracts only. The memory positive control validates assay sensitivity, while the PhyWorld inspection and state baselines validate external data compatibility and baseline headroom.
