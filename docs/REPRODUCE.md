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

## Scope

The test and smoke commands reproduce software contracts only. The memory positive control can validate assay sensitivity, but it cannot produce evidence about a learned world model or practical benchmark contamination.
