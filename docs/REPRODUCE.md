# reproduction

## prerequisites

- python 3.11+
- uv

## install

```bash
uv sync
```

## tests

```bash
uv run pytest -q
```

all tests should pass. they cover:

- exact equal-mass velocity exchange;
- momentum and kinetic-energy conservation;
- rejection of trajectories leaving the arena;
- same-state/different-render hashes;
- relationship precedence; and
- bounded metric failure handling.

## lint and format

```bash
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

## smoke

```bash
uv run python scripts/smoke.py
```

expected semantic output:

```json
{
  "different_render_hash": true,
  "frames": 24,
  "relationship": "same_trajectory",
  "same_trajectory_hash": true
}
```

exact hashes are intentionally omitted because they are implementation artifacts. the smoke writes `results/smoke.json` and `results/smoke_manifest.json`.

## registered memory positive control

```bash
uv run python scripts/pilot_memory.py
```

this runs the configuration frozen in `experiments/memory_positive_control.json` and writes `results/memory_positive_control.json`. it uses a nearest-trajectory memory baseline, not a trained neural world model.

## scope

the test and smoke commands reproduce software contracts only. the memory positive control can validate assay sensitivity, but it cannot produce evidence about a learned world model or practical benchmark contamination.
