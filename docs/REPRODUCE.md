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

## scope

these commands reproduce software contracts only. they do not train a predictor or produce evidence for the research hypothesis.
