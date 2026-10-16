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

## ConvLSTM Smoke

Install the optional model dependencies and run the frozen three-seed configuration:

```bash
uv sync --locked --extra models --no-config --default-index https://pypi.org/simple
uv run --extra models python scripts/convlstm_smoke.py
```

Expected mean centre errors:

| Predictor | Error |
|---|---:|
| Persistence | 0.09886 |
| Constant velocity | 0.04882 |
| Nearest trajectory | 0.02903 |
| ConvLSTM | 0.02965 |

The command writes `results/convlstm_smoke.json` with the full configuration and per-seed results.

## Matched Exposure Pilots and Confirmation

The runner needs the `models` extra (PyTorch). Every report is effect-blinded by default.

```bash
# Discarded, effect-blinded variance pilot on the confirmatory protocol (about 7 minutes on Apple MPS)
uv run --extra models python scripts/exposure_experiment.py \
  --config experiments/exposure_pilot_v5.json --out results/exposure_pilot_v5_blinded.json --device mps

# Pre-treatment balance check using base checkpoints only
uv run --extra models python scripts/exposure_balance_check.py --device mps

# Finite-sample joint power for the decision contract from the blinded pilots (v4 and v5 pooled)
uv run python scripts/plan_confirmation.py

# Empirical checks of the closer-neighbor sampler
uv run python scripts/validate_closer_sampler.py
```

The confirmation (`experiments/exposure_confirmation.json`, 352 blocks, 4.9 to 6.5 hours) is run the same way. Raw block records require `--unblind PATH --freeze-manifest MANIFEST`; the manifest (template: `experiments/freeze_manifest.template.json`) must pass `scripts/freeze.py`. The frozen analysis is `uv run python scripts/analyze_confirmation.py --records PATH --freeze-manifest MANIFEST --out PATH`. Two MPS runs of the same configuration reproduced pilot v3 dispersion exactly.

`scripts/plan_exposure_power.py` and `scripts/exposure_pilot.py` are superseded and kept only to document earlier runs.

## Detector Benchmark

Requires PyTorch, torchvision, and Pillow. Pretrained SSCD, DINOv2, and RAFT-small weights download on first use.

```bash
uv venv .venv-detectors --python 3.11
uv pip install --python .venv-detectors/bin/python torch torchvision pillow numpy h5py
PYTHONPATH=src .venv-detectors/bin/python scripts/detector_benchmark.py
```

The registered configuration is `experiments/detector_benchmark_v4.json`; outputs are `results/detector_benchmark_v4.json` and per-query scores `results/detector_benchmark_v4_scores.json.gz` (about 75 minutes on Apple MPS). Then run `uv run python scripts/analyze_detectors.py` for per-transformation rates, family-bootstrap intervals, the copy-control gate, and hard-constrained recall. Use `--scale 0.03` for a two-minute smoke run.

## Detector Smoke

```bash
uv run python scripts/detector_smoke.py
```

The command runs the frozen 64-family classical-detector check and writes `results/detector_smoke.json`.

## Scope

The software checks, memory positive control, PhyWorld state baselines, and ConvLSTM smoke establish the measurement and feasibility path for the later matched-exposure experiment.
