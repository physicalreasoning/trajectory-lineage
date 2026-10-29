# Trajectory-lineage

[![Tests](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml/badge.svg)](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Controlled audits of trajectory-family overlap in video world-model evaluation**

[[Workstream proposal]](docs/WORKSTREAM_PROPOSAL.md)
[[Data sources]](docs/DATA.md)
[[Effect-size contract]](docs/EFFECT_SIZE.md)
[[Detector audit]](docs/DETECTORS.md)
[[Research plan]](docs/RESEARCH_PLAN.md)
[[Evaluation protocol]](docs/EVALS.md)
[[Reproduction]](docs/REPRODUCE.md)
[[Findings]](FINDINGS.md)

## Status

**Registered 2D study; confirmation runs after registration.** The registered study is the 2D collision design ([Registered 2D study](#registered-2d-study)) with three added axes: predictor capacity, a latent world-model recipe with frozen pretrained encoders, and exposure dose. All of its pilots are effect-blind: they release dispersions, counts and runtime only.

**1D record.** In an independent 16-run 1D interim sample, reusing a physical trajectory under new colors gave a detectable but negligible evaluation advantage (reuse gap +0.0015, 95% CI 0.0011–0.0018; about 6% lower error on reused trajectories), no larger than training on very close independent trajectories. Standard duplicate checks miss this reuse; trajectory matching catches it. The 352-run 1D confirmation finished on October 20 under a provisional freeze (`approvals/provisional_freeze.json`). It is sealed, not analyzed, and not pooled with the 2D study.

The broad claim that trajectory overlap causes benchmark contamination did not survive adversarial review. This repository begins with the narrower question that remained:

> In a frozen procedural collision setting, how much does deliberate reuse of source-family trajectories change the measured advantage of one controlled training policy over another?

This is an evaluation assay, not a historical provenance detector. It cannot establish that a proprietary model trained on a particular video, website, or dataset.

<p align="center">
  <img src="assets/overview.svg" width="900" alt="Trajectory-lineage audit overview">
</p>

> **Figure 1.** A visually new video can still reuse a physical event encountered during training. The audit compares gains on unseen rerenders with gains on independently sampled events.

## Why this exists

Video deduplication usually asks whether two files or frames look alike. World-model evaluation can have another dependency: two videos can contain different pixels while rendering the same stored physical trajectory.

This repository makes those relationships explicit:

| relationship | meaning |
|---|---|
| `exact_render` | the stored render was reused |
| `same_trajectory` | the stored physical state sequence was reused under a different render |
| `derived_trajectory` | a recorded derivation edge (including siblings of one parent) |
| `nearby_trajectory` | independently generated states linked by a declared similarity rule |
| `independent` | documented independent generation |
| `unresolved` | no documented provenance |

Relationships come from provenance, not similarity: equal pixels or equal states are reported separately and never imply lineage. Known lineage lets us run a controlled experiment. It does not tell us what an undocumented model saw during pretraining.

## What we will build

1. A deterministic collision generator with independent state and rendering layers;
2. Immutable trajectory, render, clip, split, and exposure manifests;
3. Matched continued-training branches with successful-update accounting;
4. Bounded autonomous-forecasting measurements that retain invalid outputs;
5. An evaluation-reuse assay over independent events and source-family rerenders; and
6. A findings record that preserves negative results and corrections.

## Gate 0: Recover before scaling

Before any confirmatory model run, the project must establish:

- An executable and frozen event population;
- State-preserving rerenders with distinct pixel hashes;
- A model that beats constant velocity on the easy task;
- A deliberate-exposure positive control;
- Generated-output measurement validity;
- Measured wall time, memory, storage, and failure rate; and
- A feasible preregistered inference plan.

If these fail after two documented pilot revisions, the experiment stops rather than expanding.

## Current code

- Exact continuous-time equal-mass collisions with independent appearance rerendering;
- Provenance-based lineage records and deterministic manifests;
- A 1.09M-parameter ConvLSTM visual-state predictor;
- A matched exposure runner (`scripts/exposure_experiment.py`) with cloned model and optimizer state, bounded metric, role-separated random streams, a separate competence screen, and per-item losses;
- A frozen confirmatory analyzer (`scripts/analyze_confirmation.py`) and freeze gate (`scripts/freeze.py`);
- A duplicate-detector benchmark (`scripts/detector_benchmark.py`, `scripts/analyze_detectors.py`) and an exploratory filtering analysis (`scripts/exposure_filtering.py`); and
- A validated loader for the PhyWorld collision evaluation set;
- A 2D collision simulator (`src/trajectory_lineage/simulator2d.py`) and the registered 2D analyzer (`scripts/analyze_2d.py`);
- Predictor capacity as a registered block factor (ConvLSTM with 16, 64 or 256 hidden channels);
- A latent world-model predictor (`src/trajectory_lineage/latent.py`): a frozen DINOv2-small or V-JEPA 2-L encoder and a trained width-256 transformer;
- A joint analyzer with one Holm family across the 2D confirmation and its axes (`scripts/analyze_exp3_axes.py`); and
- Parallel block workers on one GPU, blinded resumable checkpoints for pilots, and per-block state hashes.

Claims are limited to this synthetic assay and the predictors listed; no claim is made about policy success or released world models.

## First local pilot

The registered nearest-trajectory memory baseline ran across 64 independent blocks:

| metric | result |
|---|---:|
| source-family gain | 0.03337 |
| independent-population gain | 0.00012 |
| 50% evaluation-reuse effect | 0.01662 |
| normal 95% lower bound | 0.01613 |
| registered threshold | 0.015625 |
| runtime | 0.48 seconds |
| gate | **pass** |

This is the expected positive control: a predictor that literally stores source trajectories is perfect on those trajectories and provides essentially no advantage on independent events. It shows that the lineage, paired-policy, metric, and result pipeline can detect deliberate memory. It does not show that a learned video world model behaves this way.

<p align="center">
  <img src="assets/memory-positive-control.svg" width="820" alt="Memory positive-control results">
</p>

> **Figure 2.** The pre-specified memory positive control passes across 64 independent simulation blocks.

## PhyWorld baseline

The public PhyWorld collision evaluation set is integrated as an external validation source. With the first 4 of 32 state frames observed, one-nearest-neighbor trajectory prediction outperforms constant-velocity extrapolation on the within-group split and both inspected cross-group transfers.

| Split | Constant velocity | Nearest trajectory |
|---|---:|---:|
| Within `00000` | 0.2577 | 0.1653 |
| `00000` → `00001` | 0.4729 | 0.2329 |
| `00000` → `00002` | 0.5090 | 0.2604 |

<p align="center">
  <img src="assets/phyworld-baselines.svg" width="820" alt="PhyWorld state-baseline results">
</p>

> **Figure 3.** Nearest-trajectory retrieval is a stronger baseline than constant-velocity extrapolation on all three evaluated PhyWorld splits.

See [Data Sources](docs/DATA.md), [Findings](FINDINGS.md), and the committed machine-readable result for the full protocol and uncertainty intervals.

## ConvLSTM smoke

A compact pixel-conditioned ConvLSTM was trained for 300 updates on 512 generated collision videos and evaluated on 128 independently generated videos, across three seeds. It consumes 8 RGB frames and predicts 16 future object-centre states.

| Predictor | Mean centre error |
|---|---:|
| Persistence | 0.09886 |
| Constant velocity | 0.04882 |
| Nearest trajectory | **0.02903** |
| ConvLSTM | 0.02965 |

The ConvLSTM reduces error by 39.3% relative to constant velocity and lands within 2.1% of nearest-trajectory retrieval.

## Matched exposure experiment (1D)

Each independent training run trains a base ConvLSTM on 512 events, then continues matched copies (identical updates, batch positions, and 12–13 presentations per example) on independent trajectories (B), the evaluation-source trajectories under new appearances (C), independent trajectories closer than anything in base training (D), banded neighbors (D2), or the exact evaluation renders (E). Thresholds were fixed before any result was observed (`docs/EFFECT_SIZE.md`): consequential if the reuse gap's lower bound exceeds 0.010, negligible if its interval lies within ±0.002, harm margin 0.001.

| Interim (independent 16-run sample; 15 screened) | Estimate | 95% CI |
|---|---:|---:|
| Reuse gap G | +0.00146 | [+0.00113, +0.00179] |
| Source gain | +0.00172 | [+0.00071, +0.00273] |
| Harm on independent events | −0.00026 | [−0.00108, +0.00057] |
| Same trajectory vs closer independent neighbor | +0.00008 | [−0.00017, +0.00032] |

Outcome under the contract: **negligible differential gap**, while still detectable. Reuse would need to make up 56–89% of an evaluation set to move a model comparison by the 0.001 margin, and the advantage matches that from very close independent trajectories, consistent with local generalization. See `FINDINGS.md` §5 and §7.

## Registered 2D study

Two disks with one exact elastic collision in 2D. Each block trains a base model, then five matched continuations (arms B, C, D, D2, E, as in 1D). The contrasts and margins are the same as in 1D, with two revised rules (whole-interval prevalence rule, reachable positive control).

| Configuration | Predictor | Reused sources | Blocks | Role |
|---|---|---|---:|---|
| `exposure2d_confirmation` | ConvLSTM, 16 / 64 / 256 channels | 64 | 800 + 400 + 400 | primary claim (16 channels); capacity secondary |
| `exposure2d_confirm_latent_dinov2` | frozen DINOv2-small + latent transformer | 64 | 200 | architecture secondary |
| `exposure2d_confirm_latent_vjepa2` | frozen V-JEPA 2-L + latent transformer | 64 | 200 | architecture secondary |
| `exposure2d_confirm_dose16` | ConvLSTM, 16 channels | 16 (about 50 presentations each) | 200 | dose secondary |
| `exposure2d_confirm_dose256` | ConvLSTM, 16 channels | 256 (about 3 presentations each) | 200 | dose secondary |

Effect-blind pilots (24 blocks per level; capacity, architecture and dose) all completed with every block competent, the positive control passing and no failures. Their blinded reports are in `results/*_blinded.json`. The confirmation starts only with an approved freeze manifest. The runner checkpoints every block, and the analyzers refuse to run without the manifest.

```bash
# effect-blind pilot on a GPU, three blocks at a time, resumable
PYTHONPATH=src python scripts/exposure_experiment.py --config experiments/exposure2d_latent_vjepa2_pilot_v1.json \
  --out results/latent_vjepa2_pilot_v1_blinded.json --device cuda --workers 3 \
  --blinded-checkpoint ck.jsonl --blinding-key DO_NOT_OPEN/key.json
```

The latent predictor needs `transformers` (`pip install transformers`). On an A100 with three workers, a block takes about 20 s to 2.5 min of wall time, depending on the configuration.

## Duplicate-detector audit

A registered benchmark (native 224 px, 8 transformations, 14 detector variants, calibrated query-level false removal, a conventional-copy positive control, and an independent-seed robustness check) found that perceptual similarity is dominated by appearance rather than trajectory:

| Detector (nominal 1% false removal) | Re-encoded copy | Same trajectory, new appearance | Different nearby trajectory, same appearance, removed |
|---|---:|---:|---:|
| Exact hash | 0% | 0% | 0% |
| SSCD, aligned + normalized (passes copy gate) | 58.5% | 11.5% | 31% |
| OFS-k motion-copy score (published; fails copy gate) | 2% | 3.5% | — |
| Centroid tracks + mirror/time search (fails gate: JPEG corrupts extraction) | 0% | 100% | 0% |

Trajectory-aware matching recovers rerenders under appearance, texture, mirror, and time-shift changes when extraction is reliable, and calibrating only on unrestricted negatives removes most independent events that lie closer to a training trajectory than any other training clip. See `FINDINGS.md` §6 and `docs/DETECTORS.md`.

Applied to the exposure experiment (8 interim runs, exploratory), SSCD filtering removes 12% of reused clips and leaves the distortion unchanged, like random removal of equal size, while trajectory matching removes all reused clips and keeps 99.2% of independent clips (`FINDINGS.md` §7).

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # add --extra models for torch (model tests and experiments)
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/smoke.py
```

Expected smoke properties:

```text
same_trajectory_hash = true
different_render_hash = true
relationship = same_trajectory
```

## Layout

```text
src/trajectory_lineage/  simulator, lineage contracts, bounded metrics
scripts/                 one executable experiment per file
results/                 committed machine-readable results
experiments/             frozen experiment configurations
approvals/               freeze manifests binding configuration, code, and analyzer
tests/                   physics, lineage, and measurement invariants
docs/
  RESEARCH_PLAN.md        gates and kill criteria
  SPEC.md                 lineage and simulator contracts
  EVALS.md                metrics and claim boundaries
  REPRODUCE.md            exact commands and expected outputs
  EFFECT_SIZE.md          pre-specified thresholds and decision rule
  DETECTORS.md            duplicate-detector audit protocol and results
FINDINGS.md               append-only findings and corrections
```

## Non-goals

- Identifying hidden proprietary training records;
- Treating similarity as proof of membership;
- Claiming a special memorization mechanism;
- Extrapolating one small predictor to all video world models;
- Counting frames or rerenders as independent training runs; or
- Using a successful smoke test as paper evidence.

## License

Apache-2.0. Generated datasets and model checkpoints will receive explicit, separate terms before release.
