# Trajectory-lineage

[![Tests](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml/badge.svg)](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Controlled audits of trajectory-family overlap in video world-model evaluation**

[[Workstream proposal]](docs/WORKSTREAM_PROPOSAL.md)
[[Research plan]](docs/RESEARCH_PLAN.md)
[[Evaluation protocol]](docs/EVALS.md)
[[Reproduction]](docs/REPRODUCE.md)
[[Findings]](FINDINGS.md)

## Status

**Conceptual pilot. The registered memory positive control passes; no neural world model has been trained.**

The broad claim that trajectory overlap causes benchmark contamination did not survive adversarial review. This repository begins with the narrower question that remained:

> In a frozen procedural collision setting, how much does deliberate reuse of source-family trajectories change the measured advantage of one controlled training policy over another?

This is an evaluation assay, not a historical provenance detector. It cannot establish that a proprietary model trained on a particular video, website, or dataset.

## Why this exists

Video deduplication usually asks whether two files or frames look alike. World-model evaluation can have another dependency: two videos can contain different pixels while rendering the same stored physical trajectory.

This repository makes those relationships explicit:

| relationship | meaning |
|---|---|
| `exact_render` | identical rendered tensor |
| `same_trajectory` | different render, identical stored physical states |
| `nearby_trajectory` | independently generated states linked by a declared similarity rule |
| `independent` | no derivation edge or declared neighbor relation |

Known lineage lets us run a controlled experiment. It does not tell us what an undocumented model saw during pretraining.

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

This initial scaffold implements only the contracts needed before model training:

- Exact continuous-time equal-mass collisions;
- Deterministic state trajectories;
- Independent appearance rerendering;
- Lineage relationships and deterministic manifest writing;
- Bounded center-error measurement; and
- The evaluation-reuse effect calculation.

It intentionally does **not** include a trained neural predictor or claim a world-model result.

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

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
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
results/                 committed machine-readable pilot results
experiments/             future frozen experiment configurations
tests/                   physics, lineage, and measurement invariants
docs/
  RESEARCH_PLAN.md        gates and kill criteria
  SPEC.md                 lineage and simulator contracts
  EVALS.md                metrics and claim boundaries
  REPRODUCE.md            exact commands and expected outputs
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
