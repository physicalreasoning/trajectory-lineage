# trajectory-lineage

[![tests](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml/badge.svg)](https://github.com/physicalreasoning/trajectory-lineage/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**controlled audits of trajectory-family overlap in video world-model evaluation**

[[research plan]](docs/RESEARCH_PLAN.md)
[[evaluation protocol]](docs/EVALS.md)
[[reproduction]](docs/REPRODUCE.md)
[[findings]](FINDINGS.md)

## status

**conceptual pilot. there are no empirical findings yet.**

the broad claim that trajectory overlap causes benchmark contamination did not survive adversarial review. this repository begins with the narrower question that remained:

> in a frozen procedural collision setting, how much does deliberate reuse of source-family trajectories change the measured advantage of one controlled training policy over another?

this is an evaluation assay, not a historical provenance detector. it cannot establish that a proprietary model trained on a particular video, website, or dataset.

## why this exists

video deduplication usually asks whether two files or frames look alike. world-model evaluation can have another dependency: two videos can contain different pixels while rendering the same stored physical trajectory.

this repository makes those relationships explicit:

| relationship | meaning |
|---|---|
| `exact_render` | identical rendered tensor |
| `same_trajectory` | different render, identical stored physical states |
| `nearby_trajectory` | independently generated states linked by a declared similarity rule |
| `independent` | no derivation edge or declared neighbor relation |

known lineage lets us run a controlled experiment. it does not tell us what an undocumented model saw during pretraining.

## what we will build

1. a deterministic collision generator with independent state and rendering layers;
2. immutable trajectory, render, clip, split, and exposure manifests;
3. matched continued-training branches with successful-update accounting;
4. bounded autonomous-forecasting measurements that retain invalid outputs;
5. an evaluation-reuse assay over independent events and source-family rerenders; and
6. a findings record that preserves negative results and corrections.

## gate 0: recover before scaling

before any confirmatory model run, the project must establish:

- an executable and frozen event population;
- state-preserving rerenders with distinct pixel hashes;
- a model that beats constant velocity on the easy task;
- a deliberate-exposure positive control;
- generated-output measurement validity;
- measured wall time, memory, storage, and failure rate; and
- a feasible preregistered inference plan.

if these fail after two documented pilot revisions, the experiment stops rather than expanding.

## current code

this initial scaffold implements only the contracts needed before model training:

- exact continuous-time equal-mass collisions;
- deterministic state trajectories;
- independent appearance rerendering;
- lineage relationships and deterministic manifest writing;
- bounded center-error measurement; and
- the evaluation-reuse effect calculation.

it intentionally does **not** include a trained predictor or claim a result.

## setup

python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/smoke.py
```

expected smoke properties:

```text
same_trajectory_hash = true
different_render_hash = true
relationship = same_trajectory
```

## layout

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

## non-goals

- identifying hidden proprietary training records;
- treating similarity as proof of membership;
- claiming a special memorization mechanism;
- extrapolating one small predictor to all video world models;
- counting frames or rerenders as independent training runs; or
- using a successful smoke test as paper evidence.

## license

Apache-2.0. generated datasets and model checkpoints will receive explicit, separate terms before release.
