# Findings

This file is append-only. Corrections should be recorded beside earlier statements rather than replacing them silently.

## 0 · No empirical finding yet (2026-10-06)

Three rounds of claim, hostile scrutiny, repair, and rescrutiny reduced the initial proposal to a conceptual pilot.

What survived:

- New pixels do not necessarily represent an independent physical event;
- Controlled source-family exposure can be randomized and measured;
- A bounded evaluation-reuse estimand can be defined; and
- Negative results can still constrain whether the assay is useful.

What did not survive:

- Claims of proprietary membership or provenance;
- A distinct trajectory-ancestry mechanism;
- Automatic benchmark-contamination language;
- A demonstrated filtering or split-policy correction;
- Practical significance for a 50% reuse mixture;
- Confirmatory feasibility; and
- Publication readiness.

No neural model has been trained in this repository. The smoke result checks software contracts only.

## 1 · Registered memory positive control passes (2026-10-06)

`scripts/pilot_memory.py` ran the configuration frozen in `experiments/memory_positive_control.json` before the result was generated.

| measure | result |
|---|---:|
| independent blocks | 64 |
| mean source-family gain | +0.033367 |
| mean independent-population gain | +0.000117 |
| 50% evaluation-reuse effect | +0.016625 |
| effect standard deviation | 0.002033 |
| normal 95% lower bound | +0.016127 |
| registered effect threshold | +0.015625 |
| runtime | 0.48 seconds |

**The registered gate passes.** The source-memory policy exactly retrieves every source-family future, while its average advantage on independently generated trajectories is effectively zero. The lower bound also happens to exceed the registered effect threshold, although the gate only required it to exceed zero.

**Interpretation:** The software and estimand detect a deliberately memory-sensitive policy. The positive control validates the paired banks, independent-event sample, bounded metric, and aggregation path.

**What this does not establish:** The predictor receives exact state trajectories and performs nearest-neighbor retrieval. The result is intentionally easy and says nothing about a learned pixel world model, visual safeguards, practical benchmark prevalence, pretraining, or proprietary data. The next gate is a small trainable predictor that must beat constant velocity before any source-family comparison is interpreted.

Source: `results/memory_positive_control.json`.
