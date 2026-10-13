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

## 2 · PhyWorld state baselines establish headroom (2026-10-12)

`scripts/phyworld_state_baselines.py` ran the protocol frozen in `experiments/phyworld_state_baselines.json` against the public PhyWorld collision evaluation file with SHA-256 `63904280119e7f73af782713c64350ebe9c4bf318ca93c1a0130e5c0714ad865`.

The task observes 4 of 32 ground-truth state frames and predicts the remaining 28. The nearest-trajectory baseline uses visible radii and observed positions; source velocity parameters are excluded.

| Split | Test examples | Constant velocity | Nearest trajectory | Difference [bootstrap 95% interval] | Nearest better |
|---|---:|---:|---:|---:|---:|
| Within group `00000` | 187 | 0.2577 | 0.1653 | −0.0924 [−0.1200, −0.0662] | 58.3% |
| `00000` → `00001` | 260 | 0.4729 | 0.2329 | −0.2401 [−0.2678, −0.2137] | 86.2% |
| `00000` → `00002` | 443 | 0.5090 | 0.2604 | −0.2486 [−0.2627, −0.2333] | 92.8% |

Errors are mean Euclidean disk-center distances across future frames, normalized by the source world scale. Negative differences favor nearest-trajectory prediction.

**Finding:** Constant velocity is not a sufficient baseline for the collision set. A one-nearest-neighbor predictor substantially reduces average future-state error in the within-group split and both inspected cross-group transfers.

**Research consequence:** The learned pixel predictor must be compared with both baselines. Beating constant velocity alone is too weak; the next model gate should require improvement over, or a clearly different generalization profile from, nearest-trajectory retrieval.

**Limits:** The analysis uses source-provided state arrays rather than decoded pixels. The HDF5 groups are reported by identifier because the downloaded file does not encode a semantic split name in its group keys. Bootstrap intervals describe variation across dataset examples, not independently trained models.

Source: `results/phyworld_state_baselines.json`.

## 3 · ConvLSTM visual-state smoke passes its learning gate (2026-10-12)

`scripts/convlstm_smoke.py` ran the configuration frozen in `experiments/convlstm_smoke.json` on three seeds. Each run trained a 1,085,552-parameter ConvLSTM for 300 updates on 512 generated collision videos and evaluated 128 independently generated videos. The model received 8 RGB frames and predicted the next 16 object-centre states.

| Predictor | Mean centre error | Seed SD |
|---|---:|---:|
| Persistence | 0.09886 | 0.00062 |
| Constant velocity | 0.04882 | 0.00493 |
| Nearest trajectory | **0.02903** | 0.00104 |
| ConvLSTM | 0.02965 | 0.00160 |

**The registered smoke gate passes.** Mean ConvLSTM error is below persistence and constant velocity on all three deterministic training seeds. Total runtime was 35.5 seconds on Apple MPS.

**Finding:** A compact pixel-conditioned recurrent model learns enough collision structure to reduce centre error by 39.3% relative to the privileged constant-velocity baseline. It does not beat the state-space retrieval baseline on the three-seed mean: ConvLSTM error is 0.00062 higher, approximately 2.1% relative to retrieval.

**Research consequence:** The model is usable for the discarded exposure pilot, but retrieval remains the stronger baseline. The next experiment should compare matched continuation branches and preserve the per-seed model-versus-retrieval result rather than using only the easier persistence and constant-velocity gates.

Source: `results/convlstm_smoke.json`.
