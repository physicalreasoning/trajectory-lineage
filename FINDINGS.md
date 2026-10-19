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

At the time of this smoke test no neural model had been trained in this repository; the smoke result checked software contracts only. (A ConvLSTM was trained later; see finding 3.)

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

**Initial finding:** On the registered all-state metric, one-nearest-neighbor prediction substantially reduces average future-state error in the within-group split and both inspected cross-group transfers.

**Visibility sensitivity analysis:** The source trajectories continue after objects leave the nominal scene. Restricting the score to frames where both disks are fully visible reverses the baseline ordering:

| Split | Included examples | Constant velocity | Nearest trajectory | Nearest minus constant [bootstrap 95% interval] |
|---|---:|---:|---:|---:|
| Within `00000` | 187 | 0.0643 | 0.1054 | +0.0411 [+0.0317, +0.0519] |
| `00000` → `00001` | 260 | 0.0597 | 0.1083 | +0.0486 [+0.0392, +0.0574] |
| `00000` → `00002` | 442 | 0.0634 | 0.1240 | +0.0605 [+0.0513, +0.0695] |

One group-`00002` example has no future frame in which both disks remain fully visible and is excluded from this sensitivity calculation. Each included example is first averaged over its eligible frames, then examples are weighted equally.

**Finding:** The evaluation window determines which simple baseline wins. Retrieval is stronger for full-state extrapolation, especially after objects leave view; constant velocity is stronger while both disks remain visible.

**Research consequence:** The learned predictor must be reported separately on visible dynamics, off-screen continuation, and invalid outputs. “Beats constant velocity” and “beats retrieval” are not one common capability gate.

**Limits:** The analysis uses source-provided state arrays rather than decoded pixels. The HDF5 groups are reported by identifier because the downloaded file does not encode a semantic split name in its group keys. Bootstrap intervals describe variation across dataset examples with fixed retrieval banks and partitions, not independently trained models.

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

## 4 · Classical detector smoke distinguishes clean rerenders (2026-10-15)

`scripts/detector_smoke.py` ran the configuration frozen in `experiments/detector_smoke.json` over 64 independently generated source families. Each family contains a source rendering, an appearance-only rerender of the same stored trajectory, and an independently sampled collision.

| Detector | Same-trajectory recall | Independent false-positive rate |
|---|---:|---:|
| Exact video hash | 0.0 | 0.0 |
| Color-invariant silhouette | 1.0 | 0.0 |
| Foreground-centroid motion | 1.0 | 0.0 |
| Oracle state trajectory | 1.0 | 0.0 |

**The registered smoke gate passes.** Exact hashing misses every appearance-only rerender, while the clean synthetic silhouette, motion, and oracle descriptors recover every same-trajectory pair without matching the independently sampled controls.

**Scope:** This is a software and protocol check on solid-color synthetic renders with zero-distance thresholds. It does not establish performance under compression, clutter, camera motion, partial visibility, nearby trajectories, or real video. Those conditions require calibrated development/test families and stronger SSCD, DINOv2, temporal-alignment, and optical-flow baselines.

Source: `results/detector_smoke.json`.

## 5 · Matched learned-model exposure protocol is ready for confirmation; effect remains blinded (2026-10-16)

Five discarded, effect-blinded pilots shaped the confirmatory protocol (`experiments/exposure_confirmation.json`, status `draft_pending_signoff`). Each pilot released only contrast dispersion, runtime, pre-treatment balance, design descriptors, and a binary exact-render positive control. **No learned-model reuse effect estimate has been inspected or released.**

| Pilot | Finding that changed the design |
|---|---|
| v1 | Nearby events were perturbations of sources (derived, not independent) |
| v2 | Code audit: overlapping random streams across blocks, uncapped metric, failed blocks excluded |
| v3 | Band neighbors were farther than each source's nearest base-training event for 86% of sources |
| v4 | Box half-widths capped at half a parameter's range excluded valid band members; competence screen reused the scored population |
| v5 | Final protocol: 8/8 blocks completed and competent, positive control passed, 12–13 presentations per source |

The final protocol has five matched continuation arms from shared model and optimizer state—independent replacements (B), appearance-changed rerenders of the evaluation sources (C), independent neighbors closer than anything in base training (D, median distance 0.0061 versus 0.0143 for the nearest base event), banded independent neighbors (D2), and exact renders (E)—evaluated on held-out source renders, exact renders, and 512 independent events with a bounded metric.

Supporting checks:

- **Determinism.** Two MPS runs of pilot v3 reproduced every dispersion exactly.
- **Balance.** Across 24 base-only blocks, base source − independent error was −0.0006 (95% CI −0.0015 to 0.0003); one base model in 24 failed to beat constant velocity, motivating a pre-treatment competence screen on a separate population.
- **Neighbor sampler.** `results/closer_sampler_validation.json`: enlarging the proposal box 6× → 9× produced no detected marginal difference in any parameter's offset distribution (largest KS 0.092 vs approximate critical 0.124; draws are clustered within 30 sources, so the cutoff is approximate and this is evidence, not proof, of containment), and box sampling showed no detected difference from reservoir selection on a band both can reach (KS 0.083 vs 0.175). D is therefore defined as the box-conditioned band distribution.
- **Power.** Pooled pilots v4+v5 (14 df): the harm gate is binding because continuation shifts the whole model by a block-specific amount (corr(S, H) = −0.95). With 336 analyzed blocks, joint power is 0.99 when true harm is half the margin at the pooled 80% upper SD (0.95 using v5 alone); size at the threshold is about 0.025.

The pre-specified contract (`docs/EFFECT_SIZE.md`) uses a declared ranking margin m = 0.001, a consequential threshold of 0.010 (m at 10% prevalence), a negligible bound of ±0.002 (m at 50% prevalence), and a harm margin of 0.001, with outcomes reported on separate magnitude, safeguard, and identity-specificity axes. Unblinding requires a manifest binding the configuration, a clean code commit, the analyzer `scripts/analyze_confirmation.py`, and named sign-offs from the paper lead and the evaluation-science reviewer.

Source: `results/exposure_pilot_v5_blinded.json`, `results/confirmation_power.json`, `results/exposure_balance_check.json`, `results/closer_sampler_validation.json`.

## 6 · Perceptual similarity is dominated by appearance, not trajectory (2026-10-16)

`scripts/detector_benchmark.py` ran the registered `experiments/detector_benchmark_v4.json` from clean commit `fd14187` (native 224 px; 500-trajectory gallery; per split 200 positives, 200 independent hard negatives closer to a target than its nearest other gallery trajectory, and 600 independent easy-negative families; 8 transformations; 14 detector variants). `scripts/analyze_detectors.py` computed per-transformation rates with family-bootstrap intervals that propagate calibration uncertainty, a copy-control gate, hard-constrained recall, and track validation against simulator positions.

Pre-registered gate (necessary, not sufficient): copy-control AUROC ≥ 0.95 and recall ≥ 0.80 at 5%. The copy control re-encodes the identical render; its negatives share gallery appearances.

| Finding | Evidence (test split) |
|---|---|
| Only SSCD with alignment and background normalization passes the gate among practical detectors | Copy AUROC 0.96, recall 0.85 at 5%; DINOv2, RAFT, silhouettes, and centroid tracks fail |
| It misses most same-trajectory rerenders | Recall 0.12 (bootstrap 0.07–0.16) at 1% and 0.22 at 5% under appearance change; 0.01–0.06 at 1% with texture, compression, crop, mirror, time shift, or distractors |
| It flags different trajectories that share appearance | In the copy control it removes 31% of same-appearance closer independent trajectories at 1% (56% at 5%); copy-versus-hard AUROC 0.70 shows some, but limited, trajectory information |
| The published OFS-k motion-copy detector (Chen et al., 2024) also fails the gate | Copy AUROC 0.57 (k = 3; 0.58 for k = 5, 10) although it ranks the true source first for 51–67% of copies; same-trajectory rerender recall 3.5% (bootstrap 1–6%) at 1%; adapted to 128 px flow on a 16 × 16 grid, without natural-motion filtering (no panning or static frames here) |
| Trajectory-aware matching recovers rerenders when extraction works | Centroid tracks with mirror and time-offset search: 100% recall under appearance, texture, mirror, and time shift at 1.7–2.5% false removal; tracks are within 0.3–0.4 px of truth there, but JPEG corrupts extraction (24–27 px), and crop breaks the matcher's coordinates although extraction stays accurate |
| Easy-negative calibration over-removes close independent events | 69% (centroid, appearance) and 53% (state reference) at the pooled 1% point despite perfect separation (a state-distance removal rule of score > −10⁻⁶ keeps 100% recall with no negative removals); a hard-negative constraint keeps recall at 1.00 with 1.0–2.5% hard removal |
| Pooled calibration does not control per-transformation false removal | Up to 5.8% (silhouette, appearance) and 7.8% (normalized SSCD, compression) at the pooled 1% point |

**Robustness check:** v3 (a changed protocol with independent seeds; superseded hard-negative sampler; fresh-palette copy negatives) gave the same headline values (copy recall 0.84 at 5%; rerender recall 0.12 at 1%; invariant centroid recall 1.00). **Scope:** synthetic collisions; full clips at 224 px (curation setting), not the ConvLSTM's 64 px prefixes; one gallery per split; 600 independent negative families limit precision at 1%. The 64 px v2 run failed its copy control and supports no perceptual conclusion.

Source: `results/detector_benchmark_v4.json`, `results/detector_benchmark_v4_scores.json.gz`, `results/detector_benchmark_v4_analysis.json`; OFS-k on the identical clips: `results/detector_benchmark_v4_ofs_analysis.json` (v3: `results/detector_benchmark_v3*`); weights and environment: `results/detector_weights_manifest.json`, `results/detector_environment.txt`.
