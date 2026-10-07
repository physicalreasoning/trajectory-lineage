# Trajectory Lineage Audit: Hidden Event Reuse in Video World-Model Evaluation (Lovejesh)

## Problem

Video and world-model evaluations usually deduplicate files or visually similar frames. However, a test video can contain new pixels while rendering the same underlying physical trajectory that appeared during training.

This creates a hidden train–test dependency: a model may perform well on a new-looking video because it has already seen the event, not because it learned dynamics that transfer to independently generated events.

## Hypothesis

Continued training on designated source trajectories improves prediction on unseen rerenders of those trajectories more than matched continued training on independent replacement trajectories.

The source-specific gain is larger than the gain on unrestricted independently generated events. If this occurs, a render-disjoint evaluation can overstate independent-event generalization even when exact-file and frame-level duplicate checks pass.

## Setup

**Initial environment:** Analytic two-disk elastic collisions in a unit arena

**Observation:** 8 RGB frames at 64 × 64

**Prediction:** 16 future frames

**Primary model:** Small ConvLSTM pixel predictor implemented through OpenSTL

**Positive-control model:** Nearest-trajectory memory predictor

**Baselines:**

- Last-frame persistence
- Constant-velocity extrapolation
- Analytic physics from measured state
- Nearest trajectory by observed-state distance

**Lineage levels:**

1. Exact render: identical rendered tensor
2. Same trajectory: different appearance, identical stored physical states
3. Nearby trajectory: independently generated states linked by a frozen distance rule
4. Independent event: no derivation edge or declared neighbor relationship

Every trajectory, render, split, and consumed training example is recorded in an immutable lineage manifest.

## Method

### 1. Build the Controlled Population

Generate collision and matched near-miss trajectories while varying initial position, velocity, impact timing, and appearance.

For every designated source trajectory, create multiple appearance-only rerenders from the same stored state sequence. Independently sample nearby and unrestricted population events.

The initial pilot uses:

- 32 source families per block
- 32 matched replacement families per block
- 128 unrestricted independent evaluation events per block
- 8 discarded independent training blocks for runtime and variance estimation

The confirmatory block count is selected only after the pilot measures block-level variance and compute cost.

### 2. Validate the Assay

Run a deliberately memory-sensitive nearest-trajectory predictor before training a neural model.

The registered positive-control gate requires:

- Mean 50% evaluation-reuse effect greater than 1/64 arena width
- Normal-approximation 95% lower bound above zero
- Zero source-bank prediction loss for the source-memory policy

Passing this gate validates the lineage, paired-policy, metric, and aggregation pipeline. It does not validate the research hypothesis for a learned world model.

### 3. Train the Predictor

Train one base ConvLSTM per independent block. Clone the base model and optimizer state into matched continued-training branches.

Use identical update counts, learning-rate schedules, replacement locations, and presentation counts across branches.

### 4. Evaluate on Separate Populations

Evaluate every branch on:

- The exact exposure render
- Unseen rerenders of source trajectories
- Nearby but independently generated trajectories
- Unrestricted independently sampled events

Report every component separately so worse performance on independent events cannot masquerade as better source-family learning.

### 5. Audit Overlap Detection

Compare the known lineage graph with:

- Exact byte and frame hashes
- Perceptual frame similarity
- Temporal or motion matching
- State-trajectory distance

Measure which source-family relationships each detector misses and the independent-event false-removal rate.

This detector analysis is diagnostic. It does not establish that retraining on a filtered dataset improves the model.

## Arms

| Arm | Description |
|---|---|
| A | Base predictor before continued training |
| B | Continued training on independent, event-matched replacement trajectories |
| C | Continued training on unseen rerenders of designated source trajectories |
| D | Continued training on nearby but independently generated trajectories |
| F | Optional diagnostic: continued training on the exact evaluation render |

The primary comparison is C versus B. D distinguishes same-event reuse from ordinary local generalization. F separates exact-pixel effects from trajectory-level effects.

## Metrics

**Base forecasting measure:** Mean disk-center error across all predicted object frames, normalized by arena width and capped per object-frame

**Source-family gain:** Error(B on source rerenders) − Error(C on source rerenders)

**Independent-population gain:** Error(B on independent events) − Error(C on independent events)

**Source-specific generalization gap:** Source-family gain − independent-population gain

**Evaluation-reuse effect at prevalence π:**

```text
π × (source-family gain − independent-population gain)
```

**Secondary metrics:**

- Collision or near-miss outcome accuracy
- Outgoing-velocity error
- Object persistence failure rate
- Invalid-frame penalty contribution
- Nearest-training-trajectory distance
- Overlap-detector recall
- Independent-event false-removal rate
- Retained event and difficulty coverage

**Statistics:**

- Independent training block is the unit of replication
- Paired block-level effects and 95% confidence intervals
- Individual-block plots and leave-one-block-out sensitivity
- Family bootstrap only for additional stimulus uncertainty
- Confirmatory block count determined from discarded-pilot variance

Frames, rerenders, checkpoints, and repeated forecasts are not independent training replications.

## Comparisons

**C versus B:** Does same-trajectory exposure create a source-specific forecasting advantage?

**C versus D:** Is the advantage larger than ordinary generalization from nearby independent events?

**F versus C:** How much of the advantage is specific to exact pixels rather than the underlying trajectory?

**A versus B:** Does continued training itself change independent-event performance?

**Source-family versus independent evaluation:** Does render-disjoint evaluation overstate independent-event generalization?

## Status

A local repository scaffold is complete at:

```text
~/Desktop/research/trajectory-lineage
```

Implemented:

- Analytic collision simulator
- State-preserving rerendering
- Trajectory and render hashing
- Lineage relationships and manifest writing
- Bounded center-error measurement
- Registered experiment configurations
- GitHub Actions workflow
- Research plan, evaluation protocol, findings record, and reproduction guide
- 9 passing tests

### Registered Memory Positive Control

| Metric | Result |
|---|---:|
| Independent blocks | 64 |
| Mean source-family gain | 0.033367 |
| Mean independent-population gain | 0.000117 |
| 50% evaluation-reuse effect | 0.016625 |
| Effect standard deviation | 0.002033 |
| Normal 95% lower bound | 0.016127 |
| Registered threshold | 0.015625 |
| Runtime | 0.48 seconds |
| Gate | **Pass** |

The positive control confirms that the assay detects deliberate trajectory memory. No neural world model has been trained yet.

The GitHub organization repository has not been created because the current account does not have `CreateRepository` permission in `physicalreasoning`.

## Next

1. Confirm with Jiankai that physical-event lineage is in scope for the training-data paper.
2. Confirm the minimum result that would change the paper’s conclusion.
3. Add the ConvLSTM predictor and verify that it beats constant velocity.
4. Run the 8-block discarded runtime and variance pilot.
5. Validate tracking on generated failures, not only ground-truth renders.
6. Freeze the population, arms, primary endpoint, effect threshold, and confirmatory analysis.
7. Run confirmation only if the powered design fits the committed compute budget.

## Risks

**Question mismatch:** If “training-data source” means identifying websites, creators, or proprietary datasets, this experiment is adjacent rather than central.

**C ≈ B:** Same-trajectory exposure adds no measurable advantage. Report the negative result and stop this version.

**C ≈ D:** The result is explained by ordinary local generalization rather than event reuse.

**Source gain is caused by independent-event degradation:** Report the component losses; do not call this improved source learning.

**Tracker or invalid-output penalties dominate:** Repair measurement before interpreting the model.

**Visual or temporal matching catches all consequential overlap:** The proposed hidden-overlap failure does not exist under the tested safeguard.

**Powered replication is unaffordable:** Stop or renegotiate scope. More frames or rerenders cannot replace independent training blocks.

**Toy-domain limitation:** A positive collision result does not establish behavior in frontier video world models. Any broader claim requires a separately approved replication.

## Claim Boundary

This work can establish a causal effect of controlled trajectory-family exposure in the tested model and environment.

It cannot establish:

- Historical membership in a proprietary training set
- Original website or dataset provenance
- A special internal memorization mechanism
- Universal behavior across video world models
- That visually similar examples are necessarily copied
- That filtering detected overlap improves a retrained model
- Human-like physical reasoning
