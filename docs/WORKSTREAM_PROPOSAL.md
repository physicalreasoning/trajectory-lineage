# Trajectory Lineage Audit: Measuring Hidden Train–Test Reuse in Video World Models (Lovejesh)

## Problem

Video and world-model benchmarks typically deduplicate exact files or visually similar frames. This project extends deduplication to the underlying physical event shared across multiple derived videos.

A test video may contain different pixels, colors, textures, or camera settings while reproducing a physical trajectory that appeared during training. In that case, the evaluation item looks new but is not independent of the training data. A model may therefore appear to have learned transferable dynamics when it has instead benefited from prior exposure to the same event.

This project studies whether trajectory-level train–test overlap inflates world-model evaluation results after exact-file and frame-level duplicate checks pass.

## Hypothesis

A video predictor continued-trained on designated source trajectories will improve more on unseen rerenders of those trajectories than on unrestricted, independently generated events, relative to a predictor receiving the same number of updates on independent replacement trajectories.

This source-specific advantage will be larger than the advantage produced by training on nearby but independently generated trajectories. If supported, the result would show that render-disjoint evaluation can overstate independent-event generalization.

## Setup

**Initial environment:** Analytic two-disk elastic collisions in a unit arena

**Observation:** 8 RGB frames at 64 × 64 resolution

**Prediction horizon:** 16 future frames

**Primary predictor:** Small ConvLSTM pixel predictor using OpenSTL

**Positive-control predictor:** Nearest-trajectory memory baseline

**Baselines:**

- Last-frame persistence
- Constant-velocity extrapolation
- Analytic physics using measured state
- Nearest trajectory by observed-state distance

**Lineage categories:**

1. **Exact render:** The rendered tensors are identical.
2. **Same trajectory:** The pixels differ, but the videos are rendered from the same stored physical state sequence.
3. **Nearby trajectory:** The events are independently generated, match the same event class, and fall within a physical-distance band defined over standardized initial states and future trajectories. The distance rule is frozen before model training.
4. **Independent event:** The event has no derivation edge or declared neighbor relationship with the training example.

Every trajectory, render, split assignment, transformation, and consumed training example is recorded in an immutable lineage manifest.

## Method

### 1. Construct the Controlled Population

Generate collision and matched near-miss trajectories while varying initial position, velocity, impact timing, and visual appearance.

For each designated source trajectory, create multiple appearance-only rerenders from the same stored state sequence. Separately sample nearby trajectories and unrestricted independent events.

The discarded feasibility pilot begins with:

- 32 source families per independent block
- 32 matched replacement families per block
- 128 unrestricted independent evaluation events per block
- 8 independent blocks for runtime and variance estimation

The confirmatory block count is determined only after the pilot measures block-level variance and actual compute cost.

### 2. Validate the Evaluation Assay

Before training a neural predictor, run a deliberately memory-sensitive state-space nearest-trajectory baseline.

The pre-specified positive-control gate requires:

- A mean 50% evaluation-reuse effect greater than 1/64 arena width, approximately one pixel at 64 × 64 resolution
- A normal-approximation 95% lower bound above zero
- Zero source-bank prediction loss for the source-memory policy

Passing this gate establishes that the lineage records, paired policies, metric implementation, and aggregation pipeline are sensitive to deliberate trajectory memory before the learned-model experiment begins.

### 3. Train the Predictor

Train one base ConvLSTM for each independent block. Clone the model and optimizer state into matched continued-training branches.

Hold the following constant across branches:

- Number of optimizer updates
- Learning-rate schedule
- Batch positions receiving replacement examples
- Number of distinct replacement examples
- Presentation count per example
- Evaluation checkpoint

Across the matched continued-training branches, the intended treatment is the lineage relationship between the replacement examples and the evaluation population. Event class, example count, repetition, optimization, and evaluation timing remain matched.

### 4. Evaluate Distinct Generalization Populations

Evaluate every branch on:

- The exact exposure render
- Unseen rerenders of source trajectories
- Nearby but independently generated trajectories
- Unrestricted independently sampled events

Report every component separately. A positive headline effect must not be interpreted as improved source learning if it is caused by degraded performance on independent events.

### 5. Audit Overlap Detection

Compare the known lineage graph against commonly available overlap checks:

- Exact byte and frame hashes
- Perceptual frame similarity
- Temporal or motion matching
- State-trajectory distance

For each method, measure:

- Recall on known source-family relationships
- False removal of independent events
- Retained event and difficulty coverage

This analysis quantifies which forms of source-family overlap each detection method captures and the evaluation coverage retained at its selected operating point.

## Experimental Arms

| Arm | Description |
|---|---|
| A | Base predictor before continued training |
| B | Continued training on independent, event-matched replacement trajectories |
| C | Continued training on one render of each designated source trajectory; evaluation uses separate, unseen rerenders of the same state sequences |
| D | Continued training on nearby but independently generated trajectories |
| E | Optional diagnostic: Continued training on the exact evaluation render |

The primary comparison is **C versus B**. Arm D distinguishes same-event reuse from ordinary local generalization. Arm E separates exact-pixel effects from trajectory-level effects.

## Metrics

### Primary Forecasting Measure

Mean disk-center error across all predicted object frames, normalized by arena width and capped per object-frame. Invalid frames remain in the denominator and receive a predefined penalty.

### Derived Effects

**Source-family gain:**

```text
Error(B on source rerenders) − Error(C on source rerenders)
```

**Independent-population gain:**

```text
Error(B on independent events) − Error(C on independent events)
```

**Source-specific generalization gap:**

```text
Source-family gain − Independent-population gain
```

**Evaluation-reuse effect at prevalence π:**

```text
π × (Source-family gain − Independent-population gain)
```

### Secondary Metrics

- Collision versus near-miss accuracy
- Outgoing-velocity error
- Object-persistence failure rate
- Invalid-frame penalty contribution
- Distance to the nearest consumed training trajectory
- Overlap-detector recall
- Independent-event false-removal rate
- Retained event and difficulty coverage

### Statistical Analysis

- The independent training block is the unit of replication.
- Report paired block-level effects with 95% confidence intervals.
- Show every individual block and leave-one-block-out sensitivity.
- Use family-level bootstrap intervals only for additional stimulus uncertainty.
- Determine the confirmatory block count from discarded-pilot variance.

Frames, rerenders, checkpoints, and repeated forecasts are not independent training replications.

## Comparisons

**C versus A:** Does source-family continuation improve over the original base predictor?

**C versus B:** Does same-trajectory exposure create a source-specific forecasting advantage beyond matched continued training?

**C versus D:** Is the advantage larger than ordinary generalization from nearby independent events?

**E versus C:** How much of the advantage depends on exact pixels rather than the underlying physical trajectory?

**A versus B:** Does continued training itself alter independent-event performance?

**Source-family versus independent evaluation:** Does a render-disjoint evaluation overstate independent-event generalization?

## Status

The controlled simulator, state-preserving rerendering, lineage records, and bounded evaluation metrics have been implemented and unit-tested.

### Preliminary Assay Validation

A pre-specified positive control used a state-space nearest-trajectory memory predictor to validate sensitivity to source-specific trajectory reuse.

| Metric | Result |
|---|---:|
| Independent simulation blocks | 64 |
| Mean source-family gain | 0.033367 |
| Mean independent-population gain | 0.000117 |
| 50% evaluation-reuse effect | 0.016625 |
| Effect standard deviation | 0.002033 |
| Normal 95% lower bound | 0.016127 |
| Pre-specified threshold | 0.015625 |
| Gate | **Pass** |

As intended, the memory predictor was perfect on stored source trajectories and provided essentially no average advantage on independently generated events. The next experimental milestone applies the validated assay to the ConvLSTM predictor.

## Next Steps

1. Align on whether physical-event lineage is within the intended scope of the training-data study.
2. Agree on the minimum result that would materially change the study’s conclusion.
3. Implement the ConvLSTM predictor and verify that it beats constant velocity on held-out collisions.
4. Run the 8-block discarded runtime and variance pilot.
5. Validate tracking and failure penalties on generated predictions, not only on ground-truth renders.
6. Freeze the event population, experimental arms, primary endpoint, practical effect threshold, and confirmatory analysis.
7. Run confirmation only if the powered design fits the committed compute budget.

## Risks

**C ≈ B:** Same-trajectory exposure provides no measurable advantage. Report the negative result and stop this version of the study.

**C ≈ D:** The effect is explained by ordinary local generalization rather than reuse of the same physical event.

**Source gain is caused by independent-event degradation:** Report all component losses and do not interpret the result as improved source-family prediction.

**Measurement artifacts dominate:** If tracking failures, identity switches, or invalid-output penalties account for the effect, repair the measurement before interpreting model behavior.

**Overlap safeguards already work:** If competent temporal or motion matching detects all consequential source-family overlap, the proposed hidden-overlap failure is not present under the tested safeguard.

**Powered replication is unaffordable:** Set the confirmatory scope from the number of independent training blocks supported by the measured pilot variance and compute budget. Additional frames and rerenders reduce within-block measurement noise but do not increase training-level replication.
