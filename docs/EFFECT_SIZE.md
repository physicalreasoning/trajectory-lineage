# Draft Effect-Size and Power Contract

## Status

This document is groundwork for the learned trajectory-reuse pilot requested in Jiankai’s feedback. The thresholds are provisional until reviewed by the paper lead and Yining.

## Primary Estimand

For continued-training arms B and C, define:

```text
source gain      = Error(B, source rerenders) − Error(C, source rerenders)
independent gain = Error(B, independent events) − Error(C, independent events)
reuse gap        = source gain − independent gain
```

The reuse gap is the primary prevalence-free quantity. Evaluation-reuse effects at a particular prevalence remain sensitivity analyses:

```text
reuse effect at prevalence π = π × reuse gap
```

## Practical Scale

The ConvLSTM feasibility run provides one scale reference:

```text
Constant-velocity mean error = 0.048823379
ConvLSTM mean error          = 0.029653877
Observed headroom            = 0.019169502
```

Define the normalized reuse gap:

```text
normalized reuse gap = reuse gap / observed headroom
```

The provisional minimum effect of interest is 25% of the observed headroom:

```text
minimum normalized gap = 0.25
minimum raw gap        = 0.004792375
```

This threshold asks whether trajectory reuse explains at least one quarter of the improvement separating the learned model from constant-velocity extrapolation in the feasibility task. It is a draft scientific decision threshold, not an estimate from the exposure experiment.

## Component Requirements

A confirmatory source-specific effect requires all of the following:

1. The reuse-gap confidence interval excludes zero in the pre-specified direction.
2. Source-family gain is positive.
3. Independent-population degradation remains within the pre-specified harm margin.
4. The same-trajectory arm exceeds the nearby-trajectory arm under the corresponding source-versus-independent contrast.
5. Every scheduled independent training block is included.

A provisional independent-population harm margin is 10% of the observed headroom:

```text
maximum tolerated harm = 0.001916950
```

The harm margin and 25% effect threshold require review before confirmatory training.

## Pilot and Confirmation

The first eight independent blocks are discarded feasibility blocks. They estimate:

- Block-level standard deviation
- Training runtime and failure rate
- Generated-output measurement error
- Source and independent gains
- Required confirmatory block count

The confirmatory calculation uses paired block-level contrasts. Frames, rerenders, checkpoints, and repeated forecasts reduce within-block noise but do not increase the training-level sample size.

Under a normal approximation, the block count for power `1 − β` and two-sided level `α` is:

```text
n ≈ ((z_(1−α/2) + z_(1−β)) × σ / δ)^2
```

where `σ` is the pilot block-level standard deviation and `δ` is the effect above the decision threshold that the study is designed to detect. Final planning should use a finite-sample calculation and joint power for all required component gates.

The current normal-approximation sensitivity table assumes the study is powered to detect an effect one half-threshold above the minimum meaningful gap (`0.002396` above `0.004792`) at 90% power and two-sided 5% significance:

| Pilot block SD | Approximate required blocks |
|---:|---:|
| 0.0025 | 12 |
| 0.0050 | 46 |
| 0.0075 | 103 |
| 0.0100 | 184 |

This table is a planning diagnostic. The eight-block discarded pilot determines whether a confirmatory study is computationally credible.

## Reporting

Report:

- All branch errors on every evaluation population
- Reuse gap and normalized reuse gap
- Prevalence sensitivity for several declared values of `π`
- Individual block effects
- Leave-one-block-out sensitivity
- Independent-event harm
- Nearby-arm comparison
- Measurement failures and penalties
- Compute cost and failed runs
