# Evaluation protocol

## Unit of independence

One independent training block contains an independently generated corpus, model initialization, base training, and paired continuation branches. Branches sharing a base are paired conditions, not independent replications.

Frames, rerenders, windows, checkpoints, and repeated forecasts reduce measurement noise but do not increase the number of training-policy replications.

## Bounded state score

Primary measurement primitive:

- Track both disk centers through all forecast frames;
- Compute Euclidean error in arena-width units;
- Cap each disk-frame error at one;
- Assign one to every invalid disk-frame; and
- Average only after penalties are applied.

Report localization, identity-switch, missing-object, extra-object, and penalty contributions separately.

## Evaluation-reuse effect

For paired block-level gains on source-family rerenders `g_source` and independently sampled events `g_population`, a reuse fraction `π` changes the measured gain by:

```text
reuse_effect = π × mean(g_source - g_population)
```

This is an algebraic property of the declared mixture. Scientific value requires an empirical difference in the two gains and a reason the mixture represents an evaluation practice that matters.

## Required component reporting

Never report only the mixture effect. Always report:

- Source branch and background branch losses on source-family rerenders;
- Both losses on independent events;
- Invalid-output penalty contributions;
- Distance to every consumed trajectory;
- Model-learning positive controls; and
- Cost and variance across independent blocks.

## Controls

- Persistence;
- Constant velocity;
- Measured-state analytic physics;
- Privileged true-state physics;
- Nearest trajectory by state distance;
- Exact pixel matching;
- Perceptual frame matching; and
- Temporal/motion alignment.

## What the measurements cannot tell us

- That an undocumented model consumed a candidate video;
- That same-trajectory gains are a distinct memorization mechanism;
- That late continued training matches pretraining;
- That one small predictor represents all video world models;
- That excluding detected overlaps improves a retrained model; or
- That a statistically detectable effect has practical benchmark importance.
