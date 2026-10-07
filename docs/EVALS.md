# evaluation protocol

## unit of independence

one independent training block contains an independently generated corpus, model initialization, base training, and paired continuation branches. branches sharing a base are paired conditions, not independent replications.

frames, rerenders, windows, checkpoints, and repeated forecasts reduce measurement noise but do not increase the number of training-policy replications.

## bounded state score

primary measurement primitive:

- track both disk centers through all forecast frames;
- compute euclidean error in arena-width units;
- cap each disk-frame error at one;
- assign one to every invalid disk-frame; and
- average only after penalties are applied.

report localization, identity-switch, missing-object, extra-object, and penalty contributions separately.

## evaluation-reuse effect

for paired block-level gains on source-family rerenders `g_source` and independently sampled events `g_population`, a reuse fraction `π` changes the measured gain by:

```text
reuse_effect = π × mean(g_source - g_population)
```

this is an algebraic property of the declared mixture. scientific value requires an empirical difference in the two gains and a reason the mixture represents an evaluation practice that matters.

## required component reporting

never report only the mixture effect. always report:

- source branch and background branch losses on source-family rerenders;
- both losses on independent events;
- invalid-output penalty contributions;
- distance to every consumed trajectory;
- model-learning positive controls; and
- cost and variance across independent blocks.

## controls

- persistence;
- constant velocity;
- measured-state analytic physics;
- privileged true-state physics;
- nearest trajectory by state distance;
- exact pixel matching;
- perceptual frame matching; and
- temporal/motion alignment.

## what the measurements cannot tell us

- that an undocumented model consumed a candidate video;
- that same-trajectory gains are a distinct memorization mechanism;
- that late continued training matches pretraining;
- that one small predictor represents all video world models;
- that excluding detected overlaps improves a retrained model; or
- that a statistically detectable effect has practical benchmark importance.
