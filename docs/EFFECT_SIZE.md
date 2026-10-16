# Effect-Size and Decision Contract

## Status

Draft for sign-off by the paper lead and Yining Hua. The executable form is `experiments/exposure_confirmation.json` (`status: draft_pending_signoff`) and the frozen analysis is `scripts/analyze_confirmation.py`. No learned-model reuse effect estimate has been inspected or released. Raw block records (including per-render losses for later detector filtering) are written only when a freeze manifest binds the configuration SHA-256, a clean code commit, the analyzer SHA-256, named sign-offs for the `paper_lead` and `evaluation_science` roles, and the excluded pilot seeds; `scripts/exposure_experiment.py` refuses otherwise.

This responds to Jiankai Sun’s request to “pre-specify a practically meaningful effect size” for the reuse pilot.

## Estimands

Per independent training block, with `L(a, P)` the mean bounded center error of continuation arm `a` on population `P`:

```text
S  = L(B, source) − L(C, source)            source gain
I  = L(B, indep)  − L(C, indep)             independent gain
H  = −I                                     independent harm
G  = S − I = S + H                          reuse gap (primary)
Q  = [L(D, source) − L(C, source)] − [L(D, indep) − L(C, indep)]   same trajectory vs closer independent neighbor
Q2 = the same contrast against the band neighbor arm D2
A_C = L(C, indep) − L(C, source)            fixed-model apparent improvement of C on reused items
```

Interpretation for an additive per-item metric and an evaluation set whose reused fraction is `π`:

- `π·G` is the change in C’s measured advantage over B, a relative quantity. It is not the absolute score inflation of one model.
- `π·A_C` is the apparent improvement of model C caused by replacing independent items with reused ones.
- If `H > 0` and `G > 0`, the B–C ordering crosses at `π = H/G`.
- The break-even prevalence for a ranking margin `m` is `π* = m/G`, reported with the interval `[m/U_G, m/L_G]`; it is unbounded above when `L_G ≤ 0`.
- A fixed judge's mean per-episode success is additive; absolute calibration error, AUROC, rank correlation, and rankings are not and are recomputed, never extrapolated.

## Thresholds (operational convention)

The thresholds are a declared convention, not an estimate of anyone’s utility function.

| Quantity | Value | Rationale |
|---|---:|---|
| Ranking margin `m` | 0.001 arena widths | About 3.4% of the ConvLSTM’s clean error, the size of relative improvement a paper would report; about 1.6× the observed ConvLSTM–retrieval mean difference (whose own 95% interval spans zero, and whose retrieval baseline receives privileged state input) |
| Consequential threshold | `m / 0.10 = 0.010` | Consequential if reuse at a modest 10% prevalence would move a measured advantage by one margin |
| Negligible bound | `m / 0.50 = 0.002` | Negligible only if reuse at a high 50% prevalence would still move it by less than one margin |
| Harm margin `h` | 0.001 | Independent-event degradation applies to every independent item, so the same margin applies at full prevalence |

Prevalence values are scenarios, not estimates of physical-trajectory reuse in any real benchmark. The asymmetric references make each decisive claim conservative: a consequential claim must hold at low prevalence, and a negligible claim must hold at high prevalence. The earlier 25%-of-headroom value (0.004792) is retained only as history and confers no claim. Sensitivity settings never become alternative success criteria.

## Decision Rule

Block-level paired contrasts, two-sided 95% t intervals (one-sided α = 0.025 per bound). Primary analysis uses blocks passing a pre-treatment base-competence screen on a separate 256-event population that is never scored (the base model must beat constant velocity there); the intent-to-treat analysis over all completed blocks is a sensitivity analysis. Infrastructure errors retry once with identical seeds; other exceptions are algorithmic failures; if failures exceed 5% of scheduled blocks, or fewer than 240 blocks pass the screen, the result is inconclusive.

Outcomes are reported on separate axes, so a failed safeguard is never relabeled as a prevalence effect.

| Axis | Categories |
|---|---|
| Gap magnitude | consequential `lower(G) > 0.010`; prevalence-dependent `0.002 < lower(G) ≤ 0.010`; negligible `−0.002 < lower(G)`, `upper(G) < 0.002`; reverse `upper(G) < −0.002`; otherwise inconclusive |
| Safeguards | source gain supported `lower(S) > 0`; harm bounded `upper(H) < 0.001`; assay sensitive (E exact gain positive in at least 75% of screened blocks) |

| Primary claim | Condition |
|---|---|
| Consequential reuse advantage | Consequential magnitude and both safeguards (intersection–union test) |
| Prevalence-dependent reuse advantage | Prevalence-dependent magnitude and both safeguards; break-even prevalence interval reported |
| Differential gap without safeguards | Either positive magnitude with a failed safeguard |
| Negligible differential gap | Negligible magnitude and assay sensitive |
| Assay insensitive | Negligible magnitude but the positive control failed |
| Reverse gap | Reverse magnitude |

Secondary hypotheses, Holm step-down at one-sided 0.025: identity-specific reuse (H0: Q ≤ 0.010, against the closer-than-base-training arm), detectable reuse (H0: G ≤ 0), consequential harm (H0: H ≤ 0.001). A claim that same-trajectory exposure helps *beyond closer independent neighbors* requires the identity-specific rejection. Q2, the distance profile, and detectors are exploratory.

Break-even prevalence `π* = m/G`: finite interval `[m/upper(G), m/lower(G)]` when `lower(G) > 0`; unbounded above when `lower(G) ≤ 0 < upper(G)`; no positive crossing when `upper(G) ≤ 0`; also reported intersected with `[0, 1]`. The harm-to-gap crossover `H/G` gets a block-bootstrap interval with the fraction of resamples where it is undefined.

What each outcome licenses is narrow: it concerns this model, exposure dose (12–13 presentations per source), continuation phase, population, and metric. A negligible gap does not establish that training is harmless, that both arms are equivalent on every population, or that appearance-level curation is adequate elsewhere.

## Power and Sample Size

`scripts/plan_confirmation.py` pools the effect-blinded dispersion of pilots v4 and v5 (16 blocks, 14 degrees of freedom; the C/B contrasts S, H, and G do not depend on the changes between those pilots). Because `G = S + H`, the blinded SDs of `S`, `H`, and `G` determine their covariance (pooled corr(S, H) = −0.95). Joint power of the consequential rule is simulated under coherent alternatives (`G = S + H = 0.015`), with exact Student-t critical values and SDs inflated to the pooled 80% upper confidence limit, then by 1.5× and 2×.

| SD multiplier | H = 0 | H = 0.0005 (half the margin) |
|---:|---:|---:|
| 1.0 | 1.00 | 0.99 |
| 1.5 | 1.00 | 0.83 |
| 2.0 | 0.99 | 0.59 |

Using pilot v5 alone (7 df), the H = 0.0005 column is 0.95, 0.67, 0.44. The harm gate is binding: continuation on any 64 examples shifts the whole model by a block-specific amount (per-block SD about 0.0017 on independent events), which the reuse gap cancels but the harm contrast does not. Rejection rate exactly at the consequential threshold is 0.024–0.026 (nominal 0.025); negligible-outcome power at `G = 0` is 1.00; identity-specific power exceeds 0.9 when true Q ≥ 0.0102. These are conditional on normal block contrasts and the stated SDs; they are not assurance.

352 blocks are scheduled and planned as 336 analyzed (about 4% screen attrition; at least 320 must pass), about 4.9 hours on Apple MPS.

## Pilot History

| Pilot | Change | Status |
|---|---|---|
| v1 | Source-perturbed neighbors | Superseded |
| v2 | Reservoir neighbors | Superseded: overlapping random streams across blocks, uncapped metric |
| v3 | Corrected runner, final evaluation allocation | Superseded: band neighbors were farther than each source’s nearest base-training event for 86% of sources |
| v4 | Adds closer-than-base-training neighbor arm D; band arm becomes D2 | Superseded: half-range box cap excluded valid band members; screen reused the scored population |
| v5 | Full-range box cap; separate screen population; failure classes | Final protocol pilot |

All pilots are discarded and report only dispersion, runtime, balance, design descriptors, and a binary exact-render positive control.
