# Effect-Size and Decision Contract

## Status

Draft for sign-off by the paper lead and Yining Hua. The executable form is `experiments/exposure_confirmation.json` (`status: draft_pending_signoff`). No learned-model reuse effect estimate has been inspected or released. Raw block records can be written only with a freeze manifest whose configuration hash matches and which records sign-off (`scripts/exposure_experiment.py --unblind ... --freeze-manifest ...`).

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
- Non-additive metrics (AUROC, rank correlation, judged success rates) are recomputed, never extrapolated.

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

Block-level paired contrasts, two-sided 95% t intervals (one-sided α = 0.025 per bound). Primary analysis uses blocks passing the pre-treatment base-competence screen (base model beats constant velocity on the block’s independent population); the intent-to-treat analysis over all completed blocks is a sensitivity analysis.

| Outcome | Rule |
|---|---|
| Consequential | `lower(G) > 0.010` and `lower(S) > 0` and `upper(H) < 0.001` (intersection–union test; no further correction for this single conjunction) |
| Negligible | `−0.002 < lower(G)` and `upper(G) < 0.002` |
| Prevalence-dependent | `lower(G) > 0.002` but not consequential; report the break-even prevalence interval |
| Inconclusive | Otherwise; report which bound or gate failed |

Secondary claims (Holm-corrected): identity-specific reuse (`lower(Q) > 0.010`), detectable reuse (`lower(G) > 0`), and consequential harm (`lower(H) > 0.001`, reported regardless of outcome). Distance bands, detectors, and models are exploratory.

What each outcome licenses is narrow: it concerns this model, exposure dose (12–13 presentations per source), continuation phase, population, and metric. A negligible gap does not establish that training is harmless, that both arms are equivalent on every population, or that appearance-level curation is adequate elsewhere.

## Power and Sample Size

`scripts/plan_confirmation.py` uses the effect-blinded pilot v4 dispersion. Because `G = S + H`, the blinded SDs of `S`, `H`, and `G` determine their covariance (pilot v4: corr(S, H) = −0.96). Joint power is simulated under coherent alternatives (`G = S + H`), with exact Student-t critical values, inflating SDs to the 80% upper confidence limits and then by 1.5× and 2×.

| SD multiplier | True harm H | Joint power, N = 256 |
|---:|---:|---:|
| 1.0 | 0 | 1.00 |
| 1.0 | 0.0005 | 0.97 |
| 1.5 | 0 | 1.00 |
| 1.5 | 0.0005 | 0.73 |
| 2.0 | 0 | 0.97 |
| 2.0 | 0.0005 | 0.48 |

The harm gate is binding: the continuation on any 64 examples shifts the whole model by a block-specific amount (per-block SD about 0.0015 on independent events), which the reuse gap cancels but the harm contrast does not. Rejection rate exactly at the consequential threshold is 0.023–0.026 (nominal 0.025). Negligible-outcome power at `G = 0` is 1.00. These are conditional on normal block contrasts and the stated SDs; they are not assurance.

N = 256 blocks takes about 3.5 hours on Apple MPS.

## Pilot History

| Pilot | Change | Status |
|---|---|---|
| v1 | Source-perturbed neighbors | Superseded |
| v2 | Reservoir neighbors | Superseded: overlapping random streams across blocks, uncapped metric |
| v3 | Corrected runner, final evaluation allocation | Superseded: band neighbors were farther than each source’s nearest base-training event for 86% of sources |
| v4 | Adds closer-than-base-training neighbor arm D; band arm becomes D2 | Basis for planning |

All pilots are discarded and report only dispersion, runtime, balance, design descriptors, and a binary exact-render positive control.
