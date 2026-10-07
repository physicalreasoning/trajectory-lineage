# Research plan

All gates are written before their experiment runs. A failed gate stops or narrows the programme; it does not trigger an automatic scope expansion.

## Question

Can controlled reuse of a physical trajectory under new pixels change the measured advantage of a training policy relative to evaluation on independently sampled events?

## Claim boundary

The experiment can identify a causal effect of its own continued-training replacement policy. It cannot identify undocumented historical training data or a special internal memorization mechanism.

## Gate 0 · Collaborator fit

**Question:** Does this estimand answer a live question in the training-data workstream?

**Pass:** The collaboration accepts the population, treatment, score, and minimum useful result.

**Stop:** “Source” means historical website, owner, or proprietary-corpus identification and no controlled-exposure result is needed.

## Gate 1 · Simulator and lineage

**Run:** Generate analytic collision trajectories, rerender identical states, and build immutable manifests.

**Pass:**

- Physical invariants hold within frozen tolerance;
- Rerenders have identical state hashes and different render hashes;
- No unauthorized trajectory family crosses a split; and
- Every consumed example can be reconciled to an optimizer update.

**Stop:** State changes during rendering, lineage is incomplete, or the event population cannot be specified reproducibly.

## Gate 2 · Measurement headroom

### Registered memory positive control

Before training a neural predictor, run `experiments/memory_positive_control.json`. A nearest-trajectory predictor serves as an intentionally memory-sensitive upper bound. Sixty-four independently generated blocks compare a source-family memory bank with an equally sized independent background bank on source-family and independent-population events.

**Registered pass:** Mean 50% evaluation-reuse effect above `1/64`, normal-approximation 95% lower bound above zero, and zero source-bank loss for the source-memory policy. Passing validates assay sensitivity only; it is not evidence about a neural world model.

**Run:** Compare persistence, constant velocity, analytic measured-state prediction, and the candidate predictor on discarded data. Validate center tracking on both ground-truth and generated outputs.

**Pass:** The predictor beats constant velocity on the easy collision task, the exposure positive control moves the bounded state score, and invalid-output penalties do not dominate the effect.

**Stop:** Two documented pilot revisions fail any condition.

## Gate 3 · Cost and variance

**Run:** Complete discarded independent training blocks and record wall time, memory, storage, failure rate, component losses, and block-level effects.

**Pass:** A preregistered inference plan can estimate the chosen effect at meaningful precision within committed compute.

**Stop:** Confirmation requires an unaffordable number of independent training blocks. More frames or rerenders cannot substitute for blocks.

## Gate 4 · Confirmation

This gate is intentionally undefined until gates 0–3 provide collaborator acceptance, an executable population, and measured variance. Any final threshold, mixture prevalence, or inference rule written before that evidence is a planning candidate rather than a preregistration.

## Kill criterion

Stop this exact specification if any of these hold:

1. The collaboration does not need the controlled-exposure estimand;
2. The event population cannot be recovered or frozen;
3. The model or measurement fails gate 2 after two repairs;
4. Powered confirmation exceeds committed resources; or
5. Close prior work already answers the same operational question.

## What gets released after a valid pilot

- Population and lineage specification;
- Hashes and exposure logs;
- Frozen experiment configurations;
- All scheduled block results, including failures;
- Measurement-validation report;
- Cost model;
- Analysis code; and
- An updated findings record.
