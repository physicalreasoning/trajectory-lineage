# research plan

all gates are written before their experiment runs. a failed gate stops or narrows the programme; it does not trigger an automatic scope expansion.

## question

can controlled reuse of a physical trajectory under new pixels change the measured advantage of a training policy relative to evaluation on independently sampled events?

## claim boundary

the experiment can identify a causal effect of its own continued-training replacement policy. it cannot identify undocumented historical training data or a special internal memorization mechanism.

## gate 0 · collaborator fit

**question:** does this estimand answer a live question in the training-data workstream?

**pass:** the collaboration accepts the population, treatment, score, and minimum useful result.

**stop:** “source” means historical website, owner, or proprietary-corpus identification and no controlled-exposure result is needed.

## gate 1 · simulator and lineage

**run:** generate analytic collision trajectories, rerender identical states, and build immutable manifests.

**pass:**

- physical invariants hold within frozen tolerance;
- rerenders have identical state hashes and different render hashes;
- no unauthorized trajectory family crosses a split; and
- every consumed example can be reconciled to an optimizer update.

**stop:** state changes during rendering, lineage is incomplete, or the event population cannot be specified reproducibly.

## gate 2 · measurement headroom

### registered memory positive control

before training a neural predictor, run `experiments/memory_positive_control.json`. a nearest-trajectory predictor serves as an intentionally memory-sensitive upper bound. sixty-four independently generated blocks compare a source-family memory bank with an equally sized independent background bank on source-family and independent-population events.

**registered pass:** mean 50% evaluation-reuse effect above `1/64`, normal-approximation 95% lower bound above zero, and zero source-bank loss for the source-memory policy. passing validates assay sensitivity only; it is not evidence about a neural world model.

**run:** compare persistence, constant velocity, analytic measured-state prediction, and the candidate predictor on discarded data. validate center tracking on both ground-truth and generated outputs.

**pass:** the predictor beats constant velocity on the easy collision task, the exposure positive control moves the bounded state score, and invalid-output penalties do not dominate the effect.

**stop:** two documented pilot revisions fail any condition.

## gate 3 · cost and variance

**run:** complete discarded independent training blocks and record wall time, memory, storage, failure rate, component losses, and block-level effects.

**pass:** a preregistered inference plan can estimate the chosen effect at meaningful precision within committed compute.

**stop:** confirmation requires an unaffordable number of independent training blocks. more frames or rerenders cannot substitute for blocks.

## gate 4 · confirmation

this gate is intentionally undefined until gates 0–3 provide collaborator acceptance, an executable population, and measured variance. any final threshold, mixture prevalence, or inference rule written before that evidence is a planning candidate rather than a preregistration.

## kill criterion

stop this exact specification if any of these hold:

1. the collaboration does not need the controlled-exposure estimand;
2. the event population cannot be recovered or frozen;
3. the model or measurement fails gate 2 after two repairs;
4. powered confirmation exceeds committed resources; or
5. close prior work already answers the same operational question.

## what gets released after a valid pilot

- population and lineage specification;
- hashes and exposure logs;
- frozen experiment configurations;
- all scheduled block results, including failures;
- measurement-validation report;
- cost model;
- analysis code; and
- an updated findings record.
