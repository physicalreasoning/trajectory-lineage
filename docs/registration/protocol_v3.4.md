# Seen and Unseen: Registration Protocol

**Version:** registration draft v3.4, revised November 14, 2026 (supersedes the October 23 plan, the October 28 draft, v3 of October 29, v3.1 of October 30, v3.2 of October 31 and v3.3 of November 1; changes listed in §0.3). **Registration:** one OSF Preregistration for all three experiments, embargoed until peer review ends, submitted before any confirmatory run (the sealed 1D confirmation of §3.8 is the one disclosed exception).

**Relation to the registration.** The OSF record is the registration form and the registration core (`docs/registration/registration_core_v3.4.md`, uploaded to OSF as `preregistration_v3.4.pdf`), which states every confirmatory claim, estimator, threshold, exclusion and fallback. This protocol is stored in the repository at the registered commit (`docs/registration/protocol_v3.4.md`). It gives operational detail, the evidence for each choice and the interpretation tables, and adds no confirmatory commitment. If the two documents disagree, the core applies and the difference is logged as a deviation (§6).

**Authors:** contributors are listed on the OSF registration; paper authorship, order and CRediT roles are settled at paper submission (§9).

## 0. Overview

**Goal.** Test whether world-model policy evaluation misorders policies far apart in true success little more often than the same judge on real rollouts while getting their success rates wrong, and whether two hidden factors cause the error:

1. the share of failures in the world model's training data;
2. the reuse of training trajectories in evaluation.

**Claims under test.** First, estimated success from world-model policy evaluation depends on the share of failures in the world model's training data, while policies far apart in true success are misordered little more often in imagined rollouts than by the same judge on real rollouts. Second, in controlled 2D forecasting, reusing training trajectories in evaluation lowers forecast error on those trajectories relative to independent events. Experiment 1 (primary) tests the failure-share hypothesis in closed-loop policy evaluation. Experiment 2 checks the open-loop evidence on logged robot data. Experiment 3 tests the reuse hypothesis in controlled 2D forecasting, including with position predictors on frozen pretrained encoders; its effect on policy evaluation is registered for a follow-up.

**Experiments.**

| # | Experiment | Role | Owner | Status at registration |
|---|---|---|---|---|
| 1 | Closed-loop ManiSkill3 PickCube: failure share vs level and ranking errors, with judge-bias and transition controls; capacity secondary | Primary | Closed-loop workstream | Rules locked in §1; pilot in November; operational values and analysis code in the November 20 amendment |
| 2 | Transition-matched Kuka control | Robustness of the open-loop result | Experiment 2 workstream | §2 implemented as code; real-data certificate passed (October 28; recertified October 30); certified on the registered data (the copy rebuilt from public shards); draft freeze manifest ready to sign |
| 3 | Trajectory reuse and duplicate-detector audit, 2D collisions, with predictor capacity, architecture (frozen DINOv2 / V-JEPA 2 encoder + trained position predictor) and exposure dose as registered axes | Evaluation lineage | Experiment 3 workstream | §3 fully specified; simulator, analyzers, effect-blind pilots for design, capacity, architecture and dose, v3 dry runs complete; the analyzers that implement §3.5 are at the registered commit (code commit `f361b7e`) |

**0.1 What goes where.** On OSF: the registration form and the registration core (PDF), which carry the existing-data disclosures and the Experiment 3 attestation. In git: this protocol and a copy of the core under `docs/registration/` at the registered commit, the code at registration (the commits named in §7) and code freezes as dated tags before each experiment's confirmatory runs (§5, §7). With the paper: asset ledger, data dictionary, CRediT, interests.

**0.2 Registration and amendment schedule.**

| Date | Event |
|---|---|
| Registration (before any confirmatory run) | All three experiments together, embargoed |
| After registration | Experiment 2 training (110 runs; 2–3 h on one GPU); Experiment 3 confirmation (1,600 + 4 × 200 blocks; about 27 h wall-clock on one A100 with three concurrent blocks, checkpointed) |
| Before the Experiment 2 runs and the Experiment 3 confirmation | Signed freeze manifests that pin the analyzers at the registered commit (they implement the Experiment 2 Holm rule, core §C, and §3.5); no code amendment is needed |
| Nov 20 | Amendment: Experiment 1 operational values and analysis code; Experiment 3 detector-audit contract |
| Dec 8 | Analysis amendment and unblinding |

The paper venue is not chosen. The submission will be anonymized and will link neither this registration nor any repository.

**0.3 Changes from the plan sent to external reviewers on October 23** (this protocol is the version of record):

| Item | Plan (Oct 23) or earlier draft | Protocol | Why |
|---|---|---|---|
| Exp 1 policy selection | On true success; J1 bias reported | Same; an earlier protocol draft's J1-bias selection filter is dropped; P1–P3 are repeated on policies with J1 bias within 5 points as a sensitivity check | Selecting on the judge would bias the judge-bias estimate |
| Exp 1 P2 / S6 | Cross-fitted absolute shifts | Same for support (split-seed cross-fitting); absence statistic revised in v3.2 (row below) | Absolute values averaged before cross-fitting are biased upward |
| Exp 1 S3 | Percentile anchors; one logit offset; ratio upper bound below 0.5 | S3a = the plan's offset fit; S3b = an intercept-and-slope logistic recalibration, registered as a second secondary; calibration slope and intercept reported | Evaluation-science review: if weak policies are overestimated most, one offset cannot correct them |
| Exp 1 J1 threshold | Fixed 0.5; ECE below 0.05, Platt fallback | Same | A tuned threshold leaks the calibration set into the estimate |
| Exp 1 inference | Paired t over seeds; bootstrap as sensitivity | Same, locked, except P2 and S6: a bootstrap of the whole cross-fitting procedure (seeds resampled within the fixed halves) | Evaluation-science review: cross-fitting makes the two seed halves dependent, so a t interval is not valid for them |
| Exp 1 capacity sweep | Descriptive, 3 seeds | Secondary S7, 5 seeds, plus a descriptive encoder arm if budget allows | A GPU budget now exists; the "model too weak" risk deserves a tested answer |
| Exp 1 outcome table (§1.7) | "Use it to rank, not to report levels" | Scoped to these 12 policies and to pairs at least 15 points apart; close pairs follow S4; a row for P1 with J1 but not J2 | Evaluation-science review: P3 covers only far pairs; a J1-only shift is not just the world model's; bounds describe these policies |
| Exp 1 limitation | n/a | Stated: imagined episodes have no ground truth, so world-model error and judge error are not fully separable | Evaluation-science review |
| Exp 1 secondary family | S1–S6 | S1, S2, S3a, S3b, S4–S7 (eight tests, Holm) | S3b and S7 added |
| Exp 3 harm limit | Harm upper bound below 0.001 | Same; reachability checked from the blinded pilot dispersions (§3.6) | Evaluation-science review |
| Exp 2 V arms | Add transitions until the arm totals 31,600 | Same (an earlier draft added a fixed 15,800) | Transition-matched to design A |
| Exp 2 O, B, V | Descriptive | Secondaries with Holm over ten p-values | Stricter; the primary is unchanged |
| Exp 2 capacity | n/a | v3: descriptive width-512 arm. v3.2: dropped | It needed a change to the original pipeline's trainer; Experiment 1's S7 covers capacity |
| Exp 3 positive control | Arm E gap lower bound above 0.002 (required for "negligible") | Two tiers: *detected* (lower bound of E above 0) and *at the margin* (above 0.002). Only the second permits a "negligible" reading; a negligible G with E detected below the margin is reported as inconclusive about negligibility (§3.5) | v3 had dropped the 0.002 requirement; v3.2 restores it for the negligible reading. A control that learns less than 0.002 cannot show the assay would see a 0.002 gap (the frozen analyzer's dry run called G "negligible" in 96% of datasets with E = 0.001). Stricter, decided before any 2D effect was seen |
| Exp 3 prevalence-dependent | Lower bound in (0.002, 0.010] | Whole interval in (0.002, 0.010] | An interval reaching past 0.010 must not be called prevalence-dependent |
| Exp 3 size | 200 seeds × 5 arms, assuming a gap SD of 0.022 | 800 blocks at the primary capacity (pilot gap SD 0.0055), plus 400 + 400 at 64 and 256 channels | Powered on measured dispersions; capacity added as a factor |
| Exp 3 sampler | Sequential | Per-source parallel with spawned streams; same distribution, worker-count invariant | Runtime |
| Exp 3 architecture axis | n/a | Two added configurations (200 blocks each): latent world-model predictor on a frozen DINOv2-small or V-JEPA 2-L encoder; secondary "reuse detectable under recipe" in the joint Holm family | Moves the claim from a from-scratch CNN to the recipe real world models use |
| Exp 3 dose axis | Repetition sweep listed as a follow-up | Two added configurations (200 blocks each): 16 and 256 reused sources (~50 and ~3 presentations); secondary "dose increase" | Mechanism and a practical dedup threshold |
| Exp 3 secondary family | Three tests | Seven tests in one Holm family across the main confirmation and the axes | Familywise control over everything claimed |
| Sign-offs | Sign-off on Exp 3 thresholds by the two outside reviewers | Internal signers (paper lead + statistics reviewer); comments from every author welcome | No single reviewer is on the critical path |
| Exp 1 seeds | 10 per arm (halves of 5) | 12 per arm (halves of 6) | Cross-fitting with 5 seeds per half allows only 126 distinct bootstrap resamples per half (462 with 6); P1 power rises from about 0.90 to 0.95 at a 6-point effect (SD 3); P2 needs about 5.9–7.7 points for 90% power instead of 6–8 |
| Exp 1 S3a / S3b anchors | 3 (25/50/75th percentiles) | 4 (20/40/60/80th), 100 labels each | A slope fitted on three points is barely identified |
| Exp 1 headline | P1 ∧ P2 ∧ P3 | P1 ∧ P3 (intersection–union); P2 a registered primary test decided at its own 5% and reported next to the headline | §1.5 expects P2 may be inconclusive (open-loop shift 4.8 against a 5-point bar); the core claim should not fail on the size test. Decided before registration |
| Exp 1 S3 outcome row | "S3a and S3b refuted: world-model training data needs failures" | "Recalibration on these anchors is not enough for these policies" | Recalibration failing does not show that training data needs failures; that would be an untested causal claim |
| Exp 1 margins | Stated | Justified: 3 points matches Experiment 2; 5 points is the smallest close-pair gap, so a larger error can reverse a close comparison | Every margin has a stated practical meaning, as in Experiment 3 |
| Exp 1 descriptive labels | "High" selection regret, "steady" fall undefined | High regret: one-sided 95% lower bound above 5 points; steady: S2 passes and the φ=.25 shift lies between .10 and .50 | A label needs a bar fixed in advance |
| Exp 1 gates | Failure reported | A failed competence gate or a J2 that passes only at 5 cm limits what the run can claim (§1.7); the pilot reports J2's gate estimates with intervals | Every safeguard has a stated consequence |
| Exp 1 further analyses | n/a | All-pairs excess misordering; judge-recalibration sensitivity; policy-resampling generalization check; exploratory open-loop bridge to Experiment 2; exploratory released-world-model check | Far pairs rarely flip; bounds one route of judge error; shows how far the result stretches; links simulation to real data; answers "does this happen in real models?" |
| Exp 1 limits | Judge/world-model limitation | Plus: "failure share" means the resampling intervention (it changes coverage); one fixed ~100-step horizon | Stops readings beyond the intervention |
| Support levels | Implied | Exps 1–2 one-sided 5%; Exp 3 one-sided 2.5% because one interval drives several magnitude declarations | Stated so the difference is not read as arbitrary |
| Reporting | Implied | Confirmatory families counted (§4); analyst separation, claims ledger and reporting commitment (§5); sealed 1D confirmation reported after the 2D unsealing | Prevents selective reporting and post-hoc framing |
| Reproducibility | Code tags | Freeze manifests also carry a lockfile hash and runtime hash; hardware class fixed per experiment (§7) | GPU training is not bit-identical across hardware |
| Exp 3 detector disclosure (v3.2) | n/a | States that H3.3–H3.6 were informed by the public 1D detector benchmarks and the 1D interim filtering analysis (§3.8) | They are directional replications in 2D, not first tests |
| Exp 2 certificate (v3.2) | n/a | Updated to the October 30 recertification (V arms under the matched-total rule; manifest hash `1dcb390f…`); π corrected to the certified 0.355797; support count corrected to 15 lengths | v3.1 cited the superseded October 28 hash, a π from the synthetic manifest, and 17 support lengths (17 is Design B's length count) |
| Exp 2 template, dates | "Secondary Data Preregistration or general form"; Oct 23 header; "after October 30 registration" | One OSF Preregistration; dates removed; runs launch only after registration | Contradicted §0.1; registration date not fixed |
| Compute (§8) | 810–1,210 GPU-hours | Per-experiment figures matching §1.6, §2 and §3 | The old total contradicted §1.6 |
| Authors | Internal list; external collaborators added when confirmed in writing | Authors no longer listed in the documents; OSF lists everyone who agreed by submission day (Jiankai Sun and Yining Hua accepted); paper authorship settled at submission | Listing on OSF should not block registration |
| Registration scope | Everything on OSF | Form + registration core (with disclosures) on OSF; this protocol in the repository at the registered commit; code at registration pinned by commit; freezes as git tags; ledger, dictionary, CRediT, interests with the paper | A concise registration of claims, estimators, thresholds, exclusions and fallbacks that points to a repository commit and does not repeat implementation detail |
| Registered document (v3.2) | n/a | New registration core (the registration core, about 10 pages) holds every confirmatory commitment; this protocol moves to the repository, cited by commit | The protocol runs to about 50 pages (18,000 words); the commitments now read in one short document |
| Exp 1 policy overlap (v3.2) | Stated in the plan | Restored: the distilled-PPO family imitates the teacher that generated half the world-model training data; judge and world-model training data kept separate (§1.2, limits) | Dropped between the plan and v3 |
| Exp 1 bootstrap percentiles (v3.2) | n/a | Percentiles calibrated by the pilot's synthetic simulation over registered scenarios (§1.4), recorded November 20 | With six seeds per half the plain 5th percentile covers only about 0.91–0.92 when all shifts are equal; the rule is fixed before the pilot |
| Exp 1 failed runs (v3.2) | Rerun once with a reserve seed | A failed seed index reruns in all five shares under a listed reserve index (13–15, 16–18), keeping its half; complete-case sensitivity; takes precedence over §6 | A reserve seed in one arm would break pairing, nesting and cross-fitting; §1.4 and §6 also contradicted each other |
| Exp 1 P2 / S6 absence (v3.2) | Upper bound below 2 (plain average of absolute shifts) | Upper bound below 2 for the mean over policies of each policy's absolute 12-seed mean shift | The per-seed average is reachable only when per-policy noise is small (absence probability about 0.16 at SD 2.5 with every true shift zero); averaging seeds first keeps it biased only upward |
| Exp 1 S4 absence (v3.2) | n/a | Absent when the 90% upper bound is below 5 (unadjusted) | §1.7 had an "S4 absent" row with no rule |
| Exp 1 J1-only row (v3.2) | n/a | Split into S1 absent (judges disagree; not attributed to the world model) and S1 inconclusive (attribution untested) | An inconclusive S1 cannot show J1 is the cause |
| Exp 1 J2 in imagination (v3.2) | J2 uses the world model's predicted velocities | Restored (§1.2) | Dropped between the plan and v3 |
| Exp 1 amendment-hash check (v3.2) | Analysis code checks the amendment's hash before reading outcomes | Restored (§1, §5) | Dropped between the plan and v3 |
| Exp 1 policy-sample sensitivity (v3.2) | Bootstrap over seeds and policies for P1 to P3 | Restored to P1 to P3 | v3 had narrowed it to P1 and P3 |
| Exp 1 budget drop order | Stretch arm, then capacity sweep, then stochastic model to 3 seeds | Stretch arm, then the descriptive V-JEPA 2 encoder arm, then S7 and S5 to 3 seeds (descriptive) | The capacity sweep became secondary S7; the encoder arm is separate from S7 |
| Exp 3 detector predictions | Predictions outside the pass tables (descriptive) | H3.3–H3.5 one Holm family at familywise 0.05, confirmatory only if their details are recorded November 20; H3.4 names object-track matching (above 80%, and above OFS-3) | Exp 3 workstream decision in v3; the 2D audit uses new data, but the predictions were informed by 1D detector results (§3.8) |
| Exp 3 event-specific test | Gated: runs only after Consequential holds | One of the seven Holm-adjusted secondaries | Holm controls the familywise rate without the gate, and Q stays tested when G is not consequential |
| Exp 3 filtering (H3.6) | Descriptive, no pass or fail; keeps at least 90% of independent items; gap against coverage | v3 and v3.1: confirmatory if frozen before unblinding, floor dropped. v3.2: descriptive again, with the floor and coverage curve | The confirmatory version had no unambiguous decision rule |
| Exp 3 follow-up | Reuse inside Exp 1 registered for a follow-up, with thresholds | v3: "proposed follow-up". v3.2: plan thresholds restored (§3.9) | Dropped between the plan and v3 |
| Exp 3 memory check | Lookup predictor's reuse gap lower bound above 0.010 | Mean above 0.0156 and lower bound above 0 (the 1D gate) | Exp 3 workstream decision in v3; the 2D result (0.055, lower bound 0.054) passes both |
| Exp 3 competence screen | A failing base model is replaced before any arm trains | Excluded before arms are assigned, with minimum screened counts and an all-block sensitivity | A fixed, frozen block schedule; exclusion is pre-treatment |
| Exp 1 BC-check fallback | Competence check fails: 16-step loss, then 8×8 grid; BC check fails: 8×8 grid | Same (restored in v3.2; v3 listed one order for both) | Dropped between the plan and v3 |
| Exp 1 pair classes (v3.2) | Pair classes use the 10,000-episode truth | Restored (§1.3, core D) | Dropped between the plan and v3 |
| Exp 3 error measure | Distance of predicted from true positions | Capped at 1 arena width per disk-frame; non-finite outputs score 1 and stay in | A bounded measure keeps one diverging block from dominating the mean |
| Exp 3 claim tiers | Consequential, prevalence-dependent, negligible, reverse | Adds "detectable, magnitude unresolved" and "differential gap without safeguards" | Every combination of magnitude and safeguards gets a fixed reading |
| Exp 1 encoder arm (v3.2) | none (earlier draft: DINOv2-base tokens at width 256) | Frozen V-JEPA 2-L encoder with a readout to the DINOv2-small tokens policies and judges read; descriptive; only if budget allows | Tests the effect under a video encoder pretrained by latent prediction instead of a larger copy of the same encoder; DINOv2-small stays the main encoder because Experiment 2 and the open-loop bridge use it |
| Exp 3 encoder comparison (v3.2) | n/a | Exploratory: G(V-JEPA 2) − G(DINOv2), Welch interval, reported as an encoder difference (§3.5) | Registered before any 2D outcome so a difference is not a post hoc finding; labelled exploratory because the encoders differ in more than pretraining |
| Exp 3 positive-control estimand | "Arm E's gap" | E = L(B,x) − L(E,x), the gain on the exact renders | States which contrast the control uses |
| Exp 3 failures and axes (v3.2) | n/a | A main-confirmation failure voids the primary claim and the main-block secondaries; an axis failure enters with p = 1 and leaves only that axis unresolved; the detector audit is unaffected | §3.4 and §3.5 contradicted each other |
| Exp 3 "independent" and "negligible under recipe" labels (v3.2) | n/a | Also need lower95(E) > 0.002 in each compared configuration | Same logic as the main negligible reading |
| Exp 2 judge gate and B fallback | n/a | Judge AUROC below 0.75 stops the runs (no confirmatory result); B budget fallback with A comparators (130 runs) | Implemented with the registered code (§2) |
| v3.3 summary and rules (v3.3) | n/a | One-page summary and margins table; fixed 8-member Exp 1 Holm family (p = 1 for untested); S6 Holm p-value via calibrated percentiles; zero-variance bounds; closed list of replaceable training failures; detector test statistics registered | Reviews of v3.2: undefined or contradictory rules |
| v3.4 corrections (v3.4) | n/a | Page-1 claims match the tests (P1 lowers estimates; P3's 5-point margin; Exp 3 claims stated as the reuse gap); Exp 2 labelled a robustness check of a known drop; S6 Holm levels made reachable and their range disclosed (§1.5); "not robust to replacement" only qualifies a supported claim; missing-run rules made consistent (an Exp 1 run that stays unresolved, including a failed retry or a training failure found after scoring, makes its decisions inconclusive); the statistics reviewer is an author who runs no confirmatory jobs (was: did not run that experiment's jobs); H3.5 operating point and bootstrap p-value defined; attestation names the Experiment 3 lead and states that the 1D interim estimates are public; positive-control margin provenance stated | Reviews of v3.3 |
| v3.4 review round (v3.4) | n/a | Every Holm family keeps its registered size (an untested or unresolved member enters with p = 1; declared at an adjusted p of at most the familywise level), and the Exp 2 analyzer is patched to match; Exp 1 failures classified by cause (an infrastructure failure at any stage is retried); judge-validation failures have stated consequences, and each competence gate must pass at φ = .10 and .65 separately; S3 fitting method and logit clipping fixed; P1 attributed to the world model only with S1; a descriptive replay check of imagined action sequences in the simulator; encoder-arm readout fitted once, with a readout-matched reference; core deviation rule; amendment dates are targets and parts may be filed per experiment; Exp 2 operator blind to per-run estimates; detector bootstrap re-estimates thresholds per draw, and that family is at one-sided 0.05; Exp 3 secondaries name their blocks; Exp 3 latent predictor described as a position regressor, and Exp 3 limits stated in the core | Four further reviews of v3.4 (world models, cognitive-science methods, communication, evaluation science) |
| R1: P3 and the ranking claim (v3.4, November 14) | P3 stated as "far pairs keep their order" | The claim is worded as misordering in excess of the same judge's on real rollouts; if J1 misorders far pairs on real rollouts in any seed, a supported P3 and headline are worded as limited additional ordering error in imagined rollouts, and otherwise as far-pair misordering in imagined rollouts below 5 points, never as correct ranking; absolute far-pair misordering against simulator truth and the two discordant shares are reported next to P3 (core Summary, §B, §D; §1.3, §1.4, §1.7) | Coauthor review of the v3.4 PDF, comment 1 |
| R2: J2 fallback (v3.4, November 14) | Core §C called S1 untested under the 5 cm rule while §D called it a weaker check | Three tiers: J2 passes at 2.5 cm (S1 as registered); passes only under the 5 cm rule (S1 tested and labelled; attribution to the world model only with that qualifier); fails under the 5 cm rule (S1 untested, p = 1). The 5 cm tier's pass criterion is stated (balanced accuracy at least 0.90), and the share of episodes decoded between 2.5 and 5 cm is reported (core §C, §D; §1.2, §1.7) | Coauthor review of the v3.4 PDF, comment 2 |
| R3: P2/S6 calibration scenarios (v3.4, November 14) | Every policy's absolute true shift at the margin, six positive and six negative; estimated coverage of at least 95% | Five shift patterns with the mean absolute shift at the margin (equal, a quarter at zero, graded, concentrated, half near zero), each with alternating signs and with every sign positive; the simulation is run once at exactly 20,000 replicates with a seed fixed beforehand (was "at least 4,000"); a percentile passes only if the lower end of the one-sided 95% Clopper–Pearson interval for its coverage is at least the required level (core §D fallback 7; §1.4, §1.5) | Coauthor review of the v3.4 PDF, comment 3. The planning run showed the lowest coverage in the same-sign equal pattern, so same-sign runs are included; it also showed that the replicate count can change the calibrated percentile, so the count is fixed |
| A1.1, A1.2, A1.3: reuse gap and model ranking (v3.4, November 14) | "A 0.010 gap shifts a model comparison by 0.001" | On a test set with reuse fraction p, C's error advantage over B is pG − H; pG is the increase relative to an evaluation on independent events only, not C's full advantage; on independent events only the advantage is −H; the order of C and B reverses between the two evaluations only if H lies strictly between 0 and pG (core Summary margins table, §F; §3.5) | Coauthor review of the v3.4 PDF, comment 4. No rule, threshold or analyzer changes |
| R5: P2 and the headline (v3.4, November 14) | P2 decided separately and reported next to the headline | Also: the headline does not establish a large shift; the abstract and the claims ledger state the P2 outcome in the same sentence as the headline, in one of three fixed forms (supported, absent, inconclusive) (core Summary, §A, §D; §1.7) | Coauthor review of the v3.4 PDF, comment 5 |
| A2.1: base commit and analyzers (v3.4, November 14) | Registered commit on top of `831f0b6`; analyzers recorded by a code amendment | Registered commit on top of `28e4ada`; the Experiment 2 and 3 analyzers that implement the registered rules are already at that commit (code commit `f361b7e`, merged November 9), so the code amendment is dropped (core §A, §G; §0, §3.6, §5, §7, §10) | `main` moved after `831f0b6` |
| Code pointers (v3.2) | n/a | Code at registration named by commit for every repository, including the original Experiment 2 pipeline (§7); the Experiment 3 analyzer commit is recorded by amendment before that confirmation | The OSF record points to specific commits; later changes are amendments |
| Exp 2 data and decisions (v3.2) | Owner reruns the certificate on his cache; owner confirms each decision | The rebuilt copy is the registered data; the internal signers adopt the implemented defaults | Removes a dependency that cannot change any registered rule; the rebuilt copy is reproducible from public shards |

**Inference conventions.**

- **Unit of replication:** one training run (Experiments 1–2) or one block (Experiment 3).
- **Outcomes for every test:** supported, absent (shown by equivalence), refuted (a bound past the threshold in the opposite direction), or inconclusive. Missing a threshold never counts as absence.
- **Support bounds:** one-sided 95% for Experiments 1–2; two-sided 95% for Experiment 3.
- **Absence bounds:** 90% intervals.
- **Why the support levels differ:** Experiments 1–2 keep the October 23 plan's one-sided 5% tests. Experiment 3 is stricter (two-sided 95%, one-sided 2.5%) because one interval of G drives several competing magnitude declarations (§3.5).
- **Multiplicity:** Holm within each experiment's secondary family.

## 1. Experiment 1: Closed-loop test in simulation (primary)

> Rules locked at registration; the closed-loop workstream confirms the operational values listed in the Experiment 1 owner guidance and records them in the November 20 amendment.

**Owner:** closed-loop workstream. **Status:** hypotheses, margins, decision rules and the inference method are locked at registration. Operational values that the pilot fixes (token grid, training budget, judge weights and thresholds, final policy hashes) and the analysis code are recorded in the November 20 amendment, before any Experiment 1 outcome is computed. Pilot outcomes stay blinded until then; pilot checks print only pass or fail, and the analysis code checks the amendment's hash before reading any outcome.

**1. Question and hypotheses**

As the training failure share φ changes, how far do world-model success estimates move for distinct policies, and do rankings hold? Quantities are in percentage points. *World-model shift* is the judge's estimate on imagined rollouts minus its estimate on real simulator rollouts of the same policy from the same start states; *judge bias* is the judge's estimate on real rollouts minus true success; *excess misordering* is how often imagined estimates put a pair in the wrong order (ties wrong) minus how often the judge on real rollouts does. Far pairs are at least 15 points apart in true success; close pairs 5 to 10 points. *Why these margins:* 3 points is the smallest shift treated as consequential, the same margin as Experiment 2, so both experiments share one scale; 5 points is the smallest gap within a close pair, so an error above 5 points can reverse a close comparison between two policies a team might choose between.

| ID | Question | Supported if |
|---|---|---|
| P1 | Does failure share move estimates? | Shift at φ=.10 minus φ=.65, averaged over policies, has a one-sided 95% lower bound above 3 points |
| P2 | Is the error large at a 10% failure share? | Cross-fitted average absolute shift at φ=.10 has a bootstrap lower bound above 5 points |
| P3 | Is the excess misordering of well-separated policies limited? | Excess misordering on far pairs at φ=.10 has an upper bound below 5 points |
| S1 | Does P1 hold with a second judge? | P1 also holds with J2 |
| S2 | Does the shift fall steadily? | The slope of shift over φ=.10,.25,.50 has an upper bound below 0 |
| S3a | Does cheap recalibration fix the levels? | A one-offset logit refit on the anchor policies halves the mean absolute error against truth on the other policies at φ=.10: upper bound of the ratio below 0.5 |
| S3b | Does it need a slope too? | The same with an intercept-and-slope logit refit (logistic recalibration): upper bound of the ratio below 0.5 |
| S4 | Are close comparisons fragile? | Excess misordering on close pairs at φ=.10 has a lower bound above 5 points |
| S5 | Does it hold for a stochastic model? | P1 holds for the stochastic world model |
| S6 | Is the error large with no failures? | Cross-fitted average absolute shift at φ=0 has a bootstrap lower bound above 5 points |
| S7 | Does the effect survive more capacity? | At φ=.10 minus .65, the shift of the largest model (width 512, 9,000 episodes) has a lower bound above 3 points |

The headline needs P1 and P3 together (intersection–union, each at 5%, no correction). P2 is a registered primary test of the size of the error at a 10% failure share; it is decided at its own one-sided 5% and reported next to the headline, but the headline does not require it. Each of the two claims controls its own error; no joint rate is claimed. S7 is the capacity sweep of the plan, promoted from descriptive to a secondary with five seeds; its slope of shift over log width is reported descriptively.

**2. Design**

*Task.* ManiSkill3 PickCube-v1, Panda, control mode `pd_joint_delta_pos` throughout. One fixed horizon: the shortest of 50/75/100/150/200 steps at which the noiseless motion planner succeeds in 95% of 1,000 episodes (about 100; the episode limit is raised to match). Terminations are ignored; after finishing, generators hold still with the gripper closed. Success is read at the final step: cube centre within 2.5 cm of the goal and the arm static (arm joint speeds at most 0.2 rad/s, finger joints excluded). Planner actions are the next planner target minus the current joint positions, computed online and clipped to the controller limit.

*Inputs.* The base camera at 224×224, frozen DINOv2-small: class token plus the patch grid pooled to 4×4 (17 tokens; 8×8 = 65 tokens as the fallback), joint positions and velocities, and the goal position. `tcp_pose` and `is_grasped` are excluded.

*Training data.* Motion-planner and PPO-teacher episodes, 50/50. Each episode draws its settings (generator, noise scale, grasp-offset scale, early-release probability) from a small fixed set; every settings cell contributes equal successes and failures; cells with success below 10% or above 90% are dropped. Pools: 10,000 successes and 10,000 failures. A classifier on settings and start pose must predict pool membership with AUROC at most 0.55; if start pose predicts it, each failure is paired without replacement with the nearest success in start pose within its cell. The intervention is therefore outcome-stratified resampling of matched pools at a fixed training budget; it changes coverage as well as outcome composition, and per-arm coverage and failure types are published.

*World model.* A small latent transformer (DINO-WM style; width 256, depth 4), last 3 frames in, next tokens and proprioception out one step at a time, trained with a 4-step rollout loss (16 as the fallback). It rolls out a full episode from one real frame.

*Arms.* φ∈{0,.10,.25,.50,.65}, 3,000 episodes per arm (transitions matched by the fixed horizon), 12 seeds per arm. A seed sets initialization and the episode subsample, drawn from a fixed per-seed ordering of each pool so subsamples are nested across shares.

*Policies.* 12 policies, 4 from each of behaviour cloning, a PPO teacher distilled by DAgger, and ACT; all receive the same inputs in simulator and world model; BC and ACT train on the frozen tokens from planner successes outside the world-model pools; each family's four come from at least two training runs. Policies act deterministically; ACT runs each chunk open loop without temporal ensembling. The distilled-PPO family imitates the teacher that generated half of the world-model training data, so its rollouts sit closer to that data than the other families'; results are reported per family, and the leave-one-family-out sensitivity (§1.4) shows whether this family drives a result.

*Judges.* Both see arm speed. J1 (primary): a small MLP on the final step's tokens, joint positions and velocities, and goal, trained on held-out generator episodes balanced by outcome, then frozen. It labels success above probability 0.5, fixed in advance; it must reach AUROC 0.80 on real rollouts and expected calibration error (10 bins) below 0.05, with Platt scaling as the fallback. J2 (outcome-label-free check): decodes the 3D cube position from final-step tokens and joint positions and applies the success rule with joint velocities (in imagined rollouts, the world model's predicted tokens, joint positions and joint velocities); it must reach median cube error under 1 cm and balanced accuracy 0.90. If J2 fails: the 8×8 grid; if it still fails, J2 alone uses a 5 cm rule while truth and J1 stay at 2.5 cm; under that rule J2 passes if its balanced accuracy against true success on real rollouts is at least 0.90; its median cube error is reported and is not a pass criterion, and the share of episodes whose decoded final cube distance lies between 2.5 and 5 cm is reported for real and imagined rollouts at φ = .10 and .65.

*Evaluation.* Truth from 10,000 simulator episodes per policy on one shared start-state set, environment seeds disjoint from world-model and judge training (checked by hashing). Each seed uses a fixed random 2,000 of these starts, shared across shares, policies and judges; seeds 1–6 draw from the first 5,000 and seeds 7–12 from the rest. Evaluation-policy rollouts never enter world-model or judge training, and judge training data is kept separate from world-model training data.

**3. Procedure**

1. *Data.* Train the PPO teacher on privileged state at the fixed horizon; generate the pools; run the balance check.
2. *Judges.* Validate both on real rollouts as above.
3. *Pilot.* World models at φ=.10 and .65, 3 seeds, no policies. Training budget = the step count that minimizes 100-step open-loop validation error at φ=.10. Competence, on open-loop replays of 1,000 held-out episodes at both shares, averaged over seeds: decoded final cube-to-goal distance separates real successes from failures (AUROC 0.75), predicted final stillness reaches balanced accuracy 0.85, and a quick BC run on the same grid reaches 75% success. Fallbacks in order: if the competence check fails, the 16-step loss, then the 8×8 grid; if the BC check fails, the 8×8 grid (a grid change retrains and revalidates both judges and reruns the pilot). If a check still fails at the last setting, the experiment runs and the failure is reported with every result; what such a run may claim is fixed in §1.7. The pilot also reports J2's validation estimates (median cube error, balanced accuracy overall and on the stillness component) with 95% intervals and whether its 2.5 cm gate passes, so S1's strength is known before the main sweep. Grid and budget are fixed before any evaluation policy is trained; the pilot records GPU-hours.
4. *Policies.* Save at least 15 checkpoints per family and score each on 1,000 simulator episodes. **Select on true success only** by a fixed rule: maximize cross-family close pairs, with at least two policies in each quarter of the success range and at least 20 far pairs; ties go to more far pairs; fallbacks: far-pair minimum 15, then one quarter with a single policy. Measure truth on 10,000 episodes, **report each policy's J1 bias**, hash the policies. Far and close pair classes for analysis use the 10,000-episode truth. Fallback if the third family is late: 8 policies from two families, two per quarter, at least 10 far pairs, 2 anchors for S3a (S3b descriptive).
5. *Main sweep.* 5 shares × 12 seeds; roll out all 12 policies in every model; score imagined and real rollouts with both judges; replay each policy's real actions open loop through the same models.
6. *Mechanism arms.* Stochastic model (S5): same transformer plus a per-step latent inferred from the next frame in training and sampled from a learned prior in rollouts (DreamerV3-style KL balancing), all five shares, 5 seeds; counts only if the latent is used (on at least 5% of held-out episodes, 20 samples from the same start and actions do not all get the same J1 label) and the competence check passes; otherwise S5 is untested. Capacity and data sweep (S7): widths 128/256/512 × 3,000/9,000 episodes at φ=.10 and .65, 5 seeds, trained for the pilot's step count times the data-size ratio. *Descriptive encoder arm* (separate from S7): added if the pilot's cost projection allows (descriptive): the same width-256 transformer at φ=.10 and .65, 5 seeds, conditioned on a frozen V-JEPA 2-L encoder (pooled tokens per 2-frame tubelet, as in Experiment 3's architecture axis) and rolled out in that feature space, with a readout fitted once on the full balanced pools (not per arm) that maps its features to the 17 DINOv2-small tokens the policies and J1 read, so policies and judges are unchanged; the arm's reference is real rollouts read through the readout's tokens of real frames, so the readout's error on real frames cancels in the shift; its shift is reported next to the main model's. It shows whether the effect holds when the world model learns on video features pretrained by latent prediction, the recipe of V-JEPA 2-AC; the closed-loop workstream confirms feasibility and cost in the November 20 amendment. PushCube-v1 replication: stretch, if budget remains.

**4. Analysis rules**

- *Unit.* One training run (seed). Each criterion is computed per seed, weighting policies equally, and bounded by a t interval over seeds; S2 uses each seed's least-squares slope (all five shares are also reported; the fall is called *steady* only if S2 passes and the shift at φ=.25 lies between those at .10 and .50, otherwise *decreasing*), S3a and S3b each seed's log ratio. The exception is the cross-fitted quantities (P2, S6): cross-fitting makes the two seed halves dependent, so their bounds come from a bootstrap of the whole procedure (below). Policies are fixed and crossed with seeds, so every bound describes these 12 policies (8 under the fallback), not a population of policies. A bootstrap over seeds and policies (policies resampled within family) is a sensitivity check for P1 to P3, not the decision method: it shows how wide they would be if the policies were viewed as a sample, next to the decision, which stays conditional on these 12.
- *Outcomes.* Supported; absent or refuted (equivalence or opposite-direction bound); or inconclusive. Missing a threshold never counts as absence. One-sided 95% bounds for support; 90% intervals for absence. Decisions use bounds, never point estimates.
- *Margins for absence.* P1, S1, S5, S7 absent when the 90% interval lies within ±3 points. P2 (and S6) absent when the bootstrap upper bound of the mean over policies of each policy's absolute 12-seed mean shift is below 2 points (averaging over seeds before taking absolute values keeps this reachable; it is still biased only upward). S4 absent when the 90% upper bound of close-pair excess misordering is below 5 points. P3 refuted when the lower bound of far-pair excess misordering is above 5 points; S3a or S3b refuted when the lower bound of its ratio is above 0.5. S5 with 3 seeds is descriptive.
- *Absolute shifts.* Noise biases averaged absolute values upward, so P2 and S6 use **split-seed cross-fitting**: seeds 1–6 and 7–12 are fixed halves; in each seed, each policy's shift is multiplied by the sign of its mean shift in the other half, then averaged over policies. This is biased only downward and is used for support. The absence test uses the mean over policies of each policy's absolute 12-seed mean shift, which is biased only upward. **Intervals:** for both, the whole procedure is bootstrapped: 10,000 draws resample seeds with replacement within each fixed half (six from seeds 1–6, six from 7–12), recompute every policy's sign from the other half of the draw, and recompute the average; the one-sided 95% bound is the bootstrap percentile (support: lower percentile of the cross-fitted average; absence: upper percentile of the absence statistic; percentiles fixed by the calibration below). The bootstrap seed is fixed in the analysis code. With six seeds per half the bootstrap is still coarse (462 distinct resamples per half), so the analysis code reports its Monte Carlo error and the pilot's synthetic simulation reports its coverage. **Registered calibration of the percentiles** (as in the registration core, §D fallback 7). The pilot's synthetic simulation sets the bootstrap percentiles before the main sweep. It is run once, with 20,000 replicates per scenario, 10,000 fresh bootstrap draws in each replicate, and a random seed written in the script before the run. (a) Scenarios. Every scenario holds the mean absolute true shift at the margin (5 for support, 2 for absence). Five patterns set the policies' absolute shifts, as multiples of the margin: equal (every policy at 1); a quarter at zero (a quarter of the policies at 0, the rest at 4/3); graded (policy k of n at 2(k − 1)/(n − 1)); concentrated (a quarter at 2.8, the rest at 0.4); and half near zero (half at 0.1, half at 1.9). Each pattern is run twice: with signs alternating across policies ordered by absolute shift, and with every sign positive. The scenarios also vary the per-policy, per-seed SD of the shift (1 and 3), the cross-policy correlation ρ (0 and 0.5) and the error distribution (normal and t(3)). Each error is SD × (√ρ·c + √(1 − ρ)·e), independent across seeds, where c is shared by the policies in a seed, e is specific to a policy, and both are standard normal or t(3) divided by √3. That gives 80 scenarios, each run at both margins; the same patterns apply to 8 policies under fallback 4. (b) Pass rule. Coverage is the share of replicates in which the support lower bound is at or below the margin, or the absence upper bound is at or above it. Percentiles are searched on a 0.25-point grid. To allow for Monte Carlo error, a percentile passes a scenario only if the lower end of the one-sided 95% Clopper–Pearson interval for its coverage in that scenario is at least 95%. (c) Percentile for the bounds. The decision uses the least extreme percentile that passes every scenario; it is never less extreme than the 5th (support) or 95th (absence) and is capped at the 1st or 99th. A bound that does not pass at the cap cannot establish support or absence. (d) S6's Holm p-value. The same simulation calibrates, for each Holm level α_j = 0.05/j (j = 1, …, 8), the least extreme percentile on the grid that is no less extreme than 100·α_j, is no further out than the 0.25th, and passes every scenario with required coverage 1 − α_j (the lower end of the same one-sided 95% Clopper–Pearson interval is at least 1 − α_j). Calibrated percentiles are monotone in α_j; a level with no such percentile is unreachable. S6's p-value is the smallest reachable α_j at which its lower bound exceeds 5, or 1 if there is none. In a planning run the 0.25th percentile covered about 0.978 in the worst scenario, so S6 is expected to be declarable only late in the Holm sequence (§1.5). The percentiles and the calibration script's hash are recorded in the November 20 amendment, before any Experiment 1 outcome is computed. (With equal-size shifts the plain 5th percentile covers only about 0.91–0.92 at six seeds per half, so calibration is expected to move it. §1.5 gives a planning run of all 80 scenarios.)
- *S3a, S3b.* Anchors are the policies whose true success is closest to the 20th, 40th, 60th and 80th percentiles of the selected set (33rd and 67th with 8 policies), 100 labelled real rollouts each, drawn for each seed from that seed's own real rollouts. S3a fits a, and S3b fits a and b > 0, in logit(real success) = a + b·logit(imagined success), with b = 1 for S3a, by binomial maximum likelihood over each anchor's 100 labels; imagined success rates are clipped to [1/4,000, 3,999/4,000] before any logit is taken. Each fit is applied to the other policies; the ratio is their mean absolute error against truth after the fit over before, per seed. If weak policies are overestimated most, an offset alone cannot correct them, which S3b tests. S3b with two anchors (8-policy fallback) is descriptive.
- *Calibration slope and intercept (descriptive).* Per seed and share, the slope and calibration-in-the-large of imagined success against true success across the 12 policies, on the logit scale, with t intervals over seeds. A slope below 1 means weak policies are overestimated more than strong ones.
- *Multiplicity.* Each secondary's support test gives a one-sided p-value against its threshold, Holm-adjusted across S1, S2, S3a, S3b, S4, S5, S6 and S7 (eight tests). An untested or descriptive secondary enters with p = 1, so the family always has eight members; S6's p-value comes from the calibrated bootstrap (core §D, fallback 7). Absence and refutation calls for secondaries are unadjusted, labelled as such, and enter the claims ledger only as exploratory readings. If all seed values of a quantity are equal (likely for far-pair misordering), its bound is that common value; P3 is also reported with an exact sign-flip permutation bound. Descriptive analyses have no pass or fail.
- *Descriptive reports.* Selection regret per share (in each world model, every share and seed, pick the policy with the highest imagined success among the 12; regret is the best true success minus the pick's true success; reported as the mean over seeds at each share, next to the regret of picking with the judge on real rollouts; called *high* when its one-sided 95% lower bound over seeds exceeds 5 points); closed-loop minus open-loop shift per policy; absolute far-pair misordering against true success at φ = .10 for imagined rollouts and for J1 on real rollouts, reported next to P3 as the mean over the 12 seeds with a two-sided 95% t interval, together with the share of far pairs that J1 orders correctly on real rollouts and wrongly on imagined rollouts and the reverse share (if J1's real-rollout rate is above zero in any seed, a supported P3 and a supported headline are worded as limited additional ordering error in imagined rollouts; if it is zero in every seed, as far-pair misordering in imagined rollouts below 5 points against true success; never as correct ranking; core §D); false-success rate; **judge bias per policy, with a sensitivity check that repeats P1–P3 on policies whose J1 bias is within 5 points**; calibration, Pearson and Spearman, mean maximum rank violation; **all-pairs excess misordering** (all 66 pairs, with the far/close split), since far pairs rarely flip; **judge-recalibration sensitivity**: imagined rollouts re-scored with J1 after Platt recalibration fitted on real rollouts of the same policies, reporting how much of the shift and of P1 moves; AUROC by φ; per-policy and per-family results.
- *Exploratory: open-loop bridge to Experiment 2.* The open-loop shift between φ=.10 and .65 in this experiment is reported on the same scale as Experiment 2's Δ_O and Δ_A, to show whether simulation reproduces the real-data effect before its closed-loop result is relied on.
- *Exploratory: released world model.* If a publicly released world model can be run on PickCube observations (named, with its version, in the November 20 amendment; pending the closed-loop workstream), the 12 policies are scored with it under the same judges and start states. It shows whether a larger model trained by others shows the same level error; it enters no confirmatory decision.
- *Failed runs.* An infrastructure failure (input/output, memory, device or preemption error) at any stage is retried once with the identical seed: training resumes from the last complete checkpoint, and rollouts and scoring are rerun from the trained checkpoint on the same starts. A training failure (a non-finite loss, or a crash with no infrastructure cause) that is detected before any rollout of that run is scored reruns its seed index once, in every failure share of its model configuration (five for the main sweep and S5, two for S7), under the next unused reserve index (13, 14, 15 for seeds 1–6; 16, 17, 18 for seeds 7–12). The rerun keeps its half and that half's evaluation starts, and gets the same infrastructure retry. No other failure is replaced. Any run that remains unresolved, including a failed retry, a failed reserve run or a training failure found after scoring has started, leaves its index missing, and every decision that uses it is inconclusive. A sensitivity analysis drops every replaced index from all arms. If the primary analysis supports a claim and the sensitivity analysis does not, the claim is reported as "supported (not robust to replacement)". It counts as supported in the decision rules, but neither it nor a headline that relies on it is stated as supported in the abstract. Any other disagreement is reported beside the primary call, which stands. Invalid imagined rollouts are scored as failed episodes. The bootstrap in the sensitivity analysis resamples the seeds present in each half. This rule takes precedence over §6 for Experiment 1. Replacements are reported by arm. Nothing is excluded after outcomes are visible; any change after the amendment is a deviation. Sensitivity checks leave out one seed, one policy family, or one run.

**Limitation: world-model error and judge error are separable only in part.** The shift cancels the judge's bias on real rollouts, J2 is a second reading, and the replay check (core §D: each imagined action sequence executed open loop in the simulator from the same start) measures the world model's error on the action sequences policies produced in imagination, but none identifies how the judge reads imagined frames. A supported P1 therefore means that the evaluation pipeline's estimate moves with the failure share; attributing the move to the world model alone also needs S1 (§1.7). The design cannot remove this, and every result statement carries it. Three registered analyses bound it from different sides: S1 (an outcome-label-free judge), the per-policy J1 bias with the within-5-points sensitivity, and the judge-recalibration sensitivity; none measures the judge's error on imagined frames.

**Further limits.** *Failure share* in every reading means the registered intervention, outcome-stratified resampling of matched pools at a fixed budget; it changes coverage as well as the share of failure labels (§1.2), so no reading is about failure labels as such. *Horizon:* the world model, policies and judges share one fixed horizon (about 100 steps); results may not transfer to other horizons.

**5. Power (planning)**

Twelve seeds give about 95% power to clear 3 points when the true effect is 6 points and the per-seed SD of the policy-averaged difference is 3 points (ten seeds: about 90%; the open-loop seeds varied by under 1 point). P2 needs a true average absolute shift of about 5.9 to 7.7 points for 90% power as the per-seed SD goes from 1 to 3 (t approximation; the bootstrap's power comes from the pilot's synthetic simulation); the open-loop shift at φ=.10 was 4.8 points, so an inconclusive P2 is possible and is reported as such. Power at effects closer to the margin is lower: at 12 seeds and SD 3, P1 has about 0.29 power at a 4-point effect and 0.70 at 5 points (0.49 and 0.95 at SD 2). S7 at five seeds has power of about 0.58 at SD 3 (0.86 at SD 2) before Holm and about 0.15 (0.36) at Holm's first step (α = 0.05/8); a 90% chance needs a true effect near 8 points before Holm and about 11.6 points at Holm's first step (SD 3). A planning simulation of the 80 registered calibration scenarios (4,000 replicates and 2,000 bootstrap draws each; not the registered script) gives the following. The lowest coverage is in the equal pattern with every sign positive: across its noise settings the plain 5th percentile covers about 0.91, the 1.5th about 0.956 and the 1st about 0.962 (absence: 0.908, 0.954 and 0.964 at the mirrored percentiles). An independent re-implementation with fresh bootstrap draws agreed within Monte Carlo error. The concentrated pattern covers about the same (0.915 at the 5th percentile). Patterns with shifts near zero cover more (0.92 to 0.99 at the 5th percentile), because a wrong cross-fitted sign pulls the estimate down and the absence statistic up. Under the Clopper–Pearson rule at the registered 20,000 replicates, the calibrated percentile is therefore expected to lie between the 1.75th and the 1st, inside the cap (at 4,000 replicates the rule's threshold of about 0.956 would sit at the planning estimate for the 1.5th percentile, which is why the replicate count is fixed in advance and set higher). At a mean absolute shift of 8 points, P2's power at the 1st percentile is at least 0.99 at SD 1 and about 0.84 or more at SD 3 (lowest for the concentrated pattern with every sign positive; 0.95 or more for the other patterns). The 0.25th percentile covers about 0.978 in the worst scenario, so S6 is expected to be declarable at the Holm level 0.05, and at 0.025 only if its worst coverage clears the rule's threshold of about 0.977; it is not declarable at any smaller level, and only late in the Holm sequence. Joint power of the headline (P1 ∧ P3) is lower than each test's; the pilot's synthetic simulation reports it, and P2's power separately.

**6. Compute and stopping**

About 2.0 million imagined closed-loop rollouts and 1.4 million open-loop replays plus pilots and sweeps; 240–600 GPU-hours as designed (12 main seeds), capped at 1,000. The pilot's timing projects the total; if it exceeds the cap: drop the stretch arm, then the descriptive encoder arm, then cut S7 and S5 to 3 seeds (descriptive); if the main sweep alone would exceed it, the design is amended as a deviation before it runs. Stop at scheduled counts; never extend for significance.

**7. What the results would mean**

| Outcome | Meaning |
|---|---|
| P1 and P3 pass (headline), and S1 passes | For these 12 policies, world-model evaluation misorders policies at least 15 points apart less than 5 points more often than the same judge on real rollouts, and the training failure share moves the estimates under both judges. Whether levels are far off is decided by P2; what the result says about closer pairs depends on S4 |
| Headline and P2 pass | As above, and the shift is large (over 5 points) at a 10% failure share |
| Headline passes while J1 on real rollouts misorders far pairs in any seed | Limited additional ordering error in imagined rollouts; the pipeline is not shown to rank far pairs correctly. Absolute far-pair misordering against simulator truth is reported next to P3 |
| P1 passes with J1, S1 absent with J2 | The J2 shift is within the margin while J1's is not shown to be; whether the judges differ is read only from the paired J1 − J2 contrast (descriptive). The shift is not attributed to the world model alone. The headline is stated for the J1 pipeline and is not attributed to the world model |
| P1 passes with J1, S1 inconclusive or untested | The estimate moves under J1; attribution to the world model is untested |
| S4 passes | Close pairs (5–10 points apart) are misordered more often in imagination than on real rollouts: do not use the evaluation to choose between close policies |
| S4 absent (90% upper bound below 5, unadjusted) | Close-pair excess misordering is below 5 points for these policies; close-pair ordering is not shown to be reliable either |
| P3 refuted | For these policies, even pairs at least 15 points apart are misordered more often in imagination; the open-loop ranking result does not survive distinct policies |
| P1 absent | For these policies and this task, the failure share does not move closed-loop estimates beyond ±3 points; a bounded negative result |
| P2 absent while P1 and P3 pass | Failure share moves estimates for these policies, and the mean absolute shift at a 10% failure share is below 2 points; the abstract says so in the same sentence as the headline |
| P2 inconclusive while P1 and P3 pass | Failure share moves estimates for these policies, but the size at a 10% failure share is not established. The headline alone does not establish a large shift, and the abstract says so in the same sentence |
| S3a passes | A few labelled real rollouts and one offset correct the levels for these policies |
| S3a refuted, S3b passes | An offset is not enough because the error depends on policy skill (weak policies overestimated more); an intercept-and-slope recalibration corrects it |
| S3a and S3b refuted | Recalibration on these four anchors is not enough for these policies; levels from the world model cannot be repaired cheaply after the fact |
| S5 absent with the usage test passed / untested | The stochastic model's shift is within the margin; the paired stochastic − deterministic contrast (descriptive) shows whether averaging explains the effect / the claim is stated for deterministic latent models only |
| S7 passes / absent | The effect survives the largest model tested / the largest model's shift is within the margin; the paired width-512 − width-256 contrast (descriptive) shows whether it shrinks with capacity (both for these policies) |
| S2 passes, shift at .25 between .10 and .50 / S2 passes otherwise | The shift falls steadily with failure share / it decreases, but not steadily |
| High selection regret at φ=.10 (lower bound above 5 points) | Among these policies, the error changes which policy a team would pick |
| A competence gate fails at the last setting | Results are reported but describe a world model that failed that check; the headline is stated as inconclusive about competent world models |
| J2 passes only with the 5 cm fallback | S1 is tested and keeps its own p-value in the Holm family; a supported S1 is reported as "S1 under the relaxed 5 cm rule", and P1 is attributed to the world model only with that qualifier in the same sentence, which states that the check used a 5 cm threshold where the task uses 2.5 cm |
| J2 fails under the 5 cm rule | S1 is untested (p = 1 in the Holm family), and attribution to the world model is stated as untested |
| J1 fails validation (AUROC below 0.80, or calibration error 0.05 or more after Platt scaling) | P1–P3 and the J1-based secondaries are decided as registered and stated to apply only to a judge that failed validation |

## 2. Experiment 2: Transition-matched Kuka control

> Implemented as code. Every decision below is registered as stated (v3.2: the internal signers adopt the implemented defaults; no further owner confirmation is required). The original pipeline is used unchanged.

**Owner: Experiment 2 workstream.**

This section follows OSF order. The original pipeline implements episode-count arms without transition matching or length standardization (`experiment/run_arms.py`, `experiment/wm.py` in `robot-world-model-optimism-research`). The registered construction (samplers and feasibility certificate), evaluation weights, analyzer, dry run, freeze manifest and training driver are implemented in the `exp2-kuka-matched` package (https://github.com/physicalreasoning/trajectory-lineage/tree/28e4ada/exp2-kuka-matched), which imports the owner's `wm.py` unchanged. The implementation note (the Experiment 2 implementation note in the registration package) lists what each module fixes and what was verified.

**1. Study information and hypotheses**

We will test how outcome-stratified training-data selection changes open-loop success estimates on logged Kuka episodes.

The primary hypothesis is that increasing the training failure fraction from 0.10 to 0.65 reduces length-standardized estimated success by more than **3 percentage points** under design A.

Design O measures the original effect. B tests persistence after balancing training lengths. V compares adding failures with adding successes at equal added transition counts. These interventions change coverage as well as outcome composition; they do not isolate an outcome label’s causal effect.

**2. Design plan**

Retain cached DINOv2-small features: CLS plus a pooled 4×4 patch grid, giving 17×384 features per frame. Retain the width-256, four-layer residual transformer, four-step training rollout, batch size 256, AdamW, and learning-rate schedule. Evaluation starts from the real first frame and recursively predicts subsequent frames using logged actions through the recorded episode length.

Use **2,000 optimizer steps** throughout. Existing result JSONs record 2,000; the driver default is 4,000. This preserves the completed experiment’s training budget.

Let L_i denote frames; an episode contributes L_i-1 transitions. Failure fraction φ remains an **episode fraction**, not a transition fraction.

| Design | Registered construction | Runs |
|---|---|---:|
| O | Original 3,000-episode samples: 2,700 successes/300 failures and 1,050/1,950. | 20 |
| A, primary | Exactly 31,600 distinct transitions at each φ∈{0,.10,.25,.50,.65}. Episode counts may differ. | 50 |
| B | φ=.10,.65, with equal normalized outcome-conditional length distributions (equal raw histograms would force 50% failures), and identical total length histograms across arms. | 20 |
| V | Shared 1,500-success base; add further successes (V+S) or failures (V+F) until the arm holds exactly 31,600 transitions, the design A total. | 20 |

For A, use whole episodes without replacement and seeded constrained selection satisfying both the transition total and episode fraction: one permutation per outcome pool per seed (`numpy.random.default_rng(seed)`), an episode total that is a multiple of 20 so that φ N is an integer for every φ, the prefix whose total is closest to 31,600, then same-outcome swaps that strictly reduce the gap until the total is exact. Freeze the sampler, tie-breaking, selected IDs, counts, and length histograms before outcomes (`registered/design.py`; certificate `selection.certificate.json`). Do not silently redefine φ, crop episodes, or count repeated presentations as distinct transitions.

Target 31,600 transitions for B. At each length ℓ, use a common total 20k_ℓ (18 S + 2 F at φ=.10; 7 S + 13 F at φ=.65), with caps k_ℓ≤min(⌊ S_ℓ/18⌋,⌊ F_ℓ/13⌋) from the training pool. The vector k is the integer least-squares solution closest to the pooled availability shape with Σ_ℓ k_ℓ(ℓ-1)=1,580. Within each arm the normalized outcome-conditional length distributions are then identical by construction, and both arms share the total histogram. If 31,600 is infeasible, freeze the fallback budget (largest multiple of 1,000 not above 0.9 × the maximum feasible budget) and add A comparators at that budget for both fractions before training; the run count becomes 130. This prevents sample loss from masquerading as length attribution.

For V, both arms start from the same seeded 1,500 successes and add whole episodes of one outcome until the arm holds **exactly 31,600 transitions**, so V arms are transition-matched to design A (the added amount, about 15,800, varies slightly by seed). The 110-run schedule trains two augmented models, without a separately trained base-only comparator.

Capacity arm: dropped in v3.2 (it needed a change to the original pipeline's trainer; Experiment 1's S7 covers capacity).

**3. Sampling plan**

**Existing-data disclosure (paste into registration)**

> This study uses existing Kuka data and an existing analysis pipeline. Original open-loop outcomes were known before registration: approximately 51% estimated success versus 36% recorded success, length-only AUROC 0.764 versus world-model AUROC 0.761, and a 10.3-point estimated-success drop between 10% and 65% training failures. These results informed the hypotheses and controls.
>
> The inspected JSONs give 50.94% thresholded success for the zero-failure arm, recorded success 35.57%, and a 10.262-point drop for 10%→65% failures. The corresponding mean-probability drop is 7.457 points. Close AUROC point estimates do not establish equivalence.
>
> This registration specifies new analyses and training runs on previously examined data. It does not retrospectively preregister the completed study.

**Template.** All three experiments are registered in one OSF Preregistration (§0.1); this experiment's prior analysis is disclosed in full above and in the form's existing-data disclosure. Record who accessed which outputs and when; fresh training seeds do not make the evaluation data unseen.

Retain the existing split, `(shard // 3) % 5 == 0`, and freeze the dataset manifest, failure-pool cap, normalization, judge, and evaluation episodes. Shard separation alone does not verify independence of collection runs.

Schedule seeds **100–109**, paired across arms, excluding historical seeds 0–2. Each seed controls episode selection, initialization, and minibatches. Keep datasets and judge fixed. Stop after the scheduled runs; do not add seeds based on significance.

Prepare before registration; sign the freeze and launch registered runs only after registration. Disclose any earlier runs separately.

**Sample-size rationale and power**

From `wm_pred_rate` in the seed JSONs:

    d_s=100(R_.10,s-R_.65,s) =(11.1071, 8.1071, 11.5714).

    d̄=10.2619, s_d=√((Σ_s=1^3(d_s-10.2619)²)/2) =1.8805 points.

This is the **paired-contrast SD**, not a per-arm SD. Mean-probability contrasts instead have SD 1.4707 points.

For ten independent normal seed contrasts, support requires

    d̄-t_.95,9s_d/√(10)>3, t_.95,9=1.8331.

At s_d=1.8805, SE=0.5947, so the observed mean must exceed 3+1.8331(0.5947)=4.0901.

Power was computed from the noncentral-t distribution:

    P[T_9,λ>1.8331], λ=(δ-3)√(10)/σ.

| Assumed paired SD | True drop 4 points | 5 points | 6 points |
|---|---:|---:|---:|
| 1.8805, historical estimate | 46.4% | 92.7% | 99.9% |
| 3, planning sensitivity | 25.2% | 61.8% | 89.8% |
| 5, stress case | 14.5% | 31.7% | 54.3% |

These are conditional planning calculations. Three historical seeds cannot establish A’s variance; matching and weighting may increase it. They do not establish equivalence power or Holm-adjusted secondary power.

**Dry run.** Synthetic per-episode judge scores for all registered runs were generated under eleven declared truths and passed through the same validator and analyzer that will score the study. On the real selection (300 datasets per truth, paired SD 1.88): simulated power for the primary decision at a 5-point drop 0.890 (analytic 0.927, within Monte Carlo tolerance); false support at the 3-point boundary 3.7%; false equivalence at −3 3.3%; a single missing run made every dependent decision inconclusive in all datasets; no false secondary support under the global null. The difference contrasts Δ_O-Δ_A and Δ_A-Δ_B have about √(2) the paired SD and Holm's penalty, so they are decisive only for attenuations near 10 points. Full tables: implementation note.

**4. Variables**

Primary episode response:

    h_ids=1[p_ids>0.5].

Here p_ids is the frozen judge’s success probability for episode i, design/arm d, seed s. This matches `wm_pred_rate`, the plan’s classified-success definition, and the reported 10.3-point drop. Retain mean probability as a descriptive sensitivity.

Recorded `success` supplies truth Y_i. Record episode length, shard, training membership, distinct transitions, sampled starts, and valid multistep loss terms. Report standardized judge-on-real rates, signed and absolute error against truth, Brier scores, and AUROC. A lower estimated rate alone does not demonstrate improved evaluation.

**5. Analysis plan**

**Length standardization**

Use **22 candidate exact-length bins**, L=3,…,24, with edges 2.5,3.5,…,24.5, fixed from the loader’s accepted range. Exact lengths eliminate within-bin length imbalance.

Retain lengths with both training outcomes available and at least **20 evaluation episodes per outcome**. Freeze support using labels and lengths before new model scores; do not merge bins after seeing effects. Report exclusions and remaining support.

Let n_yℓ count retained evaluation episodes, n_y=Σ_ℓ n_yℓ, and

    r_yℓ=n_yℓ/n_y, q_ℓ= (min(r_0ℓ,r_1ℓ))/(Σ_kmin(r_0k,r_1k)).

Use this overlap target because it limits emphasis on poorly supported lengths. Preserve the original eligible evaluation prevalence π, 0.355797 on the certified cache (982 of 2,760 evaluation episodes); define π_1=π,π_0=1-π.

Give episode i normalized weight

    a_i=(π_Y_iq_L_i)/(n_Y_i,L_i).

Then Σ_i a_i=1, both outcomes share q, and weighted truth remains Σ_i a_iY_i=π. Apply identical weights to imagined predictions, judge-on-real predictions, truth, and prediction-error summaries. These are **evaluation weights**, not training weights.

    R_ds=Σ_i a_i h_ids, D_ds=100(R_d,.10,s-R_d,.65,s), Δ̂_d=1/(10)Σ_sD_ds.

Report ESS=1/Σ_i a_i², outcome-specific ESS, cell counts, weights, and unweighted estimates (Σ_i a_i=1 and weighted truth =π are asserted). **Real-data certificate (cache rebuilt from the public Open X-Embodiment shards with the owner's extractor; first run October 28, recertified October 30 with the V arms under the matched-total rule; episode-manifest SHA-256 `1dcb390f…`, which no longer depends on the cache folder; the October 28 run's `4db04973…` is superseded):** 13,678 eligible episodes; training pool 3,981 successes and 6,937 failures; 2,760 evaluation episodes, π=0.3558. All 110 runs certified (673 checks, 0 failures; every A, B and V arm holds exactly 31,600 transitions). Design B is feasible at 31,600 transitions (maximum feasible 41,880), with 2,700 episodes per arm over 17 lengths. Support retains 15 lengths (5–18 and 21) and 2,502 of 2,760 evaluation episodes; ESS 1,789 (successes 750, failures 1,063). This rebuilt copy is the registered data: the draft freeze manifest binds its episode-manifest hash, and training runs on it. It may differ slightly from the cache behind the original result (the original reported 35.57% recorded success; this copy has 2,760 evaluation episodes at 35.58%), so comparisons with the historical 10.3-point drop are approximate; the registered decisions do not depend on them.

**Decision contract**

Inference targets repeated training runs conditional on the frozen data, judge, and weighted evaluation population. Compute paired seed contrasts first; episodes are not independent training replicates.

| Quantity | Support | Absence |
|---|---|---|
| Primary Δ_A | One-sided 95% lower bound >3 | 90% CI wholly inside (-3,3) |
| Δ_O,Δ_B | Drop >3 | Equivalence within ±3 |
| Δ_V=100(R_V+S-R_V+F) | Drop >3 | Equivalence within ±3 |
| Δ_O-Δ_A,Δ_A-Δ_B | Attenuation >3 | Equivalence within ±3 |

All other outcomes are inconclusive.

Use ±3 points for absence: it matches the smallest effect declared consequential and Experiment 1's margin. It is about 30% of the historical 10.3-point drop, so "absent" means the drop shrank by at least about 70%. Absence means a negligible registered contrast, not absence of every training-composition effect.

Use paired t inference. For equivalence, use two one-sided tests, with p_ eq their maximum. Keep A primary. Apply Holm at familywise 0.05 to the **ten secondary p-values**: support and equivalence for each of the five secondary quantities above. A secondary is declared when its Holm-adjusted p is at most 0.05; each interval is reported at level 1 − 2α_j, where α_j is that hypothesis's Holm step level. Nominal 90% intervals alone cannot establish secondary absence. Other metrics and intermediate failure fractions are descriptive.

**Outcome-to-interpretation contract**

| Outcome | Permitted interpretation |
|---|---|
| O supported; A equivalent; O−A supported | Effect is sensitive to the transition-budget intervention; changed episode counts also matter. |
| A supported | Effect survives transition matching and evaluation standardization; training length shortcuts remain possible. |
| A supported; B equivalent; A−B supported | Training-length balancing attenuates the effect, conditional on matched budgets and reported population changes. |
| B supported | Effect persists after removing outcome–length association in WM training; this does not isolate dynamics error. |
| A absent while O's lower bound is above 3 | Transition count or episode length produced the original result; the closed-loop claim rests on Experiment 1 (plan row) |
| A or B inconclusive | Insufficient evidence; no “effect erased” claim. |
| V supported | Failure additions lower estimates relative to success additions; benefit requires error/discrimination evidence. |

**Exclusions and missing runs**

Keep the existing 3≤ L≤24 eligibility rule. Log malformed records, missing features/actions, extraction failures, support exclusions, and counts by outcome/shard. Do not exclude difficult episodes or poor-performing seeds.

Retry infrastructure failures once with the **same seed**, preserving pairing (driver flag `--retry-failed`; the run file records the attempt and the traceback). Record algorithmic failures as outcomes. If a run remains unresolved, report worst-case contrast bounds using rates in [0,100]; make the affected confirmatory decision inconclusive rather than analyzing only successful runs (implemented in `registered/analyze.py` and exercised by the dry run).

**6. Implementation**

The registered construction, weights, analyzer, dry run, freeze and training driver are code (`exp2-kuka-matched` in the public `trajectory-lineage` repository at the registered commit (§7); the owner's `wm.py` unchanged). Every run file carries per-episode scores, the training-ids hash, selection, configuration and checkpoint hashes; files produced under another selection or configuration are refused. The judge is fitted once on a seeded 2,500 + 2,500 real-trajectory sample, pickled and hashed, with AUROC, false-positive and false-negative rates, Brier and a reliability table written; AUROC below 0.75 stops the driver: the runs do not start and Experiment 2 reports no confirmatory result. Infrastructure failures are recorded and retried once with the same seed. The implementation note lists each fragility of the original pipeline and its registered fix.

Kuka truth remains its recorded success label. For comparison, official ManiSkill3 PickCube uses a default 50-step horizon and, for Panda, success requires distance ≤2.5 cm plus maximum absolute arm-joint velocity ≤0.2 rad/s, excluding fingers. Those rules do not replace Kuka labels. [Task source](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py), [Panda source](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/panda/panda.py).

Register under embargo. The anonymized submission carries no external links.

## 3. Experiment 3: Trajectory reuse and duplicate-detector audit

**Owner:** trajectory-reuse workstream. **Status of this section:** the exposure assay (§3.1–§3.6) is fully specified, and its simulator, analyzer, and dry run exist as freeze-candidate code in the public `trajectory-lineage` repository: the 2D study in pull request #18, and the capacity, architecture and dose axes, the joint analyzer, the parallel runner and the blinded checkpoints in pull request #19 (https://github.com/physicalreasoning/trajectory-lineage/pull/19), merged as commit `6b1031b` (https://github.com/physicalreasoning/trajectory-lineage/tree/6b1031b10ac5ec757e8bd8e7f1a2b47f0ddcbfef). The Experiment 3 freeze tag is created on `main` before the confirmation. Repository paths in this section refer to that commit for history; the registered commit is `osf-registration-v3.4` (§7). An unsigned draft freeze manifest (`approvals/exp3_freeze_manifest.json`, rebuilt November 9) binds code commit `f361b7e`, the five configurations, the analyzer, the pilot-seed registry and the environment; it is signed before the confirmation starts. The detector-audit contract (§3.7) is finalized in the November 20 amendment, before any 2D test outcome is scored. Thresholds are fixed at the freeze by the internal signers (paper lead and statistics check); external comments are welcome but are not on the critical path.

### 3.1 Question and hypotheses

**Question.** If evaluation clips replay physical trajectories seen in training, with new visuals, do forecasting scores inflate? And do standard duplicate detectors catch the reuse?

**Intervention.** Within one training block, we replace 64 independent continuation events with the 64 evaluation-source trajectories, rendered with new appearances. Every other part of training is held fixed.

**Estimands.** Let L(a, P) be a block's mean bounded error for arm *a* on population *P*: *s* = unseen rerenders of source trajectories, *i* = independent events, *x* = exact renders. Contrasts are computed within each block, then averaged over blocks.

| Estimand | Definition | Meaning |
|---|---|---|
| Source gain S | L(B,s) − L(C,s) | C's improvement on reused trajectories |
| Harm H | L(C,i) − L(B,i) | C's loss on independent events |
| **Reuse gap G** | S + H | Advantage specific to reused trajectories |
| Event specificity Q | [L(D,s) − L(C,s)] − [L(D,i) − L(C,i)] | Advantage of the same event over a closer-than-training neighbor |
| Distance reference Q2 | Q with D2 in place of D | Descriptive |
| Exact gain E | L(B,x) − L(E,x) | Positive control: learning the exact renders |

At reuse prevalence π, reuse shifts C's measured advantage over B by πG, and C's advantage becomes πG − H.

**Hypotheses.**

- **H3.1 (reuse inflates evaluation):** G > 0. It is consequential if G > 0.010 arena widths, with source gain present (S > 0) and harm to independent events bounded (H < 0.001).
- **H3.2 (event specificity):** Q > 0.010. Same-event reuse helps more than a closer-than-training independent neighbor.
- **H3.3 (appearance detectors are blind):** after passing a copy gate, SSCD and DINOv2 catch fewer than 50% of appearance-only rerenders at 1% false removal.
- **H3.4 (trajectory detectors see reuse):** object-track matching catches more than 80% of appearance-only rerenders at 1% false removal, and more than optical-flow similarity (OFS-3) catches.
- **H3.5 (trajectory detectors pay a cost):** object-track matching removes close independent events at a higher rate than ordinary ones.
- **H3.6 (filtering restores trust; descriptive, as in the plan):** trajectory-level filtering removes the residual reuse distortion, and random removal of the same size does not.

### 3.2 Environment, predictor, and measure

**Physics.** Two equal-mass, equal-radius disks in a unit arena, with one exact elastic contact. The velocity components along the contact normal exchange; tangential components are preserved. States are parameterized by collision geometry, so every event contains one visible contact and the impact parameter sets how glancing it is. Accepted events stay inside the arena, with no walls.

| Parameter | Range | Parameter | Range |
|---|---|---|---|
| Contact time | 0.34–0.76 s | Disk 0 velocity (vx, vy) | (0.35–0.65, −0.25–0.25) |
| Contact point x | 0.40–0.60 | Disk 1 velocity (vx, vy) | (−0.30–0.00, −0.25–0.25) |
| Contact point y | 0.35–0.65 | Impact parameter | −0.85–0.85 (× 2r) |
| Radius r | 0.04–0.065 | | |

Contact always falls after the 8-frame observed prefix and at least 4 frames before the end.

**Verified properties** (unit tests, 2,000-event check):

- Momentum and kinetic energy are conserved exactly.
- The disks never overlap.
- Contact times lie inside the window.
- Collisions range from head-on to glancing: disk-0 deflection is 51°–156° (10th–90th percentile), median 92°.

**Identity.** Disk 0 is the disk that starts on the left. It keeps that index for the whole trajectory, and errors are scored per disk with no per-frame rematching.

**Trajectory distance.** Coordinate RMS over all 24 frames, both disks, and both axes, minimized over one global identity swap.

**Predictor.** A ConvLSTM that sees eight 64×64 RGB frames (0.04 s apart) and outputs both disks' coordinates for the next 16 frames directly; nothing is extracted from images. Capacity is a registered block factor: 16 hidden channels (1.1 M parameters; the primary level), 64 (4.5 M) and 256 (21.5 M), same architecture, learning rate, batch size and step counts. An **architecture axis** replaces the ConvLSTM by a frozen pretrained encoder, as in current latent world models, that never sees the study's data (DINOv2-small, 22 M parameters, class token plus a 4×4-pooled patch grid per frame; or V-JEPA 2 ViT-L, 326 M, pooled tokens per 2-frame tubelet) and a trained latent transformer (width 256, depth 4, 3.3 M parameters) that predicts the future disk positions from the tokens, learning rate 3·10⁻⁴. Renders stay 64×64 and are upsampled at encoding, so events, appearances and arms are identical to the ConvLSTM levels; only the dynamics model ever sees a block's training trajectories.

**Measure.** Mean over future disk-frames of min(1, ‖p̂ − p‖₂), in arena widths. Nonfinite outputs receive the cap of 1 and stay in the denominator. Errors are averaged over renders within a source, then over sources. Independent events are weighted equally.

### 3.3 Design

**Unit of replication.** One block: one independently trained base model and its five matched continuations. Frames, renders, and arms are not independent replicates.

**Capacity factor.** Every block has one hidden-channel level fixed by its index (`capacity_schedule`: deterministic interleaving in proportion to the per-level counts, so any contiguous stretch of blocks is balanced): 800 blocks at 16 channels, 400 at 64, 400 at 256. The primary claim uses the 16-channel blocks only; the larger levels serve the capacity secondary (§3.5) and are otherwise descriptive.

**Architecture axis** (two further configurations, 200 blocks each, own seeds): the same design with a latent position predictor (a transformer that regresses future disk positions from frozen encoder tokens; it predicts coordinates, not future latent states) on the frozen DINOv2-small encoder and on the frozen V-JEPA 2-L encoder. Question: does reuse lower error when a trained transformer reads frozen pretrained features, so that only the trained head sees the block's training trajectories?

**Dose axis** (two further configurations, 200 blocks each): the ConvLSTM-16 design with 16 reused source trajectories (about 50 presentations each during the 100 continuation updates) and with 256 (about 3 each), against the registered 64 (12–13). Question: how many exposures does the inflation need?

All five configurations are frozen together, run after registration, and analyzed jointly (§3.5).

**Per block:**

- 512 base-training events;
- 64 source events, each with 4 held-out rerenders and 1 exact render;
- 512 independent evaluation events;
- 256 separate competence-screen events.

**Base training.** 600 AdamW updates (learning rate 0.001, batch 32). The step count was fixed from effect-blind tuning: at 300 steps the 2D model only marginally beats constant velocity (error ratio 0.93–0.98), while at 600 it is clearly competent (0.68–0.72), and longer training plateaus.

**Competence screen.** Before any arm is assigned, a block enters the primary analysis only if its base model's error on the 256 screen events is below constant-velocity extrapolation. Failed screens are reported and enter an all-block sensitivity analysis.

**Continuation arms.** Each arm clones the base weights and optimizer state, then runs 100 updates on one shared schedule: 24 base-replay plus 8 arm examples per batch. That gives 800 arm presentations, 12 or 13 per event.

| Arm | 64 continuation events | Role |
|---|---|---|
| B | Independent replacements | Control |
| C | Source trajectories, new appearances | Reuse |
| D | One independent neighbor per source, at 0.25–0.50 × the source's nearest base-training distance | Similar physics, not the same event |
| D2 | One independent neighbor per source, in the 0.1st–1st percentile band of independent pairwise distances | Distance reference |
| E | The exact renders in population *x* | Positive control |

All arms are evaluated on the same populations. The source rerenders are unseen by every arm; only E trains on *x*.

**Arm D sampling (2D).** D neighbors are drawn from the uniform state prior, restricted to a sensitivity-scaled box and conditioned on the distance band. The 1D box multiplier of 6 is infeasible in 2D's 9 parameters: no proposal was accepted in 2 million. We use 2.5. In a 12-source check, accepted states used at most 61% of the box. The registered target is explicitly this **box-conditioned** distribution. A diagnostic check on discarded data detected no difference when the box grew from 2.5 to 3.0 (KS below critical for all 9 parameter offsets; neighbor distances 0.443 vs 0.442 × reference). The box sampler also agreed with brute-force reservoir selection in a shared band (KS 0.083 vs critical 0.175). This is limited empirical agreement, not proof of containment. A per-block descriptor, `closer_box_offset_max`, is reported for every block.

**Randomization.** Each block's data, schedule, base batches, and torch streams come from `SeedSequence(experiment_seed, block_index)`, spawned by role. Appearance seeds are unique within a block. Seeds from every pilot, timing run, and tuning run are excluded by a registry that the freeze gate checks.

**Memory positive control (plan requirement, completed).** A nearest-trajectory lookup predictor that memorizes training trajectories must show a clear reuse gap on the 2D simulator. This ran before any learned-model effect was computed:

| Blocks | Mean reuse effect | 95% lower bound | Gate | Result |
|---|---|---|---|---|
| 64 | 0.055 | 0.054 | mean > 0.0156 and lower bound > 0 | Pass |

### 3.4 Sampling plan

**Scheduled blocks: 1,600** in the main confirmation (800 at the primary 16-channel level, 400 at 64, 400 at 256), plus 200 in each of the four axis configurations (latent DINOv2, latent V-JEPA 2, dose 16, dose 256): 2,400 blocks in all. Minimum screened primary-level blocks for a decisive claim: 720; minimum screened blocks at each larger capacity level: 360; in each axis configuration: 180 (10% attrition allowed; the pilots showed 0–4%). The sample-size rationale is §3.6. The plan's earlier figure of 1,000 assumed a per-block SD of 0.022 for G. The 2D pilot's upper limit is 0.0055.

**Stopping.** Stop at the scheduled count, with no interim looks. Infrastructure failures (I/O, memory, device errors) are rerun once with identical seeds. Algorithmic failures are outcomes and are never replaced.

**Complete-block rule.** Any unresolved block failure in the main confirmation makes the primary claim and every secondary that uses main blocks inconclusive. Estimates are still reported. An axis failure affects only that configuration (§3.5); the detector audit is unaffected. For Experiment 3 this rule takes precedence over the general missing-data rule in §6. The pilot had 0 failures in 24 blocks. Even so, a small per-block failure risk compounds over 1,600 blocks, so the runner retries infrastructure failures with identical seeds before analysis, and writes every finished block to a checkpoint so a stopped session resumes without loss.

### 3.5 Analysis and decision contract

Block means and paired t intervals: two-sided 95% for support, two-sided 90% for absence. The analyzer (`scripts/analyze_2d.py`):

- recomputes every contrast from component errors;
- ignores stored contrasts and screen flags;
- checks that per-item losses average to the errors;
- checks that the run's configuration, code, and manifest hashes match the approved freeze.

The outcome is reported on four separate axes.

| Axis | Outcome | Rule |
|---|---|---|
| Magnitude of G | Consequential | lower95(G) > 0.010 |
| | Prevalence-dependent | 0.002 < lower95(G) and upper95(G) ≤ 0.010 (whole interval) |
| | Detectable, magnitude unresolved | 0.002 < lower95(G) ≤ 0.010 < upper95(G) |
| | Negligible | 90% interval of G inside (−0.002, 0.002) |
| | Reverse | upper95(G) < −0.002 |
| | Inconclusive | otherwise |
| Safeguards | Source gain present | lower95(S) > 0 |
| | Harm bounded | upper95(H) < 0.001 |
| Positive control | At the margin | lower95(E) > 0.002 |
| | Detected, below the margin | 0 < lower95(E) ≤ 0.002 |
| | Not detected | lower95(E) ≤ 0 |
| Event identity | Event-specific (confirmatory) | the Holm-adjusted secondary test rejects Q ≤ 0.010 |
| | Below threshold (descriptive) | upper95(Q) < 0.010: specificity below the consequential threshold, not absence of specificity |
| | Unresolved | otherwise |

| Primary claim | Requires |
|---|---|
| Consequential reuse advantage | Consequential magnitude, source gain present, harm bounded |
| Prevalence-dependent reuse advantage | Prevalence-dependent magnitude, both safeguards |
| Detectable reuse advantage, magnitude unresolved | That magnitude, both safeguards |
| Gap detected, safeguards not established | Any positive magnitude with a safeguard not established |
| Negligible differential gap | Negligible magnitude and the positive control at the margin |
| Negligible gap, control below the margin | Negligible magnitude, positive control detected but not at the margin: the assay learned the exact renders, but not by enough to show it would see a 0.002 gap; inconclusive about negligibility |
| Positive control not detected | Negligible magnitude, but the positive control was not detected (sensitivity unresolved) |
| Reverse gap | Reverse magnitude |
| Inconclusive | Otherwise; any unresolved main-confirmation block failure; or fewer screened blocks than the minimum |

**Margins** (arena widths): ranking margin m = 0.001; consequential 0.010; negligible ±0.002; harm 0.001. Prevalence p is the share of evaluation items that are reused. On a test set with prevalence p, C's error advantage over B is pG − H. The term pG is the increase in this advantage relative to an evaluation on independent events only. It is not C's full advantage. At 10% reuse, a gap of 0.010 gives an increase of m, the ranking margin we set for closely matched models. At 50% reuse, a gap of 0.002 gives the same increase. The order of C and B reverses between the two evaluations only if H lies strictly between 0 and pG (0 < H < pG when G is positive). In the claim labels, "reuse advantage" names the gap G, not C's full advantage pG − H.

**Changes from the plan text, and why:**

1. "Prevalence-dependent" requires the whole interval below 0.010, so an interval such as [0.004, 0.020] is no longer called prevalence-dependent.
2. The positive control is reported in tiers. *Detected* (lower95(E) > 0) shows the assay learns at this dose; a "negligible" reading needs it *at the margin* (lower95(E) > 0.002), as the plan required, because a control that learns less than 0.002 cannot show the assay would see a gap of 0.002. (v3 had required detection only; v3.2 restores the plan's margin for the negligible reading. The planning scenarios assume E = 0.004, and the pilot's per-block SD of E, at most 0.0075, gives a standard error near 0.0003 at 800 blocks, so the margin is reachable if E is near its planning value.)

**Error control.** Each declaration controls its own error:

- support: one-sided 2.5%;
- absence: TOST at 5% per side;
- secondary claims: Holm at familywise one-sided 2.5%.

Competing declarations are not jointly controlled. At G = 0.002, a false positive-magnitude or false negligible declaration occurs about 7.5% of the time. No study-wide error rate across experiments is claimed.

**Secondary family (Holm, one-sided α = 0.025; the event-identity and capacity labels follow its decisions):**

- event-specific, Q > 0.010;
- detectable, G > 0;
- consequential harm, H > 0.001;
- capacity increase, G(256 channels) − G(16 channels) > 0, a Welch t comparison between the two block groups;
- reuse detectable under the DINOv2 recipe, G > 0 in the latent-DINOv2 configuration;
- reuse detectable under the V-JEPA 2 recipe, G > 0 in the latent-V-JEPA configuration;
- dose increase, G(16 sources) − G(256 sources) > 0 (Welch).

The seven p-values form one Holm family (`scripts/analyze_exp3_axes.py`). The primary claim comes from the main confirmation alone (`scripts/analyze_2d.py`) and is never affected by an axis; an axis configuration with an unresolved block failure, or too few screened blocks, enters the family with p = 1 (so the family always has seven) and is labelled unresolved.

**Architecture labels** (per encoder): "reuse detectable under recipe" if its Holm test is rejected; "negligible under recipe" if the two-sided 90% interval of its G lies inside ±0.002 and that configuration's positive control is at the margin (lower95(E) > 0.002); otherwise unresolved. Its own four-axis classification is descriptive. **Dose label:** "dose-dependent" if the dose test is rejected; "dose-independent" if the 90% Welch interval lies inside ±0.002 and lower95(E) > 0.002 in both dose configurations; otherwise unresolved.

**Capacity axis.** "Capacity-dependent" if the capacity-increase test is rejected in the Holm family; "capacity-independent" if the two-sided 90% interval of G(256) − G(16) lies inside ±0.002 (the negligible bound) and lower95(E) > 0.002 at both levels; otherwise unresolved. Per-level gap classifications (16, 64, 256) are descriptive. A record whose capacity differs from the registered level for its block index is rejected.

**Also reported:**

- break-even prevalence m/G, with interval [m/upper, m/lower];
- a Fieller set for the harm-to-gap crossover H/G;
- every arm × population error;
- Q2;
- an all-completed-blocks sensitivity analysis;
- **encoder comparison (exploratory):** G in the latent V-JEPA 2 configuration minus G in the latent DINOv2 configuration, with a Welch 95% interval, computed from the frozen block records and outside the Holm family. The two encoders differ in pretraining (video latent prediction vs image self-distillation), size (326 M vs 22 M parameters) and input (2-frame tubelets vs single frames), so a difference is reported as an encoder difference, not as an effect of the pretraining objective alone.

### 3.6 Sample-size rationale and simulated dry run

**Effect-blinded 2D pilot** (`exposure2d_pilot_v1`, 24 blocks, seed 2026102301). The runner computes contrasts internally. The released report contains dispersions, operational counts, and one coarse flag: whether the number of screened blocks with a positive arm-E exact gain reached at least 75% of the scheduled blocks. That flag passed, and only the workstream owner saw it. No contrast location was released or inspected. The registered confirmation withholds even this flag (`blinded_release_positive_control: false`).

| Quantity | Value |
|---|---|
| Blocks completed / base-competent | 24 / 23 |
| Failures | 0 infrastructure, 0 algorithmic |
| Runtime per block (Apple M4 Pro, MPS) | 101 s |
| SD upper limits (80%) of S, H, G, Q, E | S 0.0052, H 0.0011, G 0.0055, Q 0.0041, E 0.0075 |
| Arm D box offset max | 0.96 of the box (maximum over all sources). Diagnostic check: enlarging 2.5 → 3.0 detected no difference in 9/9 parameter offset distributions (KS below critical), and the sampler agreed with reservoir selection (KS 0.083 vs critical 0.175) |
| Nearest base-training distance (median) vs D2 distance (median) | nearest base-training 0.038; D 0.017 (0.45×); D2 0.047 (1.22×) |

**Sample-size rule.** We schedule the smallest grid value giving ≥ 95% power for **both** primary decisions, even if the true block SDs are **twice** the pilot's 80% upper limits. The two decisions are a consequential claim at G = 0.015 with harm 0.0005, and a negligible claim with a detected positive control at G = 0 and E = 0.002 (planning rule of v3; superseded for the negligible claim, see below). The planning effects are hypothetical, not observations. Vectorized simulation (`scripts/plan_2d.py`, 4,000 simulations per cell, 4% attrition):

| SDs vs pilot | Scheduled blocks | P(consequential claim), G = 0.015, H = 0.0005 | P(negligible claim), G = 0, E = 0.002 | False consequential at G = 0.010 | False negligible at G = 0.002 |
|---|---|---|---|---|---|
| 1× | 200 | 1.000 | 0.959 | 0.026 | 0.049 |
| 1× | 400 | 1.000 | 1.000 | 0.026 | 0.047 |
| 1× | 800 | 1.000 | 1.000 | 0.025 | 0.051 |
| 1× | 1000 | 1.000 | 1.000 | 0.029 | 0.052 |
| 1.5× | 200 | 0.988 | 0.630 | 0.025 | 0.033 |
| 1.5× | 400 | 1.000 | 0.930 | 0.030 | 0.047 |
| 1.5× | 800 | 1.000 | 1.000 | 0.025 | 0.051 |
| 1.5× | 1000 | 1.000 | 1.000 | 0.027 | 0.051 |
| 2× | 200 | 0.876 | 0.274 | 0.029 | 0.023 |
| 2× | 400 | 0.994 | 0.714 | 0.028 | 0.031 |
| 2× | 800 | 1.000 | 0.961 | 0.029 | 0.046 |
| 2× | 1000 | 1.000 | 0.986 | 0.028 | 0.054 |

The positive control is detected with 90% power if E ≥ 0.0009 (1× SDs) or ≥ 0.0018 (2× SDs) at 800 blocks.

**Sensitivity at the minimum of 720 screened blocks:**

- Consequential-claim power stays at 1.00 at 1×, 1.5×, and 2× SDs.
- Negligible-claim power is 0.94 at 2× SDs if E = 0.002.
- If the positive control is weak (E = 0.001), negligible-claim power is 0.94, 0.66, and 0.44 at 1×, 1.5×, and 2× SDs (`results/exposure2d_plan_sensitivity.json`). This failure is conservative: the claim becomes "positive control not detected", never a false negligible claim.

**Effect of the v3.2 margin rule on these numbers.** The planning table above and the sensitivity bullets used the v3 rule (control detected, at E = 0.002), and are superseded for the negligible claim. Under v3.2 a negligible reading needs lower95(E) > 0.002. With 720 screened blocks and the pilot's SD upper limit for E (0.0075), the probability that the control clears the margin is 0.95 at E = 0.003 and 1.00 at E = 0.004 (1× SDs), and 0.43 and 0.95 at twice the SDs (normal approximation). Restated at E = 0.004 (the dry run's planning value), the joint probability of a negligible claim at G = 0 is about 0.95 at 720 screened blocks and twice the SDs, and about 1.00 at the pilot SDs. So the negligible claim keeps its planned power if E is at least about 0.004; if E is near 0.002, the result is reported as "control below the margin", which is the intended conservative outcome. **Analyzers and dry runs at the registered commit.** The analyzers that implement these rules (`scripts/analyze_2d.py`, `scripts/analyze_exp3_axes.py`) were merged on November 9 (code commit `f361b7e`), and the dry runs were rerun with them (1,600 blocks; 120 replications for the capacity design and 60 for the axes; `results/exposure2d_dryrun_*.json`). With G = 0: a dead control is reported as not detected (0.99) and never as negligible; E = 0.001 is never called negligible ("control below the margin" 0.96 at 1× SDs; below the margin or not detected at 2×); E = 0.004 gives "negligible differential gap" in 1.00 (1×) and 0.98 (2×). Claims with a positive gap are unchanged. The unsigned freeze drafts were rebuilt on this code on November 9; both signers sign before the confirmation starts.

The planner assumes normal block contrasts and independent E. The dry run below checks the registered decision code itself. Event specificity (Q > 0.010) has near-certain power at Q = 0.015 (SD 0.0041).

**End-to-end dry run.** `scripts/dryrun_2d.py` generates synthetic block records under eleven declared truths, using the pilot's SD upper limits, with Cov(S, H) identified from G = S + H. Every synthetic dataset passes full record validation and the same analyzer that will analyze the study. A separate integration test (`tests/test_exp3_2d_path.py`) runs the approved path end to end:

- a confirmatory manifest is validated;
- the export identity is checked;
- the analysis runs;
- provisional manifests and tampered exports are rejected.

Results at 1,600 blocks (`results/exposure2d_dryrun_capacity_sd1.json`, `..._sd2.json`), 120 replications per truth, at 1× and 2× pilot SDs, under the registered rules. At 1× SDs every non-boundary truth receives its registered claim in 1.00 of datasets. At 2× SDs the negligible claim falls to 0.84 at G = 0.001 (the rest inconclusive or control below the margin) and to 0.93 in the capacity-from-null truth. At the consequential boundary (G = 0.010) the claim is "detectable, magnitude unresolved" (0.94). A dead positive control is reported as not detected (0.99). A weak positive control (E = 0.001) reads "control below the margin" (0.96 at 1×; 0.40 at 2×, the rest "not detected") and never "negligible". The full operating-characteristics tables are in the Experiment 3 evidence note of the registration package.

**Effect-blind capacity pilot** (`exposure2d_capacity_pilot_v1`, seed 2026102701; 72 blocks, 24 per level; run twice on an A100, once with the sequential arm-D sampler and once with the worker-invariant parallel sampler). Released: per-level dispersions, competence counts, runtime, and the pooled coarse positive-control flag. No contrast location was released or inspected; the Drive checkpoint holds blinded records (contrasts shifted by a key that nobody opened).

| Run | Level | Parameters | Completed / competent | s per block (A100, concurrent job) | SD upper limits (80%) S, H, G, Q, E |
|---|---|---|---|---|---|
| run 1 (sequential sampler) | 16 | 1.1 M | 24 / 24 | 252 | 0.0052, 0.0011, 0.0050, 0.0037, 0.0066 |
| run 1 (sequential sampler) | 64 | 4.5 M | 24 / 24 | 260 | 0.0039, 0.0010, 0.0037, 0.0032, 0.0034 |
| run 1 (sequential sampler) | 256 | 21.5 M | 24 / 23 | 349 | 0.0340, 0.0296, 0.0054, 0.0047, 0.0061 |
| run 1 (sequential sampler) | pooled | n/a | 72 / 71; failures 0 + 0; positive control passed | 286 | n/a |
| run 2 (parallel sampler) | 16 | 1.1 M | 24 / 24 | 76 | 0.0053, 0.0008, 0.0052, 0.0034, 0.0072 |
| run 2 (parallel sampler) | 64 | 4.5 M | 24 / 24 | 79 | 0.0034, 0.0007, 0.0035, 0.0023, 0.0033 |
| run 2 (parallel sampler) | 256 | 21.5 M | 24 / 23 | 96 | 0.0064, 0.0013, 0.0057, 0.0050, 0.0050 |
| run 2 (parallel sampler) | pooled | n/a | 72 / 71; failures 0 + 0; positive control passed | 84 | n/a |

The per-level gap dispersions match the October 23 pilot (0.0055), so the 800 primary-level blocks stand. With 400 blocks at each larger level, the Welch standard error of G(256) − G(16) is about 0.00034, so a trend of 0.002 is detected with power above 0.99 and a flat truth is called capacity-independent in about 98% of simulated datasets at pilot SDs (53–65% at twice the SDs under the control-margin requirement, the rest unresolved). Simulated false "capacity-dependent" calls under flat truths: at most 2.5%. Run 1 showed inflated dispersions of C's source gain and independent harm at 256 channels (0.034 and 0.030 against 0.006 and 0.001 in run 2), consistent with one unstable continuation arm in one block; the metric is bounded at 1, the gap dispersion was unaffected, and the larger levels carry no safeguard decision.

**Capacity axis, operating characteristics (120 datasets per truth, 1,600 blocks):**

| Truth | SDs | capacity-dependent | capacity-independent | unresolved |
|---|---|---|---|---|
| G = 0 everywhere | 1× | 0.02 | 0.98 | 0.00 |
| G = 0.015 everywhere | 1× | 0.02 | 0.98 | 0.00 |
| G = 0.015 everywhere (flat scenario) | 1× | 0.01 | 0.98 | 0.01 |
| G = 0.006 ×1 / ×2 / ×3 | 1× | 1.00 | 0.00 | 0.00 |
| G = 0 / +0.004 / +0.010 | 1× | 1.00 | 0.00 | 0.00 |
| G = 0 everywhere | 2× | 0.02 | 0.61 | 0.38 |
| G = 0.015 everywhere | 2× | 0.02 | 0.55 | 0.43 |
| G = 0.015 everywhere (flat scenario) | 2× | 0.01 | 0.59 | 0.40 |
| G = 0.006 ×1 / ×2 / ×3 | 2× | 1.00 | 0.00 | 0.00 |
| G = 0 / +0.004 / +0.010 | 2× | 1.00 | 0.00 | 0.00 |

**Effect-blind axis pilots** (24 blocks each, one A100 session with three concurrent blocks, October 28). Released: dispersions, competence counts, runtime and the pooled coarse positive-control flag; no contrast location.

| Level | Completed / competent | Positive control | Failures | Compute s per block | SD upper limits (80%) S, H, G, Q, E |
|---|---|---|---|---|---|
| Architecture: latent predictor, frozen DINOv2-small | 24/24 / 24 | passed | 0 + 0 | 142 | 0.0038, 0.0036, 0.0018, 0.0025, 0.0036 |
| Architecture: latent predictor, frozen V-JEPA 2-L | 24/24 / 24 | passed | 0 + 0 | 157 | 0.0026, 0.0017, 0.0016, 0.0020, 0.0039 |
| Dose: 16 sources (~50 presentations) | 24/24 / 24 | passed | 0 + 0 | 63 | 0.0068, 0.0008, 0.0071, 0.0042, 0.0073 |
| Dose: 256 sources (~3 presentations) | 24/24 / 24 | passed | 0 + 0 | 448 | 0.0037, 0.0017, 0.0031, 0.0020, 0.0088 |

The latent world models are competent in every block (DINOv2 base error ratio 0.89 against constant velocity at 600 steps, V-JEPA 2 0.65) and their gap dispersion is about a third of the ConvLSTM's, so 200 blocks per encoder give a standard error of G near 0.00013: a true gap of 0.002 is detected with power above 0.99, and the ±0.002 negligible band is decided in essentially every dataset. The dose levels have dispersions 0.0071 (16 sources, few sources per contrast) and 0.0031 (256); with 200 blocks each, the Welch standard error of the dose difference is about 0.0005.

**Joint dry run (60 datasets per truth; every configuration simulated from its own pilot dispersions, validated and analyzed with the code at the registered commit, `results/exposure2d_dryrun_axes_sd1.json` and `..._sd2.json`):**

| Truth (G by configuration) | SDs | DINOv2 label | V-JEPA 2 label | Dose label | Main primary claim |
|---|---|---|---|---|---|
| G = 0 everywhere | 1× | negligible under recipe 1.00 | negligible under recipe 1.00 | dose independent 0.83 | negligible differential gap 1.00 |
| main 0.015; recipes 0.004; dose 0.015 → 0.004 | 1× | reuse detectable under recipe 1.00 | reuse detectable under recipe 1.00 | dose dependent 1.00 | consequential reuse advantage 1.00 |
| main 0.015; recipes 0; dose flat 0.015 | 1× | negligible under recipe 0.98 | negligible under recipe 0.98 | dose independent 0.80 | consequential reuse advantage 1.00 |
| main 0.006; recipes 0.001; dose flat 0.006 | 1× | reuse detectable under recipe 1.00 | reuse detectable under recipe 1.00 | dose independent 0.78 | prevalence dependent reuse advantage 1.00 |
| main 0.006; recipes 0.003; dose 0.010 → 0 | 1× | reuse detectable under recipe 1.00 | reuse detectable under recipe 1.00 | dose dependent 1.00 | prevalence dependent reuse advantage 1.00 |
| G = 0 everywhere | 2× | negligible under recipe 0.95 | negligible under recipe 0.90 | unresolved 0.98 | negligible differential gap 0.98 |
| main 0.015; recipes 0.004; dose 0.015 → 0.004 | 2× | reuse detectable under recipe 1.00 | reuse detectable under recipe 1.00 | dose dependent 1.00 | consequential reuse advantage 1.00 |
| main 0.015; recipes 0; dose flat 0.015 | 2× | negligible under recipe 0.98 | negligible under recipe 0.90 | unresolved 0.95 | consequential reuse advantage 1.00 |
| main 0.006; recipes 0.001; dose flat 0.006 | 2× | reuse detectable under recipe 0.92 | reuse detectable under recipe 0.95 | unresolved 0.93 | prevalence dependent reuse advantage 1.00 |
| main 0.006; recipes 0.003; dose 0.010 → 0 | 2× | reuse detectable under recipe 1.00 | reuse detectable under recipe 1.00 | dose dependent 1.00 | prevalence dependent reuse advantage 1.00 |

**Reading.** (1) The primary claim does not depend on the axes: across truths with the same main gap it is the same, apart from 2% of G = 0 datasets at twice the SDs that read "control below the margin". (2) A false "reuse detectable under recipe" call under a flat null occurs in 2% of datasets, inside the nominal 2.5%. (3) The recipe tests are sensitive: a gap of 0.001 is detected in 92–100% of datasets. "Detectable" is a sign claim; a gap of 0.001 is negligible in magnitude by the main contract, so every axis label is reported together with its configuration's descriptive magnitude class, and no axis is called consequential on the strength of detection alone. (4) A dose difference of 0.010 is detected in every dataset. Dose independence is decided in 78–83% of datasets at the pilot SDs and mostly stays unresolved at twice those SDs (2–7% decided); the failure mode is conservative, never a false dose dependence. The label needs the control at the margin in both dose configurations, which the 256-source configuration often misses.

**Is the 0.001 harm limit reachable?** The consequential and prevalence-dependent labels need the 95% upper bound of the harm H on independent events below 0.001. From the blinded pilot dispersions only (80% upper limits of the block SD of H), with the registered minimum screened blocks:

| Level | SD of H (80% upper) | Min. screened blocks | P(upper bound < 0.001), H = 0 | same, 2× SD | P, H = 0.0005 | same, 2× SD |
|---|---|---|---|---|---|---|
| Main, 16 ch (Oct 23 pilot) | 0.0011 | 720 | 1.00 | 1.00 | 1.00 | 1.00 |
| Main, 16 ch (capacity pilot run 2) | 0.0008 | 720 | 1.00 | 1.00 | 1.00 | 1.00 |
| Main, 64 ch | 0.0007 | 360 | 1.00 | 1.00 | 1.00 | 1.00 |
| Main, 256 ch (run 2) | 0.0013 | 360 | 1.00 | 1.00 | 1.00 | 0.94 |
| Main, 256 ch (run 1, one unstable block) | 0.0296 | 360 | 0.09 | 0.05 | 0.05 | 0.04 |
| Latent DINOv2 | 0.0036 | 180 | 0.96 | 0.45 | 0.45 | 0.15 |
| Latent V-JEPA 2 | 0.0017 | 180 | 1.00 | 0.98 | 0.98 | 0.52 |
| Dose 16 | 0.0008 | 180 | 1.00 | 1.00 | 1.00 | 0.99 |
| Dose 256 | 0.0017 | 180 | 1.00 | 0.97 | 0.97 | 0.49 |

At the primary level the limit is reached with probability 1.00 even at twice the pilot SDs and a true harm of 0.0005; the expected upper bound at zero harm is about 0.0001, a tenth of the limit. It is also reachable at 64 channels and in the dose configurations. Two levels are weaker: 256 channels when one unstable block inflates H (capacity pilot run 1), and the DINOv2 latent level at twice its pilot SD. Neither gates a registered decision: the capacity, architecture and dose tests depend on G only, and the safeguard classes of those configurations are descriptive.

### 3.7 Duplicate-detector audit (H3.3–H3.5; filtering H3.6 descriptive)

**Populations.** Separate calibration and test splits. Each has:

- a 500-event gallery;
- 200 positive source families × 8 transformations;
- 600 ordinary independent negatives;
- 200 close independent negatives, drawn from the arm-D distance band.

**Transformations:**

1. byte re-encoding (copy control);
2. new appearance;
3. appearance plus texture;
4. appearance plus heavy JPEG;
5. central 80% crop and resize;
6. horizontal mirror;
7. three-frame time shift;
8. moving-disk distractor.

**Detectors:**

- exact hash;
- SSCD (pinned checkpoint);
- DINOv2 ViT-S/14;
- RAFT-based OFS-3 (no natural-motion filtering; labeled an adaptation);
- object-track matching, from image-extracted centers with one global identity permutation and registered mirror and ±3-frame alignment;
- privileged state distance, as a reference only.

**Status.** H3.3–H3.5 are registered directional predictions. Their confirmatory contract is frozen in the November 20 amendment, before any 2D test outcome is scored: exact threshold quantiles and tie rule, negative-family resampling, extraction-failure handling (failure means no match and stays in the denominator), and one Holm family at familywise 0.05. If it is not frozen by then, H3.3–H3.5 are exploratory. H3.6 (filtering) is descriptive, with no pass or fail, as in the plan.

**Copy gate.** A detector counts only if it first catches ordinary re-encoded copies: AUROC ≥ 0.95 against ordinary negatives and recall ≥ 0.80 at 5% false removal.

**Operating points and inference.** Thresholds are set at 1% and 5% false removal on calibration ordinary negatives and frozen before test scoring. We report:

- recall;
- ordinary, close, and pooled false removal;
- AUROC;
- extraction failures, by transformation.

The uncertainty is a 10,000-draw bootstrap over calibration families and test source families, re-estimating thresholds within each draw. Predictions H3.3–H3.5 form one Holm family at familywise 0.05.

**Filtering (H3.6).** A fixed 500-query mixture per block (50 sources plus 450 independent events, 10% reuse) is filtered at matched retention by:

- the detector;
- 1,000 random masks;
- provenance-prioritized removal.

Reported, at the frozen 5% false-removal threshold and only if it keeps at least 90% of independent items (the plan's coverage floor): the reuse gap among retained items and the residual distortion π_τ G_τ (zero when no reused item remains), for the detector, the random masks and provenance-prioritized removal, against the share of items kept (coverage), with simulator-state distance as a reference. The plan predicts that trajectory-level filtering brings the gap within ±0.002 and random removal does not. Filtering is descriptive, with no pass or fail, as in the plan.

**Real-data check (exploratory).** The same detectors run on synchronized camera views of the same DROID episodes (real same-event pairs) and on independent same-task negatives. No models are trained.

### 3.8 Existing data, blinding, and freeze history

All confirmatory 2D events are newly generated, under seeds excluded from every pilot.

| Date (EDT) | Work | What was seen |
|---|---|---|
| Oct 6–19 | 1D simulator, memory control, detector benchmarks v1–v4 and OFS-k (OFS-k results committed Oct 19, 17:45) | Detector results (engineering); memory control |
| Oct 16–19 | 1D pilots v1–v5 | Dispersion, operational diagnostics and a coarse positive-control flag only |
| Oct 19, 16:47 | 1D protocol and thresholds frozen provisionally (`a25d881`), signed by the workstream owner only | n/a |
| Oct 19, 20:11 | 16-block 1D exploratory interim run, after the freeze, and a filtering analysis on 8 of its blocks (committed 20:52) | Effect estimates, by the workstream owner |
| Oct 20 | 352-block 1D confirmation completed: 352/352, 348 base-competent, 0 failures | **Sealed; not analyzed** |
| Oct 23 | 2D simulator, memory control, base tuning, effect-blinded 2D pilot | Competence, dispersion, runtime, one pooled positive-control flag (passed); by the workstream owner |
| Oct 27–28 | Effect-blind capacity pilot, two runs on Colab (A100) | Per-level competence, dispersion, runtime, the pooled flag (passed); by the workstream owner |
| Oct 28 | Effect-blind architecture (DINOv2, V-JEPA 2) and dose (16, 256 sources) pilots, 24 blocks each, on Colab (A100) | Competence, dispersion, runtime, the pooled flag (passed); by the workstream owner |

**Exposure record and attestation.** The 16-block 1D exploratory interim estimates of October 19 (20:11 EDT), including the exploratory filtering analysis on 8 of those blocks and the positive-control estimate, were seen by Lovejesh Bangar, who leads Experiment 3, and published in the repository README, FINDINGS and `results/exposure_interim_analysis.json` the same evening (commits `9fa1778`, `5461ca6`, `af2c376`), where any reader of the public repository may have seen them. The 1D detector benchmarks and the lookup-predictor memory checks (1D and 2D) were also seen; they are not learned-model reuse effects. Every 2D release (the October 23 pilot, the October 27–28 capacity pilot and the October 28 architecture and dose pilots) carried dispersions, counts, runtime and one coarse positive-control flag only. The 0.002 positive-control margin is the negligible bound fixed on October 16 (`docs/EFFECT_SIZE.md`, commit `f0e30eb`), before any 1D effect estimate; the requirement that the control clear it comes from the October 23 plan. Lovejesh Bangar attests that: (1) the only learned-model reuse-effect values he has seen, or has shared with any author, are the 1D interim estimates above; (2) the 352-block 1D confirmation (October 20) has never been analyzed, and its raw records are untracked on his machine and in the frozen analysis clone; (3) the 2D rule changes (the whole-interval prevalence rule and the positive-control tiers) follow from design logic stated in §3.5 and not from any 1D value; (4) no Experiment 3 effect value has been sent to an external reviewer.

The 1D interim effect values are withheld from this protocol; the 1D confirmation has not been analyzed. The 1D results are not pooled into the 2D study.

**Detector results already seen.** H3.3–H3.5 and the descriptive filtering analysis (H3.6) were informed by the 1D detector benchmarks (October 6–19) and by the exploratory 1D filtering analysis on 8 interim blocks, all public in the repository (FINDINGS §4, §6, §7). The 2D audit uses new events, renders, transformations and populations, so its tests are prospective, but they are directional replications in a new setting, not first tests, and are reported as such.

**Freeze.** Before confirmation, a manifest binds:

- the configuration SHA-256;
- the analyzer SHA-256;
- the code commit (HEAD may differ only under `approvals/`);
- the pilot-seed registry (every pilot seed: 1D pilots, 2D pilot, capacity, architecture and dose pilots);
- the four axis configurations, frozen with the main configuration (the joint analyzer `scripts/analyze_exp3_axes.py` is bound through the code commit);
- per-block SHA-256 hashes of every generated state population, written into each block record by the runner (not into the manifest) and required by the analyzer, so each block can be regenerated from (configuration, seed, index) and checked byte for byte;
- dated sign-offs by the internal signers: the paper lead and the statistics reviewer (`stats_check`). External reviewers are not on the critical path.

`scripts/freeze.py` refuses to unblind otherwise. The 2D configuration forbids provisional (owner-only) freezes (`allow_provisional_freeze: false`). The runner and the analyzer both bind `scripts/analyze_2d.py`. A confirmatory 2D run also refuses to **start** without an approved confirmatory manifest (`require_approval_to_run: true`), not only to unblind. All three properties are covered by tests.

### 3.9 Interpretation and limits

| Result | Permitted reading |
|---|---|
| Positive magnitude, safeguards pass | Reuse changes measured relative performance at this dose |
| Positive gap, harm fails | Differential performance without benign transfer |
| Q event-specific | Advantage beyond a closer-than-training neighbor |
| Negligible G, control at the margin | Negligible differential gap for this assay |
| Negligible G, control detected only below the margin | Inconclusive about negligibility: the assay may be too weak at this dose to see a 0.002 gap |
| Control not detected, or competence or completeness fails | Inconclusive about learned models |
| Detector passes copy gate, misses rerenders | Measured blind spot under that transformation |

**Limits:**

- synthetic dynamics and one architecture;
- continuation, not pretraining;
- a declared neighbor distribution;
- capped errors and finite galleries.

The study does not identify what any undocumented model was trained on. It also does not test reuse-induced policy-ranking errors. **Registered for a follow-up** (as in the plan; not a claim of this study): reuse inside the Experiment 1 setup counts as consequential if reuse shifts estimated success by at least 3 points with far-pair order kept, or if reuse that favours one policy reorders a pair at least 5 points apart.

## 4. Decision rules common to all experiments

Each endpoint records estimand, unit, averaging order, eligible population, confidence method, margin, multiplicity family, missingness rule, and interpretation.

**Confirmatory families (stated so no hidden correction is assumed).** Experiment 1: the headline (P1 ∧ P3, intersection–union), P2, and one Holm family of eight secondaries (S1, S2, S3a, S3b, S4–S7). Experiment 2: the primary Δ_A and one Holm family of ten secondary p-values. Experiment 3: the primary claim (magnitude with safeguards and positive control, from the main confirmation alone), one Holm family of seven secondaries across the main confirmation and the axes, and the detector family H3.3–H3.5 (if frozen by November 20). H3.6 (filtering) is descriptive. Error is controlled within each family; there is no correction across families or experiments, and the paper reports this count.

The unit of replication is one training run (a seed) in Experiments 1–2 and one block in Experiment 3. Average policies within runs. Do not treat policy pairs, frames, or rerenders as independent replicates. Experiments 1–2 condition on fixed pools, judges, policies, and evaluation starts; Experiment 3 regenerates events per block.

**Inference method (locked):** paired t intervals over seeds for Experiments 1–2 and over blocks for Experiment 3 (Welch t for the Experiment 3 capacity and dose contrasts, which compare separate groups of blocks). Support uses one-sided 95% bounds in Experiments 1–2 and two-sided 95% intervals in Experiment 3; absence uses 90% intervals. Every decision is made on a bound, never on a point estimate. Averaged absolute shifts (Experiment 1 P2 and S6) use split-seed cross-fitting for support and, for absence, the mean over policies of each policy's absolute 12-seed mean shift; because the halves are dependent, their bounds come from a bootstrap of the whole procedure (§1.4), the one exception to the t intervals. A bootstrap over seeds and policies is a sensitivity check in Experiment 1, not a decision method. The P1 ∧ P3 headline is an intersection–union decision; P2 is decided separately (§1.1). Holm-adjust the secondary support tests within each experiment; secondary absence calls are unadjusted in Experiment 1 and Holm-adjusted in Experiment 2, as each section states.

## 5. Access, blinding, and amendments

**Freezes live in git.** Before each experiment's confirmatory runs, the paper lead tags the commit of the experiment's repository (`exp1-freeze-YYYYMMDD`, `exp2-freeze-YYYYMMDD`, `exp3-freeze-YYYYMMDD`). The tag fixes code, configuration and, for Experiments 2–3, the freeze manifest with its hashes. OSF registers the form and the core; this protocol is in the repository at the registered commit; the tags are the code freeze. **Signers are internal:** the paper lead and the designated statistics reviewer (an author who runs no confirmatory jobs) sign each freeze manifest; the experiment owner may add a third line. External reviewers comment but are not on the critical path.

Operators see configurations, losses, competence diagnostics, runtime, and failures. Confirmatory outcomes are computed only after the corresponding tag and, for Experiment 1, only after the November 20 amendment. Prior exposure is attested in §3.8.

All three experiments register together. The registration precedes every confirmatory run: Experiment 2 training, the Experiment 3 confirmation, and the Experiment 1 main sweep start only after it.

**November 20 amendment:** limited to (Experiment 1) the operational values fixed by the pilot (token grid, training steps, judge weights, which registered fallbacks fired, gate results, ACT chunk size, the confirmatory hardware class, the exploratory released world model if any, and, if the encoder arm runs, its readout and how its 2-frame tubelet steps align with per-step actions), the final policy hashes, the calibrated bootstrap percentiles (§1.4), and the Experiment 1 repository, commit and analysis code, before any Experiment 1 outcome is computed; and (Experiment 3) the detector-audit details listed in §3.7, before any 2D detector test outcome is scored. It cannot change hypotheses, endpoints, thresholds, exclusions, or analysis families. If it is not filed in time, Experiment 1 and H3.3–H3.5 are exploratory. The Experiment 1 analysis code checks the amendment's hash before reading any outcome.

**Analyst separation.** The unblinding scripts are run from the tags by the designated statistics reviewer, an author who runs no confirmatory jobs; owners receive the outputs after the run.

**Claims ledger.** Before unblinding, the December 8 amendment includes a claims ledger: for each registered outcome (the tables in §1.7, §2.5 and §3.9), the exact sentence the abstract and conclusion will use. The paper uses those sentences.

**Reporting commitment.** All three experiments are reported in the paper or a public preprint whatever the outcome; inconclusive and negative results are reported with the same prominence as their registered status.

**Analyzers at registration (Experiments 2 and 3):** the registered commit contains the analyzers that implement the Experiment 2 Holm rule (core §C: an unresolved member enters with p = 1) and the rules of §3.5 (positive-control tiers, axis labels, p = 1 for a missing axis), with tests and rerun dry runs (code commit `f361b7e`, merged November 9). No code amendment is needed for them. The signed freeze manifests bind them before the Experiment 2 runs and the Experiment 3 confirmation start.

**December 8 amendment:** records execution and deviations and authorizes unsealing the Experiment 1 pilot and main outcomes, the Experiment 2 outcomes and the Experiment 3 reuse outcomes. Its date is a target: it may be filed per experiment, and no experiment is unblinded before its part is filed. It cannot select analyses after results. Earlier Kuka or detector reporting cannot change locked contracts. The 1D confirmation stays sealed until the 2D reuse outcomes are unsealed; it is then analyzed once with the analyzer frozen at `a25d881` and reported in the paper or its supplement as a separate, provisionally frozen study, whatever the outcome.

Archive new amendment artifacts separately under embargo; preserve original attachments. Avoid automatic inclusion of sealed files or unsanitized reviews.

**Staged Experiment 3 item.** This registration expressly authorizes the November 20 amendment to also record the Experiment 3 detector-audit contract listed in §3.7 (threshold quantiles and ties, negative-family resampling, extraction-failure handling, Holm family). It may finalize only those listed items, and only before any 2D detector test outcome is scored. Otherwise H3.3–H3.5 remain exploratory.

## 6. Deviations, failures, and missing data

Log changes before implementation where possible. Retain planned and actual analyses when valid. Post-unblinding scientific changes are exploratory.

| Item/date | Planned rule/hash | Actual action | Reason | Outcomes accessible? | Affected claims | Owner/signoff |
|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … |

Infrastructure failures permit one identical-seed/configuration retry. Preserve both attempts. Algorithmic divergence is an outcome, not grounds for a replacement seed. Exclude only frozen eligibility failures, verified corruption, or unauthorized provenance overlap; retain statistical outliers.

For unresolved missing runs, vary missing component values across registered bounds: probabilities [0,1], forecasting losses [0,1]. A decisive claim and its confidence bounds must survive every admissible completion, including uncertain eligibility; otherwise it is inconclusive. Report scheduled, attempted, screened, completed, excluded, and analyzed counts.

**Precedence.** Experiment 3 applies its own failure rules (§3.4, §3.5): any unresolved main-confirmation block failure makes the primary claim and every secondary that uses main blocks inconclusive; an axis failure enters the Holm family with p = 1 and leaves only that configuration unresolved; the detector audit is unaffected. Experiment 1 applies its own failed-run rule (registration core §D): an identical-seed retry for an infrastructure failure at any stage (training resumes from the last complete checkpoint; rollouts and scoring rerun on the same starts); a training failure detected before any of that run's rollouts are scored reruns its seed index once, in every failure share of its model configuration, under reserve index 13–15 or 16–18; no other failure is replaced; any run that remains unresolved (a failed retry, a failed reserve run, or a training failure found after scoring has started) leaves its index missing, and every decision using it is inconclusive; a complete-case sensitivity analysis drops replaced indices.

## 7. Code and data identity

Code freezes are dated git tags (section 5). The registration identifies, per experiment, the repository and the configuration whose hash the freeze manifest binds; the full asset ledger (licenses, lockfiles, checkpoint hashes, access states) and the data dictionary are delivered with the paper, not here.

| Experiment | Repository | Frozen by | Configuration |
|---|---|---|---|
| 1 | closed-loop workstream repository (named in the November 20 amendment with its analysis code) | tag `exp1-freeze-YYYYMMDD` before the main sweep | pilot-fixed values recorded November 20 |
| 2 | original pipeline (public: https://github.com/subhashjprasad/robot-world-model-optimism-research at commit `ed416e1`, `wm.py` unchanged; its `uv.lock` is the lockfile the draft freeze manifest binds) plus `exp2-kuka-matched` in `trajectory-lineage` | tag `exp2-freeze-YYYYMMDD` plus the freeze manifest (config, episode manifest, selection, support and code hashes, environment); unsigned draft at `approvals/exp2/exp2_freeze_manifest.json` (rebuilt November 9) | `configs/exp2_registered.json` (version 2026-10-28) |
| 3 | `trajectory-lineage` (public: https://github.com/physicalreasoning/trajectory-lineage; registered code: tag `osf-registration-v3.4`; the draft manifest binds code commit `f361b7e`; pull request #19 (`6b1031b`) and `ccd871e` are history) | tag `exp3-freeze-YYYYMMDD` plus the freeze manifest (configuration and analyzer hashes, pilot-seed registry, per-block state hashes, environment) | `experiments/exposure2d_confirmation.json` and the four axis configurations |

**Code at registration.** `trajectory-lineage` at tag `osf-registration-v3.4`: the commit that adds the registration core and this protocol under `docs/registration/` on top of `28e4ada` (November 9: analyzers and runner aligned with the core, a locked Experiment 2 training notebook, and both unsigned draft manifests rebuilt on that code). Its full hash goes in the form. At that commit the Experiment 2 and Experiment 3 analyzers implement the registered rules (core §C and §F; code commit `f361b7e`); the core governs wherever code and document differ. The original Experiment 2 pipeline is registered at `ed416e15ab9e4fd7c7b9ba4c51bdb343b4a28ce5`. Experiment 1's repository and commit are recorded in the November 20 amendment. Other commits cited in this protocol (`4ba132e`, `6b1031b`, `ccd871e`) are history. Later changes are recorded as amendments.

**Environment and hardware.** Each freeze manifest records a lockfile hash, the exact runtime versions and their hash, and the hardware class for the confirmatory runs (Experiment 1: named in the November 20 amendment; Experiment 2: A100, as in its draft manifest; Experiment 3: A100 on Colab). The freeze tools record and check these fields (repository, October 30), and a run on another hardware class or runtime is refused unless a deviation is logged. GPU training is not bit-identical across hardware, so a change of hardware class is a logged deviation, and a rerun uses the original class.

**Review copy.** The submission links no repository. If reviewers need the code, they receive an anonymized mirror of the tagged commits, without author names in paths, history or README; the public repositories stay unlinked until review ends.

Third-party material (ManiSkill, Open X-Embodiment Kuka, DROID, pretrained encoders) keeps its own licenses and is cited, not relicensed.

## 8. Compute and stopping

Planning estimates: Experiment 1, 240–600 GPU-hours as designed, capped at 1,000 (§1.6; about 2.0 million imagined closed-loop rollouts and 1.4 million open-loop replays); Experiment 2, 2–3 GPU-hours (about 8–9 laptop-hours); Experiment 3, about 27 hours on one A100 (measured below). Measure all training, extraction, evaluation, storage, and retry costs.

Hard cap: $2,000. Stop at scheduled counts; never extend for significance. Validate joint power, equivalence power, screening attrition, and Holm-adjusted secondary power using synthetic simulations before launch.

Priority: main experiments, stochastic robustness, capacity secondaries, then stretch work. Drop stretch work first; a three-seed stochastic or capacity fallback becomes descriptive (§1.6). Keep twelve main seeds; if the main sweep alone would exceed the cap, the design is amended as a deviation before it runs. Experiment 3's 2D qualification (memory check, competence, pilots) passed October 23–28, so its earlier fallback (report the detector audit and defer the reuse confirmation) no longer applies.

**Experiment 3 (measured).** 101 s per block on an Apple M4 Pro (October 23 pilot); on an A100 with a concurrent job, 76 / 79 / 96 s per block at 16 / 64 / 256 channels (capacity pilot, parallel sampler). The main confirmation (1,600 blocks) plus the four axis configurations (800 blocks) need about 27 hours of wall-clock on one A100 with three concurrent blocks (per-block compute 63–448 s), checkpointed per block so the run can span sessions. The detector audit's runtime is measured in its 2D adaptation, due by November 20.

**Experiment 2 (measured from the owner's log).** About 4.2 minutes per run at 2,000 optimizer steps on the owner's Apple laptop (MPS), peak memory about 4 GB with 9.8k loaded episodes. The 110 registered runs need about 8 laptop-hours (130 runs, about 9); the judge fit, selection, certificate, weights and dry run add about 15 minutes. Loading the full failure pool raises memory by roughly 1.5 GB.

## 9. Contributions

Contributors are listed on the OSF registration under the names they publish under; ORCID is optional. Paper authorship, CRediT roles, author order, interests and credit providers are settled at paper submission, not in this registration.

## 10. Registration checklist

| Acceptance check | Pass | Fail |
|---|---|---|
| Form fields and the registration core PDF verified on submission day; the core and this protocol agree; both committed under `docs/registration/` at the registered tag | ☐ | ☐ |
| The registered `trajectory-lineage` commit and the Experiment 2 pipeline commit recorded in the form | ☐ | ☐ |
| Before the Experiment 2 runs and the Experiment 3 confirmation (not before registration): both freeze manifests signed (the analyzers at the registered commit already implement the registered rules, and the dry runs are rerun) | ☐ | ☐ |
| Prior exposure disclosed (§2.3, §3.8); no external reviewer receives a protected effect | ☐ | ☐ |
| Units, estimands, gates, fallbacks, multiplicity, and margins executable | ☐ | ☐ |
| Repositories and configurations identified; tags planned per experiment | ☐ | ☐ |
| Missingness, reruns, stopping, and budget rules enforceable | ☐ | ☐ |
| Dry runs on record for Experiments 2–3 (repository `results/`); Experiment 1 dry run due with the November 20 amendment | ☐ | ☐ |
| Every planned endpoint has a reporting destination | ☐ | ☐ |
| Embargo active until peer review ends; submission anonymized and link-free | ☐ | ☐ |
| Every OSF contributor has agreed to be listed (ORCID optional) | ☐ | ☐ |
| Freeze manifests carry lockfile hash, runtime hash and hardware class | ☐ | ☐ |
