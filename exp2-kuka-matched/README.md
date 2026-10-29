# Experiment 2 — transition-matched Kuka control: registered implementation

*Part of the Seen and Unseen study; kept in this repository next to Experiment 3. It uses the Experiment 2 owner's public pipeline (https://github.com/subhashjprasad/robot-world-model-optimism-research) unchanged.*

**Real-data results (October 28):** `results_real_oct28/` holds the manifest summary, the selection and its certificate (110 runs, 673 checks, 0 failures), the evaluation support and the dry run, from a Kuka cache rebuilt on Colab with the owner's extractor. The V arms in that run used the earlier fixed-addition rule; the registered rule now matches each V arm's total to 31,600. The owner's rerun on his own cache is the certificate of record.

Code that turns Protocol §2 of *Seen and Unseen* into frozen, hash-bound artefacts. It sits next to the owner's pipeline (`robot-world-model-optimism-research`) and uses the owner's `wm.py` (model, trainer, rollout, judge) unchanged. Everything here runs on episode metadata or on per-episode scores; only `run_registered.py` trains.

| Step | Module | What it fixes before any outcome is seen |
|---|---|---|
| 1 | `registered/manifest.py` | Episode manifest (id, shard, length, outcome, split) from `data/kuka/shard_*.npz`, hashed. Reads only the small arrays; seconds, no features in memory. |
| 2 | `registered/design.py` | Seeded samplers for designs O, A, B, V and the **feasibility certificate**: exact 31,600 transitions and exact φ for A; equal normalized outcome-conditional length distributions and identical total histograms for B; shared base and arm totals matched to A's 31,600 for V. 110 runs (130 if B falls back). |
| 3 | `registered/weights.py` | Evaluation support (exact-length bins with ≥ 20 evaluation episodes per outcome) and overlap weights a_i; Σa = 1, weighted truth = π, ESS reported. |
| 4 | `registered/analyze.py` | The decision contract: paired t over seeds 100–109, one-sided 95% lower bound > 3 for support, 90% CI inside (−3, 3) for absence, Holm over the ten secondary p-values, worst-case bounds and inconclusive decisions when a run is missing, outcome-to-interpretation table. |
| 5 | `registered/dryrun.py` | Synthetic per-episode scores for every run under declared truths → the real validator and analyzer → operating characteristics with automated checks. |
| 6 | `registered/freeze.py` | Freeze manifest binding config, manifest, selection, support and code by SHA-256, with named dated sign-offs. Driver and analyzer refuse to run if anything differs. |
| 7 | `registered/run_registered.py` | Training driver: frozen selection in, per-episode scores out; frozen normalization and judge (hard AUROC ≥ 0.75 gate); refuses stale artefacts; records infrastructure failures and retries once with the same seed. |

Tests: `tests/` (22 tests, including an end-to-end driver smoke test on a tiny synthetic feature cache). Run from this folder: `python -m pytest -q tests`. The large synthetic stand-in files are not committed; `python -m registered.manifest synthetic --audit-jsonl <owner repo>/results/episodes/kuka.jsonl --out results/synthetic_manifest.jsonl` regenerates them.

## What the owner runs (about 10 minutes of wall time before training)

```bash
# 0. environment: the owner's uv project already has numpy, scipy, scikit-learn, torch
cd robot-world-model-optimism-research
cp -r /path/to/exp2-kuka-matched/registered experiment/registered   # or keep it beside the repo and set PYTHONPATH
export PYTHONPATH=/path/to/exp2-kuka-matched

# 1. manifest from the real cache (metadata only; seconds)
python -m registered.manifest build --npz-dir data/kuka --out registered_out/manifest.jsonl

# 2. selection + certificate for all seeds and designs (seconds)
python -m registered.design --manifest registered_out/manifest.jsonl --out registered_out/selection.json
#    -> selection.json (ids inline), selection.certificate.json, selection.md (table)
#    The certificate states whether Design B is feasible at 31,600 or has fallen back by the registered rule.

# 3. evaluation support and weights (seconds)
python -m registered.weights --manifest registered_out/manifest.jsonl --out registered_out/support.json

# 4. dry run of the analyzer on this selection (about 3 minutes with 10 workers)
python -m registered.dryrun --manifest registered_out/manifest.jsonl --selection registered_out/selection.json \
    --support registered_out/support.json --datasets 500 --out registered_out/dryrun.json

# 5. freeze (internal signers: paper lead + stats check; the owner may add a line). Then tag the commit in git.
python -m registered.freeze create --manifest registered_out/manifest.jsonl --selection registered_out/selection.json \
    --support registered_out/support.json --approve "paper_lead:Full Name:YYYY-MM-DD" \
    --approve "stats_check:Full Name:YYYY-MM-DD" --registration "osf.io/<id>" --out registered_out/freeze.json

# 6. registered training (after Oct 30), about 110 x 4 minutes on an M-series laptop at 2,000 steps
python -m registered.run_registered --owner-repo . --npz-dir data/kuka --manifest registered_out/manifest.jsonl \
    --selection registered_out/selection.json --support registered_out/support.json --freeze registered_out/freeze.json \
    --out registered_out/runs
#    rerun with --retry-failed once if a run file says infrastructure_failure

# 7. analysis (Nov 9 target)
python -m registered.analyze --runs registered_out/runs --selection registered_out/selection.json --support registered_out/support.json \
    --manifest registered_out/manifest.jsonl --judge registered_out/runs/judge.json --freeze registered_out/freeze.json --out registered_out/decision.json
```

Attach to the registration: `selection.certificate.json`, `selection.md`, `support.md`, `dryrun.md`, `freeze.json`, and the SHA-256 of `selection.json`.

## What is already verified here, and what is not

Verified on a **synthetic stand-in manifest** (`results/synthetic_manifest.jsonl`, header `kind: SYNTHETIC`) that has the real pool sizes from the owner's log (3,989 train successes, 2,800 test episodes, 996 test successes, π = 0.3557) and the audited length-by-outcome shape (`results/episodes/kuka.jsonl`, 320 episodes); the train-failure count (6,900) is inferred from the README's 13.7k total and is an assumption:

- the samplers hit every constraint for all 110 runs (`results/selection_synthetic.certificate.json`: 673 checks, 0 failures);
- Design B is feasible at 31,600 on this stand-in (maximum feasible 36,560); with the owner's old 3,000-failure memory cap it is not, and the registered fallback to 26,000 with A comparators (130 runs) engages;
- the analyzer's operating characteristics match the analytic power (0.934 vs 0.927 at SD 1.88; 0.626 vs 0.618 at SD 3) and the boundary error rates stay at nominal (`results/dryrun_synthetic.md`).

Not verified until the owner runs steps 1–4 on the real cache: the real certificate (B's feasibility and budget), the real support table and ESS, and the dry run on the real selection. The stand-in must never be used to train or to sign a freeze; `design.py` refuses it without `--allow-synthetic`, `freeze.py` refuses it outright.

## Design notes

- **Pairing.** One permutation per outcome pool per seed; every arm of that seed takes prefixes and repairs by swaps. Different arms of a seed therefore share most episodes, which is what the paired t analysis assumes.
- **Design A repair.** Among episode totals that are multiples of 20 (so φN is an integer for every φ), the prefix total closest to 31,600 is repaired by same-outcome swaps; each swap strictly reduces the gap and never changes φ. The synthetic selection needed at most 7 swaps per arm (8 in the capped variant).
- **Design B shape.** Per-length caps k_l ≤ min(S_l // 18, F_l // 13) make both arms feasible from the same histogram. The common histogram is the exact integer least-squares solution closest to the pooled availability shape with Σ k_l (l − 1) = 1,580 (dynamic programme). Conditional distributions are then identical by construction (18k_l : 2k_l and 7k_l : 13k_l), and the certificate re-checks hist_S · n_F = hist_F · n_S at every length.
- **Fallback.** If 31,600 is infeasible for B, budget_B = largest multiple of 1,000 ≤ 0.9 × maximum feasible, and Design A comparators at budget_B are added for φ = .10 and .65; Δ_{A−B} then uses those comparators.
- **Difference contrasts.** O − A and A − B have about √2 the paired SD of a single contrast and carry Holm's penalty; the dry run shows they are decisive only for attenuations near 10 points at SD 1.88. This is a property of the design, stated so that "inconclusive" there is not read as "no attenuation".
- **Judge.** Fitted once on 2,500 + 2,500 real train trajectories (seeded 99), pickled and hashed; real AUROC, FPR, FNR, Brier and a reliability table are written to `judge.json`; AUROC < 0.75 stops the driver.
- **Not changed in `wm.py`.** The count of valid multistep loss terms is not exposed by `wm.train_wm`; the driver records distinct transitions and sampled starts (steps × 256) and notes the missing counter. Adding it is a two-line change in the owner's trainer.
- **Memory.** The driver loads the full eligible train pool plus the test set once (about 13.7k episodes, roughly 2.5 GB of float16 features; the owner's earlier run peaked near 4 GB with 9.8k episodes).
