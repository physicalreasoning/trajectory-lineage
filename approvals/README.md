# Approvals

Freeze manifests and approval records. This is the only folder that may change after a freeze commit: the freeze check accepts a checkout that differs from the frozen commit only here, so signatures can be added without breaking the binding.

| File | What it is | Status |
|---|---|---|
| `exp3_freeze_manifest.json` | Experiment 3 confirmation: binds the five 2D configurations, the analyzer, code commit `ccd871e`, every pilot seed, and the environment recorded on a Colab A100 (lockfile hash, exact package versions and their hash, hardware class) | Draft, unsigned |
| `exp2/exp2_freeze_manifest.json` | Experiment 2: binds the registered config, the episode manifest, the selection (110 runs, recertified under the current V rule), the evaluation support, the code, and the environment (the owner's `uv.lock`, exact package versions and their hash, hardware class A100) | Draft, unsigned |
| `exp2/selection.json`, `support.json`, `manifest.jsonl` | The files the Exp 2 freeze binds by hash (episode list: 13,678 Kuka episodes, no folder path in the header) | Recertified Oct 30 |
| `exp2/selection.certificate.json`, `*.md` | Certificate (673 checks, 0 failures), selection and support summaries, and the dry run (12/12 checks; same 110 runs as the Oct 30 02:29 run) | Evidence |
| `provisional_freeze.json` | 1D confirmation thresholds, frozen by the workstream owner on Oct 19 before unblinding. The 1D confirmation stays sealed | Provisional |

The runners and analyzers refuse an unsigned manifest. To sign (each of the two internal signers, once):

```bash
python scripts/freeze.py sign --manifest approvals/exp3_freeze_manifest.json --role paper_lead --name "Full Name" --date YYYY-MM-DD
python scripts/freeze.py sign --manifest approvals/exp3_freeze_manifest.json --role stats_check --name "Full Name" --date YYYY-MM-DD
python scripts/freeze.py check --manifest approvals/exp3_freeze_manifest.json   # "ok" for all five configurations
```

Experiment 2 (from `exp2-kuka-matched/`):

```bash
PYTHONPATH=. python -m registered.freeze sign --freeze ../approvals/exp2/exp2_freeze_manifest.json --role paper_lead --name "Full Name" --date YYYY-MM-DD
PYTHONPATH=. python -m registered.freeze sign --freeze ../approvals/exp2/exp2_freeze_manifest.json --role stats_check --name "Full Name" --date YYYY-MM-DD
PYTHONPATH=. python -m registered.freeze verify --freeze ../approvals/exp2/exp2_freeze_manifest.json --selection ../approvals/exp2/selection.json --support ../approvals/exp2/support.json --require-signoff
```
