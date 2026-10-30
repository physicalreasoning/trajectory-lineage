# Approvals

Freeze manifests and approval records. This is the only folder that may change after a freeze commit: the freeze check accepts a checkout that differs from the frozen commit only here, so signatures can be added without breaking the binding.

| File | What it is | Status |
|---|---|---|
| `exp3_freeze_manifest.json` | Experiment 3 confirmation: binds the five 2D configurations, the analyzer, code commit `ccd871e`, every pilot seed, and the environment recorded on a Colab A100 (lockfile hash, exact package versions and their hash, hardware class) | Draft, unsigned |
| `exp2/` | Experiment 2 freeze manifest and the files it binds (recertified selection, support, episode manifest) | To be added |
| `provisional_freeze.json` | 1D confirmation thresholds, frozen by the workstream owner on Oct 19 before unblinding. The 1D confirmation stays sealed | Provisional |

The runners and analyzers refuse an unsigned manifest. To sign (each of the two internal signers, once):

```bash
python scripts/freeze.py sign --manifest approvals/exp3_freeze_manifest.json --role paper_lead --name "Full Name" --date YYYY-MM-DD
python scripts/freeze.py sign --manifest approvals/exp3_freeze_manifest.json --role stats_check --name "Full Name" --date YYYY-MM-DD
python scripts/freeze.py check --manifest approvals/exp3_freeze_manifest.json   # "ok" for all five configurations
```
