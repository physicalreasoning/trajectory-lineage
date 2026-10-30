# Colab notebooks

Notebooks for the GPU steps of the Seen and Unseen study. Open one in Google Colab, choose a GPU runtime (A100 or L4) with High-RAM, then Runtime -> Run all. Each notebook clones this repository, mounts Google Drive for resumable checkpoints, and downloads a results zip. After a disconnect, Run all again: finished work is skipped.

| Notebook | What it runs | Status |
|---|---|---|
| `run_exp3_pilots_a100.ipynb` | Experiment 3 effect-blind pilots: architecture axis (frozen DINOv2-small or V-JEPA 2-L encoder + latent transformer) and dose axis (16 or 256 reused sources), three blocks in parallel | Run on Oct 28; reports in `results/` |
| `run_all_exp3_then_exp2.ipynb` | Part A: Experiment 3 capacity pilot. Part B: Experiment 2 Kuka feature cache on Drive, then manifest, certificate, support and dry run (`exp2-kuka-matched/`). Its training cells stay locked until a signed freeze exists | Run on Oct 28; Exp 2 results in `exp2-kuka-matched/results_real_oct28/` |
| `prepare_freeze_drafts.ipynb` | Writes the unsigned freeze manifests: Exp 3 (five configurations, analyzer, code commit, pilot seeds, and the environment recorded on the Colab A100) and Exp 2 (recertified selection on the Drive cache under the current rules, support, code, and the environment) | Before the freeze; signers then run `sign` |
| `LOCKED_exp3_confirmation.ipynb` | Experiment 3 confirmation: the five registered configurations from the freeze tag, with the signed freeze manifest; raw records stay sealed on Drive | After registration, freeze and tag only |

`build_notebooks.py` regenerates the notebooks. It refuses to write a notebook that fails its static checks (cell magics on the first line, Python syntax, shell placeholders defined earlier, `%%writefile` folders created earlier, embedded files compile).
