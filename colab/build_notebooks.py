"""Generate the Colab notebooks for the Seen and Unseen project. Embeds the exp2-kuka-matched package so nothing needs a push."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
PKG = HERE.parent / "exp2-kuka-matched"
FILES = ["configs/exp2_registered.json", "registered/__init__.py", "registered/manifest.py", "registered/design.py",
         "registered/weights.py", "registered/analyze.py", "registered/dryrun.py", "registered/freeze.py", "registered/run_registered.py"]


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def exp2_notebook() -> dict:
    cells = [
        md("""# Seen and Unseen — Experiment 2 on Colab: Kuka feature cache, certificate, support, dry run

Rebuilds the Kuka (QT-Opt) feature cache with the owner's own `extract_kuka.py` (DINOv2-small, 17 tokens per frame), then runs the registered Exp 2 preparation from `exp2-kuka-matched`:
manifest → selection + feasibility certificate (110 or 130 runs) → evaluation support and weights → dry run of the frozen analyzer.

**Runtime:** GPU (any). Extraction streams about 12 GB from the public Open X-Embodiment bucket and encodes about 200k frames: roughly 40–90 min on a T4/L4, less on an A100. The cache (about 4.3 GB) is stored on Google Drive so it is reused for the registered training later.

**Send back:** `exp2_prep_results.zip` (downloaded by cell 7). It holds the manifest summary, `selection.certificate.json`, `selection.md`, `support.md`, `support.json`, `dryrun.md`, `dryrun.json`, and the full `selection.json`.

**Do not run cell 8** (training) before the registration is submitted and the freeze manifest is signed. The driver refuses to start without the signed freeze."""),
        code("""#@title 1. Configuration
USE_DRIVE = True        #@param {type:"boolean"}   # keep the cache and outputs on Google Drive (recommended)
DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp2"  #@param {type:"string"}
SHARDS = "0:1024:3"     #@param {type:"string"}    # the owner's shard selection (345 shards)
PER_SHARD = 40          #@param {type:"integer"}
WORKERS = 6             #@param {type:"integer"}
DRYRUN_DATASETS = 300   #@param {type:"integer"}   # 500 on a machine with many cores; Colab has 2
import torch, os
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)")"""),
        code("""#@title 2. Owner's pipeline and dependencies
%cd /content
import os
if not os.path.exists("rwmo"):
    !git clone -q https://github.com/subhashjprasad/robot-world-model-optimism-research.git rwmo
%cd /content/rwmo
!git rev-parse --short HEAD
!pip install -q tfrecord crc32c protobuf requests "transformers>=5.0" accelerate pillow scikit-learn scipy pandas matplotlib
# the owner's scripts pick MPS or CPU; prefer CUDA on Colab (no other change)
!sed -i 's/device = "mps" if torch.backends.mps.is_available() else "cpu"/device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")/' experiment/extract_kuka.py
!sed -i 's/DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"/DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")/' experiment/wm.py
!grep -n "cuda" experiment/extract_kuka.py experiment/wm.py | head -3"""),
        code("""#@title 3. Storage (Drive) for the cache and outputs
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(f"{DRIVE_DIR}/kuka", exist_ok=True); os.makedirs(f"{DRIVE_DIR}/registered_out", exist_ok=True)
    !rm -rf /content/rwmo/data/kuka && ln -s {DRIVE_DIR}/kuka /content/rwmo/data/kuka
    OUT = f"{DRIVE_DIR}/registered_out"
else:
    os.makedirs("/content/rwmo/data/kuka", exist_ok=True); OUT = "/content/registered_out"; os.makedirs(OUT, exist_ok=True)
print("cache:", os.path.realpath("/content/rwmo/data/kuka"), "| outputs:", OUT)
!ls /content/rwmo/data/kuka | wc -l"""),
        code("""#@title 4. Feature extraction (resumable; skips shards already in the cache)
%cd /content/rwmo
!PYTHONPATH=experiment:audit python experiment/extract_kuka.py --shards {SHARDS} --per-shard {PER_SHARD} --workers {WORKERS} 2>&1 | grep -v -i "warn" | tail -n 25
!ls data/kuka | wc -l; du -sh data/kuka"""),
    ]
    cells.append(code("#@title 4b. Create the package folders (the next cells write the exp2-kuka-matched files)\nimport os\nos.makedirs('/content/exp2-kuka-matched/configs', exist_ok=True)\nos.makedirs('/content/exp2-kuka-matched/registered', exist_ok=True)\nprint('ok')"))
    for f in FILES:
        cells.append(code(f"%%writefile /content/exp2-kuka-matched/{f}\n" + (PKG / f).read_text()))
    cells += [
        code("""#@title 5. Manifest → selection + certificate → support → dry run (CPU, 5–15 min; the dry run is the slow part)
%cd /content/exp2-kuka-matched
import os
!PYTHONPATH=. python -m registered.manifest build --npz-dir /content/rwmo/data/kuka --out {OUT}/manifest.jsonl
!PYTHONPATH=. python -m registered.design --manifest {OUT}/manifest.jsonl --out {OUT}/selection.json | head -n 60
!PYTHONPATH=. python -m registered.weights --manifest {OUT}/manifest.jsonl --out {OUT}/support.json | tail -n 30
!PYTHONPATH=. python -m registered.dryrun --manifest {OUT}/manifest.jsonl --selection {OUT}/selection.json --support {OUT}/support.json --datasets {DRYRUN_DATASETS} --workers {os.cpu_count()} --out {OUT}/dryrun.json | tail -n 40"""),
        code("""#@title 6. Certificate summary
import json
c = json.load(open(f"{OUT}/selection.certificate.json"))
print(json.dumps({k: v for k, v in c.items() if k not in ("checks", "failed")}, indent=1))
print("failed checks:", c["failed"][:5])
print(open(f"{OUT}/selection.md").read())"""),
        code("""#@title 7. Package results for download
!cd {OUT} && zip -q -j /content/exp2_prep_results.zip manifest.summary.json selection.json selection.certificate.json selection.md support.json support.md dryrun.json dryrun.md
!ls -la /content/exp2_prep_results.zip
from google.colab import files
files.download("/content/exp2_prep_results.zip")"""),
        md("""---
## 8. Registered training — LOCKED until the freeze is signed and the registration is submitted
Upload the signed `freeze.json` to `OUT` first. The driver verifies config, selection, support and code hashes and the sign-offs, then trains the 110 (or 130) runs with the owner's `wm.py` (about 1–2 min per run on an A100 at 2,000 steps; the pipeline resumes if the runtime restarts). Then run the analyzer."""),
        code("""#@title 8a. Train (requires {OUT}/freeze.json)
import os
assert os.path.exists(f"{OUT}/freeze.json"), "no signed freeze manifest in OUT; do not train before registration"
%cd /content/exp2-kuka-matched
!PYTHONPATH=. python -m registered.run_registered --owner-repo /content/rwmo --npz-dir /content/rwmo/data/kuka --manifest {OUT}/manifest.jsonl --selection {OUT}/selection.json --support {OUT}/support.json --freeze {OUT}/freeze.json --out {OUT}/runs 2>&1 | grep -v -i "warn" | tail -n 150"""),
        code("""#@title 8b. Analyze (after all runs)
%cd /content/exp2-kuka-matched
!PYTHONPATH=. python -m registered.analyze --runs {OUT}/runs --selection {OUT}/selection.json --support {OUT}/support.json --manifest {OUT}/manifest.jsonl --judge {OUT}/runs/judge.json --freeze {OUT}/freeze.json --out {OUT}/decision.json
!cd {OUT} && zip -q -j /content/exp2_decision.zip decision.json decision.md runs/judge.json && ls -la /content/exp2_decision.zip
from google.colab import files
files.download("/content/exp2_decision.zip")"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


TL_COMMIT = "6b1031b10ac5ec757e8bd8e7f1a2b47f0ddcbfef"  # public main after pull request #19 (merge commit)


def exp3_capacity_pilot_notebook() -> dict:
    cells = [
        md("""# Seen and Unseen — Experiment 3 on Colab: effect-blind capacity pilot (2D reuse study)

Runs the registered 2D trajectory-reuse design at three predictor sizes (ConvLSTM hidden channels **16, 64, 256**, same training protocol) to release, per size, only what planning needs: contrast **dispersions**, competence counts, parameter counts and seconds per block, plus the pooled coarse positive-control flag. **No contrast location is written anywhere**: the runner keeps block records in memory and writes the blinded report only, exactly as the Oct 23 2D pilot did.

Two steps:
1. **Effect-blind base tuning** (`tune_base_2d.py`): base models only, error ratio against constant velocity at 300/600/1,200 steps for each size. No arms are built or scored.
2. **Capacity pilot** (`exposure_experiment.py` on `experiments/exposure2d_capacity_pilot_v1.json`): 72 blocks, 24 per size, interleaved.

**Resumable.** Every finished block is appended to a *blinded* checkpoint on Google Drive (contrasts shifted by a random key that lives in a separate `DO_NOT_OPEN` folder; no component errors are stored). If the session disconnects, rerun the notebook: finished blocks are skipped. One progress line per block is printed live.

**Runtime:** about 1.5–2.5 min per block on an A100/L4 after the parallel-sampling fix (the Oct 27 run took 4.8 min per block single-threaded), so 2–3 h for 72 blocks. **Send back:** `exp3_capacity_pilot_results.zip` (cell 7)."""),
        code("""#@title 1. Configuration (Part A)
TUNE_BLOCKS = 4                 #@param {type:"integer"}
TUNE_CHECKPOINTS = "300,600,1200"  #@param {type:"string"}
HIDDEN = "16,64,256"            #@param {type:"string"}
RUN_TUNING = True               #@param {type:"boolean"}
RUN_PILOT = True                #@param {type:"boolean"}
EXP3_DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp3/capacity_pilot_v1"  #@param {type:"string"}
import os, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)", "| CPUs:", os.cpu_count())
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN", exist_ok=True)
EXP3_CKPT = f"{EXP3_DRIVE_DIR}/blinded_checkpoint.jsonl"
EXP3_KEY = f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN/blinding_key.json"
print("checkpoint:", EXP3_CKPT, "| blocks already recorded:", sum(1 for _ in open(EXP3_CKPT)) if os.path.exists(EXP3_CKPT) else 0)"""),
        code(f"""#@title 2. Code: public trajectory-lineage at {TL_COMMIT} (includes pull request #19) (local branch, not yet pushed)
%cd /content
import os
if not os.path.exists("trajectory-lineage"):
    !git clone -q https://github.com/physicalreasoning/trajectory-lineage.git
%cd /content/trajectory-lineage
!git checkout -q {TL_COMMIT} && git rev-parse --short HEAD
!pip install -q numpy h5py pytest
os.environ["PYTHONPATH"] = "/content/trajectory-lineage/src"
print("ok")"""),
        code("""#@title 3. Run the capacity tests (CPU, about 1 min)
%cd /content/trajectory-lineage
!git log --oneline -1
!PYTHONPATH=src python -m pytest -q tests/test_capacity.py tests/test_analyze_2d.py tests/test_exp3_2d_path.py -p no:cacheprovider 2>&1 | tail -n 3"""),
        code("""#@title 4. Effect-blind base tuning per capacity (base models only; no arms)
%cd /content/trajectory-lineage
if RUN_TUNING:
    !PYTHONPATH=src python scripts/tune_base_2d.py --config experiments/exposure2d_capacity_pilot_v1.json --out results/exposure2d_capacity_base_tuning.json --blocks {TUNE_BLOCKS} --checkpoints {TUNE_CHECKPOINTS} --hidden {HIDDEN} 2>&1 | tail -n 40
else:
    print("skipped")"""),
        code("""#@title 5. Capacity pilot: 72 effect-blinded blocks (24 per size), resumable from the Drive checkpoint; one line per block
%cd /content/trajectory-lineage
if RUN_PILOT:
    !PYTHONPATH=src python scripts/exposure_experiment.py --config experiments/exposure2d_capacity_pilot_v1.json --out results/exposure2d_capacity_pilot_v1_blinded.json --device cuda --blinded-checkpoint {EXP3_CKPT} --blinding-key {EXP3_KEY} 2>&1 | grep -E "^block|^resuming|Error|error"
    !cp results/exposure2d_capacity_pilot_v1_blinded.json {EXP3_DRIVE_DIR}/
else:
    print("skipped")"""),
        code("""#@title 6. Blinded summary (dispersions, competence, runtime per size; pooled positive-control flag)
import json
r = json.load(open("/content/trajectory-lineage/results/exposure2d_capacity_pilot_v1_blinded.json"))
print({k: r[k] for k in ("blocks_scheduled", "blocks_completed", "blocks_base_competent", "positive_control_passed") if k in r})
for level, v in r["by_capacity"].items():
    disp = {name: round(d["standard_deviation_upper_limit"], 5) for name, d in v["dispersion"].items()}
    print(f"hidden {level}: params {v['model_parameters']}, completed {v['blocks_completed']}, competent {v['blocks_base_competent']}, s/block {None if v['seconds_per_block_mean'] is None else round(v['seconds_per_block_mean'])}, SD upper limits {disp}")
print("infrastructure failures:", len(r["infrastructure_failures"]), "| algorithmic failures:", len(r["algorithmic_failures"]))"""),
        code("""#@title 7. Package results for download
%cd /content/trajectory-lineage
!git -C /content/trajectory-lineage rev-parse HEAD > results/code_commit.txt
!zip -q -j /content/exp3_capacity_pilot_results.zip results/exposure2d_capacity_pilot_v1_blinded.json results/exposure2d_capacity_base_tuning.json results/code_commit.txt
!ls -la /content/exp3_capacity_pilot_results.zip
from google.colab import files
files.download("/content/exp3_capacity_pilot_results.zip")"""),
        md("""### Notes
- The blinded report contains no effect locations; the only non-dispersion output is the pooled coarse positive-control flag, as in the Oct 23 pilot. Do not add `--unblind` or `--checkpoint`; the runner refuses both without a signed freeze manifest.
- The Drive checkpoint holds blinded records only (contrasts shifted by the key in `DO_NOT_OPEN/`; dispersions are unaffected). Never open the key folder. If the runtime disconnects, rerun from cell 1: finished blocks are skipped."""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def combined_notebook() -> dict:
    """One notebook, one configuration cell. Part A's pilot runs in the background while Part B runs."""

    GUARD = 'assert "RUN_PILOT" in globals(), "Run the configuration cell at the top first (after any restart: Runtime -> Run all)"\n'
    cells = [
        md("""# Seen and Unseen — Colab run: Exp 3 capacity pilot (background) + Exp 2 Kuka cache and certificate

**One configuration cell at the top. After any disconnect or restart: Runtime → Run all.** Never run a middle cell alone after a restart; every step is resumable, so Run all only redoes what is missing.

| Part | What | Where it runs | Download |
|---|---|---|---|
| A | Exp 3 effect-blind capacity pilot: ConvLSTM 16/64/256 channels, 72 blinded blocks, resumable from a blinded checkpoint on Drive | background process, started in cell A3 | `exp3_capacity_pilot_results.zip` (cell A5) |
| B | Exp 2: Kuka feature cache on Drive, then manifest → certificate → support → dry run | foreground, while A runs | `exp2_prep_results.zip` (cell B5) |

GPU runtime (A100 or L4), High-RAM on. Expected: Part B about 1–1.5 h; Part A about 2–3 h in the background; both zips download when ready. The Part B training cells at the end stay locked until a signed freeze exists."""),
        code("""#@title 0. Configuration (run first; the only cell that defines settings)
# --- Part A: Exp 3 capacity pilot
TUNE_BLOCKS = 4                     #@param {type:"integer"}
TUNE_CHECKPOINTS = "300,600,1200"   #@param {type:"string"}
HIDDEN = "16,64,256"                #@param {type:"string"}
RUN_TUNING = True                   #@param {type:"boolean"}
RUN_PILOT = True                    #@param {type:"boolean"}
EXP3_DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp3/capacity_pilot_v1"  #@param {type:"string"}
# --- Part B: Exp 2 cache + certificate
SHARDS = "0:1024:3"                 #@param {type:"string"}
PER_SHARD = 40                      #@param {type:"integer"}
WORKERS = 6                         #@param {type:"integer"}
DRYRUN_DATASETS = 300               #@param {type:"integer"}
EXP2_DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp2"  #@param {type:"string"}
USE_DRIVE = True

import os, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)", "| CPUs:", os.cpu_count())
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN", exist_ok=True)
os.makedirs(f"{EXP2_DRIVE_DIR}/kuka", exist_ok=True); os.makedirs(f"{EXP2_DRIVE_DIR}/registered_out", exist_ok=True)
EXP3_CKPT = f"{EXP3_DRIVE_DIR}/blinded_checkpoint.jsonl"
EXP3_KEY = f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN/blinding_key.json"
EXP3_LOG = f"{EXP3_DRIVE_DIR}/pilot.log"
OUT = f"{EXP2_DRIVE_DIR}/registered_out"
print("Exp 3 checkpoint:", EXP3_CKPT, "| blocks already recorded:", sum(1 for _ in open(EXP3_CKPT)) if os.path.exists(EXP3_CKPT) else 0)
print("Exp 2 outputs:", OUT)"""),
        md("## Part A — Experiment 3: effect-blind capacity pilot (background)"),
        code(GUARD + f"""#@title A1. Code: public trajectory-lineage at {TL_COMMIT} (includes pull request #19)
%cd /content
import os
if not os.path.exists("trajectory-lineage"):
    !git clone -q https://github.com/physicalreasoning/trajectory-lineage.git
%cd /content/trajectory-lineage
!git checkout -q {TL_COMMIT} && git rev-parse --short HEAD
!pip install -q numpy h5py pytest
print("ok")"""),
        code(GUARD + """#@title A2. Run the capacity tests (about 1 min), effect-blind base tuning (base models only)
%cd /content/trajectory-lineage
!git log --oneline -1
!PYTHONPATH=src python -m pytest -q tests/test_capacity.py tests/test_analyze_2d.py tests/test_exp3_2d_path.py -p no:cacheprovider 2>&1 | tail -n 2
if RUN_TUNING and not os.path.exists(f"{EXP3_DRIVE_DIR}/exposure2d_capacity_base_tuning.json"):
    !PYTHONPATH=src python scripts/tune_base_2d.py --config experiments/exposure2d_capacity_pilot_v1.json --out results/exposure2d_capacity_base_tuning.json --blocks {TUNE_BLOCKS} --checkpoints {TUNE_CHECKPOINTS} --hidden {HIDDEN} 2>&1 | tail -n 12
    !cp results/exposure2d_capacity_base_tuning.json {EXP3_DRIVE_DIR}/
else:
    print("tuning skipped (disabled or already on Drive)")"""),
        code(GUARD + """#@title A3. Launch the 72-block pilot in the background (resumes from the Drive checkpoint)
import subprocess, os, time
if RUN_PILOT:
    env = dict(os.environ, PYTHONPATH="/content/trajectory-lineage/src", TRAJECTORY_LINEAGE_WORKERS=str(max(2, (os.cpu_count() or 4) - 2)))
    cmd = ["python", "scripts/exposure_experiment.py", "--config", "experiments/exposure2d_capacity_pilot_v1.json",
           "--out", "results/exposure2d_capacity_pilot_v1_blinded.json", "--device", "cuda",
           "--blinded-checkpoint", EXP3_CKPT, "--blinding-key", EXP3_KEY]
    PILOT = subprocess.Popen(cmd, cwd="/content/trajectory-lineage", env=env, stdout=open(EXP3_LOG, "a"), stderr=subprocess.STDOUT)
    time.sleep(20)
    print("pilot PID", PILOT.pid, "| alive:", PILOT.poll() is None, "| log:", EXP3_LOG)
    !tail -n 3 {EXP3_LOG}
else:
    print("pilot disabled")"""),
        md("## Part B — Experiment 2: Kuka cache, manifest, certificate, support, dry run (runs while Part A works)"),
        code(GUARD + """#@title B1. Owner's pipeline and dependencies
%cd /content
import os
if not os.path.exists("rwmo"):
    !git clone -q https://github.com/subhashjprasad/robot-world-model-optimism-research.git rwmo
%cd /content/rwmo
!git rev-parse --short HEAD
!pip install -q tfrecord crc32c protobuf requests "transformers>=5.0" accelerate pillow scikit-learn scipy pandas matplotlib
!sed -i 's/device = "mps" if torch.backends.mps.is_available() else "cpu"/device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")/' experiment/extract_kuka.py
!sed -i 's/DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"/DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")/' experiment/wm.py
!rm -rf /content/rwmo/data/kuka && ln -s {EXP2_DRIVE_DIR}/kuka /content/rwmo/data/kuka
print("cache:", os.path.realpath("/content/rwmo/data/kuka"), "| shards present:", len(os.listdir("/content/rwmo/data/kuka")))"""),
        code(GUARD + """#@title B2. Feature extraction (resumable; skips shards already in the cache)
%cd /content/rwmo
!PYTHONPATH=experiment:audit python experiment/extract_kuka.py --shards {SHARDS} --per-shard {PER_SHARD} --workers {WORKERS} 2>&1 | grep -v -i "warn" | tail -n 25
!ls data/kuka | wc -l; du -sh data/kuka"""),
        code("#@title B3. Create the package folders (the next cells write the exp2-kuka-matched files)\nimport os\nos.makedirs('/content/exp2-kuka-matched/configs', exist_ok=True)\nos.makedirs('/content/exp2-kuka-matched/registered', exist_ok=True)\nprint('ok')"),
    ]
    for f in FILES:
        cells.append(code(f"%%writefile /content/exp2-kuka-matched/{f}\n" + (PKG / f).read_text()))
    cells += [
        code(GUARD + """#@title B4. Manifest → selection + certificate → support → dry run (CPU, 5–15 min)
%cd /content/exp2-kuka-matched
import os
!PYTHONPATH=. python -m registered.manifest build --npz-dir /content/rwmo/data/kuka --out {OUT}/manifest.jsonl
!PYTHONPATH=. python -m registered.design --manifest {OUT}/manifest.jsonl --out {OUT}/selection.json | head -n 60
!PYTHONPATH=. python -m registered.weights --manifest {OUT}/manifest.jsonl --out {OUT}/support.json | tail -n 30
!PYTHONPATH=. python -m registered.dryrun --manifest {OUT}/manifest.jsonl --selection {OUT}/selection.json --support {OUT}/support.json --datasets {DRYRUN_DATASETS} --workers 2 --out {OUT}/dryrun.json | tail -n 40
import json
c = json.load(open(f"{OUT}/selection.certificate.json"))
print(json.dumps({k: v for k, v in c.items() if k not in ("checks", "failed")}, indent=1)); print("failed checks:", c["failed"][:5])"""),
        code(GUARD + """#@title B5. Package Exp 2 results for download
!cd {OUT} && zip -q -j /content/exp2_prep_results.zip manifest.summary.json selection.json selection.certificate.json selection.md support.json support.md dryrun.json dryrun.md
!ls -la /content/exp2_prep_results.zip
from google.colab import files
files.download("/content/exp2_prep_results.zip")"""),
        md("## Part A — finish: wait for the pilot, summarize, download"),
        code(GUARD + """#@title A4. Wait for the pilot (prints progress lines as they appear)
import subprocess, time, os
def pilot_running():
    return subprocess.run(["pgrep", "-f", "exposure_experiment.py --config experiments/exposure2d_capacity_pilot_v1.json"], capture_output=True).returncode == 0
seen = 0
while pilot_running():
    lines = open(EXP3_LOG).read().splitlines() if os.path.exists(EXP3_LOG) else []
    for line in lines[seen:]:
        if line.startswith(("block", "resuming")):
            print(line, flush=True)
    seen = len(lines)
    time.sleep(60)
print("pilot process finished")
!tail -n 5 {EXP3_LOG}
!cp /content/trajectory-lineage/results/exposure2d_capacity_pilot_v1_blinded.json {EXP3_DRIVE_DIR}/ 2>/dev/null || echo "no report yet: if blocks are missing, Runtime -> Run all resumes from the checkpoint" """),
        code(GUARD + """#@title A5. Blinded summary and download (dispersions, competence, runtime per size; pooled positive-control flag)
import json
path = f"{EXP3_DRIVE_DIR}/exposure2d_capacity_pilot_v1_blinded.json"
r = json.load(open(path))
print({k: r[k] for k in ("blocks_scheduled", "blocks_completed", "blocks_base_competent", "positive_control_passed") if k in r})
for level, v in r["by_capacity"].items():
    disp = {name: round(d["standard_deviation_upper_limit"], 5) for name, d in v["dispersion"].items()}
    print(f"hidden {level}: params {v['model_parameters']}, completed {v['blocks_completed']}, competent {v['blocks_base_competent']}, s/block {None if v['seconds_per_block_mean'] is None else round(v['seconds_per_block_mean'])}, SD upper limits {disp}")
print("infrastructure failures:", len(r["infrastructure_failures"]), "| algorithmic failures:", len(r["algorithmic_failures"]))
!git -C /content/trajectory-lineage rev-parse HEAD > {EXP3_DRIVE_DIR}/code_commit.txt
!cd {EXP3_DRIVE_DIR} && zip -q -j /content/exp3_capacity_pilot_results.zip exposure2d_capacity_pilot_v1_blinded.json exposure2d_capacity_base_tuning.json code_commit.txt pilot.log
!ls -la /content/exp3_capacity_pilot_results.zip
from google.colab import files
files.download("/content/exp3_capacity_pilot_results.zip")"""),
        md("""---
## Locked — Exp 2 registered training (do not run before the freeze is signed and the registration is submitted)
Upload the signed `freeze.json` to `OUT` first. The driver verifies config, selection, support, code hashes and sign-offs, then trains the runs with the owner's `wm.py` and resumes if the runtime restarts."""),
        code(GUARD + """#@title L1. Train (requires {OUT}/freeze.json)
import os
assert os.path.exists(f"{OUT}/freeze.json"), "no signed freeze manifest in OUT; do not train before registration"
%cd /content/exp2-kuka-matched
!PYTHONPATH=. python -m registered.run_registered --owner-repo /content/rwmo --npz-dir /content/rwmo/data/kuka --manifest {OUT}/manifest.jsonl --selection {OUT}/selection.json --support {OUT}/support.json --freeze {OUT}/freeze.json --out {OUT}/runs 2>&1 | grep -v -i "warn" | tail -n 150"""),
        code(GUARD + """#@title L2. Analyze (after all runs)
%cd /content/exp2-kuka-matched
!PYTHONPATH=. python -m registered.analyze --runs {OUT}/runs --selection {OUT}/selection.json --support {OUT}/support.json --manifest {OUT}/manifest.jsonl --judge {OUT}/runs/judge.json --freeze {OUT}/freeze.json --out {OUT}/decision.json
!cd {OUT} && zip -q -j /content/exp2_decision.zip decision.json decision.md runs/judge.json && ls -la /content/exp2_decision.zip
from google.colab import files
files.download("/content/exp2_decision.zip")"""),
        md("""### Notes
- Blinding: the pilot's Drive checkpoint holds blinded records only (contrasts shifted by the key in `DO_NOT_OPEN/`); the report contains dispersions and the pooled positive-control flag, no effect locations. Never open the key folder. Do not add `--unblind` or `--checkpoint` to the pilot command.
- Resume: after any disconnect, Runtime → Run all. Part A skips finished blocks, Part B skips finished shards and existing outputs."""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def exp3_pilots_a100_notebook() -> dict:
    """Effect-blind pilots for the two new Exp 3 axes, parallel on one GPU: architecture (latent predictor on a frozen
    DINOv2-small / V-JEPA 2-L encoder) and dose-response (16 / 256 reused sources with the ConvLSTM)."""

    GUARD = 'assert "WORKERS" in globals(), "Run the configuration cell at the top first (after any restart: Runtime -> Run all)"\n'
    cells = [
        md("""# Seen and Unseen — Experiment 3 pilots on an A100: architecture axis and dose-response axis (effect-blind)

Both parts use the registered 2D reuse design unchanged (same events, appearances, arms, steps, contrasts) and release only dispersions, competence counts, runtime and the pooled coarse positive-control flag. **No effect location is written anywhere.**

| Part | What changes vs the registered design | Blocks |
|---|---|---|
| A — architecture | Predictor = frozen pretrained encoder (DINOv2-small 22 M, or V-JEPA 2-L 326 M; never trained) + trained latent transformer (width 256, depth 4; 3.3 M) instead of the ConvLSTM | 24 per encoder |
| B — dose | ConvLSTM-16 with 16 or 256 reused source trajectories instead of 64, so each reused trajectory is presented ~50 or ~3 times instead of ~12 | 24 per dose |

**Speed:** blocks run `WORKERS` at a time in separate processes on the one GPU (results do not depend on the worker count; checked), the frozen encoders run in bf16, and every finished block is checkpointed (blinded) on Drive. **After any disconnect: Runtime → Run all**; finished blocks are skipped.

**Runtime (A100, WORKERS = 3):** about 2–3 h for everything; Part A alone about 1–1.5 h. **Send back:** `exp3_pilots_results.zip`."""),
        code("""#@title 0. Configuration
WORKERS = 3                    #@param {type:"integer"}   # blocks in parallel on the GPU; 3 on an A100 (40 GB), 2 on an L4
RUN_TUNING = True              #@param {type:"boolean"}
RUN_DINOV2 = True              #@param {type:"boolean"}
RUN_VJEPA = True               #@param {type:"boolean"}
RUN_DOSE = True                #@param {type:"boolean"}
TUNE_BLOCKS = 4                #@param {type:"integer"}
TUNE_CHECKPOINTS = "300,600,1200"  #@param {type:"string"}
EXP3_DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp3/pilots_oct28"  #@param {type:"string"}
import os, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)", "| CPUs:", os.cpu_count(), "| workers:", WORKERS)
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN", exist_ok=True)
PILOTS = {  # name -> (config, run flag)
    "latent_dinov2": ("experiments/exposure2d_latent_dinov2_pilot_v1.json", RUN_DINOV2),
    "latent_vjepa2": ("experiments/exposure2d_latent_vjepa2_pilot_v1.json", RUN_VJEPA),
    "dose16": ("experiments/exposure2d_dose16_pilot_v1.json", RUN_DOSE),
    "dose256": ("experiments/exposure2d_dose256_pilot_v1.json", RUN_DOSE),
}
for k in PILOTS:
    ck = f"{EXP3_DRIVE_DIR}/{k}_blinded_checkpoint.jsonl"
    print(f"{k:14s} blocks already recorded: {sum(1 for _ in open(ck)) if os.path.exists(ck) else 0}")"""),
        code(GUARD + f"""#@title 1. Code: public trajectory-lineage at {TL_COMMIT} (includes pull request #19)
%cd /content
import os
if not os.path.exists("trajectory-lineage"):
    !git clone -q https://github.com/physicalreasoning/trajectory-lineage.git
%cd /content/trajectory-lineage
!git checkout -q {TL_COMMIT} && git rev-parse --short HEAD
!pip install -q numpy h5py pytest "transformers>=5.0" accelerate
print("ok")"""),
        code(GUARD + """#@title 2. Tests (CPU, about 1 min); load the encoders once so the weights are cached
%cd /content/trajectory-lineage
!git log --oneline -1
!PYTHONPATH=src python -m pytest -q tests/test_latent.py tests/test_capacity.py -p no:cacheprovider 2>&1 | tail -n 2
import torch, sys
sys.path.insert(0, "/content/trajectory-lineage/src")
from trajectory_lineage.latent import FrozenEncoder
for name, flag in (("dinov2-small", RUN_DINOV2), ("vjepa2-vitl", RUN_VJEPA)):
    if flag:
        enc = FrozenEncoder(name, torch.device("cuda"))
        tok = enc.encode(torch.rand(4, 8, 3, 64, 64))
        print(f"{name}: {enc.hub_id}, frozen params {enc.parameters_frozen/1e6:.0f}M, tokens {tuple(tok.shape)}")"""),
        code(GUARD + """#@title 3. Effect-blind base tuning for the latent predictor (base models only; no arms)
%cd /content/trajectory-lineage
import os
for k in ("latent_dinov2", "latent_vjepa2"):
    cfg, flag = PILOTS[k]
    out = f"results/{k}_base_tuning.json"
    if RUN_TUNING and flag and not os.path.exists(f"{EXP3_DRIVE_DIR}/{k}_base_tuning.json"):
        print(f"=== tuning {k} ===")
        !PYTHONPATH=src python scripts/tune_base_2d.py --config {cfg} --out {out} --blocks {TUNE_BLOCKS} --checkpoints {TUNE_CHECKPOINTS} --hidden 256 2>&1 | grep -v -i "warn\\|Loading weights" | tail -n 8
        !cp {out} {EXP3_DRIVE_DIR}/{k}_base_tuning.json
    else:
        print(f"tuning {k}: skipped (disabled or already on Drive)")"""),
        code(GUARD + """#@title 4. Blind pilots, WORKERS blocks at a time, resumable from Drive; one line per finished block with an ETA
%cd /content/trajectory-lineage
import os
for k, (cfg, flag) in PILOTS.items():
    if not flag:
        print(f"{k}: disabled"); continue
    ck = f"{EXP3_DRIVE_DIR}/{k}_blinded_checkpoint.jsonl"; key = f"{EXP3_DRIVE_DIR}/DO_NOT_OPEN/{k}_blinding_key.json"
    out = f"results/{k}_pilot_v1_blinded.json"
    print(f"=== pilot {k} ===")
    !PYTHONPATH=src python scripts/exposure_experiment.py --config {cfg} --out {out} --device cuda --workers {WORKERS} --blinded-checkpoint {ck} --blinding-key {key} 2>&1 | grep -E "^block|^resuming|Error|error"
    !cp {out} {EXP3_DRIVE_DIR}/{k}_pilot_v1_blinded.json"""),
        code(GUARD + """#@title 5. Blinded summary and download
import json, os
for k in PILOTS:
    path = f"{EXP3_DRIVE_DIR}/{k}_pilot_v1_blinded.json"
    if not os.path.exists(path):
        print(f"{k:14s} no report"); continue
    r = json.load(open(path))
    d = {name: round(v["standard_deviation_upper_limit"], 5) for name, v in r["dispersion"].items() if name in ("C_source_gain", "C_independent_harm", "C_reuse_gap", "C_minus_D_gap", "E_exact_gain")}
    print(f"{k:14s} completed {r['blocks_completed']}/{r['blocks_scheduled']}, competent {r['blocks_base_competent']}, positive control {r.get('positive_control_passed')}, "
          f"failures {len(r['infrastructure_failures'])}+{len(r['algorithmic_failures'])}, s/block {r['runtime']['seconds_per_block_mean']:.0f}, SD80 {d}")
!git -C /content/trajectory-lineage rev-parse HEAD > {EXP3_DRIVE_DIR}/code_commit.txt
!cd {EXP3_DRIVE_DIR} && zip -q -j /content/exp3_pilots_results.zip *.json code_commit.txt
!ls -la /content/exp3_pilots_results.zip
from google.colab import files
files.download("/content/exp3_pilots_results.zip")"""),
        md("""### Notes
- Drive checkpoints hold blinded records only (contrasts shifted by the key in `DO_NOT_OPEN/`); never open that folder. Do not add `--unblind` or `--checkpoint`.
- If the GPU runs out of memory with V-JEPA (unlikely on an A100), set `WORKERS = 2` and Run all; finished blocks are kept.
- A level whose base models fail the competence screen is reported as untestable and is not registered; the others stand."""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def exp3_confirmation_notebook() -> dict:
    """LOCKED confirmation runner: all five Exp 3 configurations from the frozen tag, with the signed freeze manifest."""

    GUARD = 'assert "FREEZE_TAG" in globals(), "Run the configuration cell at the top first (after any restart: Runtime -> Run all)"\n'
    cells = [
        md("""# Seen and Unseen — Experiment 3 CONFIRMATION (locked; run only after registration, the freeze and the tag)

Runs the five registered Experiment 3 configurations from the frozen git tag with the signed freeze manifest: the main confirmation (1,600 blocks; capacity 16/64/256) and the four axis configurations (latent DINOv2, latent V-JEPA 2, dose 16, dose 256; 200 blocks each). Blocks run `WORKERS` at a time; every finished block is written to a raw checkpoint on Drive, so sessions can stop and resume.

**Sealed outputs.** The raw block records (`*_raw_checkpoint.jsonl`, `*_raw.json`) carry effect values. They stay on Drive, nobody opens them, and the analyzers run only after the December 8 amendment. The notebook prints and downloads **blinded reports only** (counts, dispersions, runtime).

**Refusals built in:** the runner will not start without a freeze manifest that matches the configuration, the analyzer and the code commit, with the internal sign-offs; the clone fails if the tag does not exist.

**Runtime:** about 27 h wall-clock on one A100 with three workers, across sessions (after a disconnect: Runtime → Run all; finished blocks are skipped)."""),
        code("""#@title 0. Configuration
FREEZE_TAG = "exp3-freeze-YYYYMMDD"   #@param {type:"string"}   # the dated git tag the paper lead created at the freeze
WORKERS = 3                           #@param {type:"integer"}
RUN_MAIN = True                       #@param {type:"boolean"}
RUN_AXES = True                       #@param {type:"boolean"}
DEVIATION = ""                        #@param {type:"string"}   # only if the GPU or runtime differs from the freeze; the reason is logged
EXP3_DRIVE_DIR = "/content/drive/MyDrive/seen-and-unseen/exp3/confirmation"  #@param {type:"string"}
import os, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)", "| CPUs:", os.cpu_count(), "| workers:", WORKERS)
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{EXP3_DRIVE_DIR}/SEALED_raw_records", exist_ok=True)
FREEZE = f"{EXP3_DRIVE_DIR}/freeze_manifest.json"
assert os.path.exists(FREEZE), f"put the signed freeze manifest at {FREEZE} first"
CONFIGS = {"main": "experiments/exposure2d_confirmation.json"}
if RUN_AXES:
    CONFIGS.update({"latent_dinov2": "experiments/exposure2d_confirm_latent_dinov2.json", "latent_vjepa2": "experiments/exposure2d_confirm_latent_vjepa2.json",
                    "dose16": "experiments/exposure2d_confirm_dose16.json", "dose256": "experiments/exposure2d_confirm_dose256.json"})
if not RUN_MAIN:
    CONFIGS.pop("main")
for k in CONFIGS:
    ck = f"{EXP3_DRIVE_DIR}/SEALED_raw_records/{k}_raw_checkpoint.jsonl"
    print(f"{k:14s} blocks already recorded: {sum(1 for _ in open(ck)) if os.path.exists(ck) else 0}")"""),
        code(GUARD + """#@title 1. Code at the frozen tag
%cd /content
import os
if not os.path.exists("trajectory-lineage"):
    !git clone -q https://github.com/physicalreasoning/trajectory-lineage.git
%cd /content/trajectory-lineage
!git fetch -q --tags && git checkout -q {FREEZE_TAG} && git rev-parse HEAD
import json
env = json.load(open(FREEZE)).get("environment") or {}
pins = [f"{k}=={v}" for k, v in (env.get("runtime") or {}).items() if k in ("numpy", "scipy", "h5py", "transformers", "accelerate") and v != "absent"]
PIN_ARGS = " ".join(pins) if pins else 'numpy h5py "transformers>=5.0" accelerate'
print("installing the frozen versions:", PIN_ARGS, "| frozen hardware class:", env.get("hardware_class"))
!pip install -q {PIN_ARGS}
!PYTHONPATH=src python -c "import json,sys; sys.path.insert(0,'scripts'); import freeze; print(json.dumps(freeze.check_runtime(json.load(open('{FREEZE}')), 'cuda', '{DEVIATION}' or None)))"
print("ok")"""),
        code(GUARD + """#@title 2. Confirmation runs (resumable; blinded progress only)
%cd /content/trajectory-lineage
DEV_ARG = f'--deviation "{DEVIATION}"' if DEVIATION else ""
for k, cfg in CONFIGS.items():
    ck = f"{EXP3_DRIVE_DIR}/SEALED_raw_records/{k}_raw_checkpoint.jsonl"
    raw = f"{EXP3_DRIVE_DIR}/SEALED_raw_records/{k}_raw.json"
    out = f"results/{k}_confirmation_blinded.json"
    print(f"=== {k} ===")
    !PYTHONPATH=src python scripts/exposure_experiment.py --config {cfg} --out {out} --device cuda --workers {WORKERS} --freeze-manifest {FREEZE} --checkpoint {ck} --unblind {raw} {DEV_ARG} 2>&1 | grep -E "^block|^resuming|^environment|refusing|Error|error"
    !cp {out} {EXP3_DRIVE_DIR}/{k}_confirmation_blinded.json"""),
        code(GUARD + """#@title 3. Blinded accounting and download (no effect values)
import json, os
for k in CONFIGS:
    path = f"{EXP3_DRIVE_DIR}/{k}_confirmation_blinded.json"
    if not os.path.exists(path):
        print(f"{k:14s} not finished"); continue
    r = json.load(open(path))
    print(f"{k:14s} completed {r['blocks_completed']}/{r['blocks_scheduled']}, competent {r['blocks_base_competent']}, failures {len(r['infrastructure_failures'])}+{len(r['algorithmic_failures'])}, s/block {r['runtime']['seconds_per_block_mean']:.0f}")
!cd {EXP3_DRIVE_DIR} && zip -q -j /content/exp3_confirmation_blinded.zip *_confirmation_blinded.json
!ls -la /content/exp3_confirmation_blinded.zip
from google.colab import files
files.download("/content/exp3_confirmation_blinded.zip")"""),
        md("""### After all five report complete
The analysis is a separate, later step after the December 8 amendment: `scripts/analyze_2d.py` on the main raw export and `scripts/analyze_exp3_axes.py` across all five, both of which refuse to run without the freeze manifest. Do not open `SEALED_raw_records/`."""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def exp3_environment_notebook() -> dict:
    """Records the environment block for the Exp 3 freeze manifest on the Colab runtime that will run the confirmation."""

    cells = [
        md("""# Seen and Unseen — Experiment 3: record the freeze environment (before the freeze)

Run this once on the same kind of Colab runtime as the confirmation (A100, High-RAM). It installs the packages the confirmation uses and writes the **environment block** for the freeze manifest: the hash of the repository lockfile, the exact versions of Python, torch, CUDA, numpy, scipy, h5py, transformers and accelerate with their hash, and the hardware class. The locked confirmation notebook installs these same versions, and the runner refuses a run on other hardware or versions unless a logged deviation is given.

Takes about 2 minutes. Downloads `environment.json`; paste it into the freeze manifest as `"environment"`."""),
        code(f"""#@title 0. Configuration
HARDWARE_CLASS = "A100"   #@param {{type:"string"}}
CODE_COMMIT = "{TL_COMMIT}"  #@param {{type:"string"}}   # the commit the freeze tag will point to
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (switch to a GPU runtime)")"""),
        code("""#@title 1. Code and packages (as the confirmation installs them)
%cd /content
import os
if not os.path.exists("trajectory-lineage"):
    !git clone -q https://github.com/physicalreasoning/trajectory-lineage.git
%cd /content/trajectory-lineage
!git fetch -q && git checkout -q {CODE_COMMIT} && git rev-parse --short HEAD
!pip install -q numpy h5py "transformers>=5.0" accelerate"""),
        code("""#@title 2. Write and download the environment block
%cd /content/trajectory-lineage
!python scripts/freeze.py environment --hardware-class "{HARDWARE_CLASS}" --out /content/environment.json
from google.colab import files
files.download("/content/environment.json")"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def validate(nb: dict) -> list[str]:
    """Static checks that catch the mistakes Colab only reports at run time.

    * a cell magic (%%...) must be the first line of its cell;
    * every non-magic Python line must parse (IPython ! and % lines count as statements);
    * every {NAME} placeholder inside a ! shell line must be assigned in this or an earlier cell
      (assignments, imports, for-loop variables, with-targets);
    * every %%writefile target directory must be created by an earlier cell;
    * every embedded .py file must compile.
    """

    import ast
    import re

    errors: list[str] = []
    defined: set[str] = {"os"}
    made_dirs: list[str] = []

    def names_in(target) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(names_in(t) for t in target.elts)) if target.elts else set()
        return set()

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines = c["source"].split("\n")
        for j, line in enumerate(lines):
            if line.startswith("%%") and j != 0:
                errors.append(f"cell {i}: cell magic '{line[:20]}' is not on the first line")
        if lines[0].startswith("%%writefile"):
            target = lines[0].split(None, 1)[1].strip()
            if not any(target.startswith(d.rstrip("/") + "/") for d in made_dirs):
                errors.append(f"cell {i}: %%writefile {target}: directory not created by an earlier cell")
            if target.endswith(".py"):
                try:
                    compile("\n".join(lines[1:]), target, "exec")
                except SyntaxError as e:
                    errors.append(f"cell {i}: embedded {target} does not compile: {e}")
            continue
        py = "\n".join((l[: len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l for l in lines)
        try:
            tree = ast.parse(py)
        except SyntaxError as e:
            errors.append(f"cell {i}: Python syntax error: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    defined.update(names_in(t))
            elif isinstance(node, (ast.For, ast.comprehension)):
                defined.update(names_in(node.target))
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                defined.update(names_in(node.optional_vars))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "makedirs" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant):
                    made_dirs.append(str(a.value))
        for line in lines:
            if line.lstrip().startswith("!"):
                for name in re.findall(r"\{([A-Za-z_][A-Za-z_0-9]*)", line):
                    if name not in defined:
                        errors.append(f"cell {i}: {{{name}}} used in a shell line before it is assigned")
    return errors


if __name__ == "__main__":
    for name, builder in (("run_all_exp3_then_exp2.ipynb", combined_notebook), ("run_exp3_pilots_a100.ipynb", exp3_pilots_a100_notebook), ("LOCKED_exp3_confirmation.ipynb", exp3_confirmation_notebook), ("prepare_exp3_freeze_environment.ipynb", exp3_environment_notebook)):
        nb = builder()
        problems = validate(nb)
        if problems:
            raise SystemExit(f"{name} NOT WRITTEN; fix these first:\n  " + "\n  ".join(problems))
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1))
        json.loads(out.read_text())
        print("validated and wrote", out, out.stat().st_size, "bytes")
