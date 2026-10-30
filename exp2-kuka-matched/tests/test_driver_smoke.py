"""End-to-end smoke test of the driver on a tiny synthetic feature cache (needs torch and the owner's repo)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from registered import load_config, write_json
from registered.analyze import analyze, load_runs, truth_from_manifest
from registered.design import build_selection
from registered.manifest import build_manifest, save_manifest
from registered.weights import evaluation_support

OWNER = Path(os.environ.get("RWMO_REPO", "/tmp/rwmo"))
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not (OWNER / "experiment" / "wm.py").exists(), reason="owner repo not available")


def tiny_cache(root: Path, rng: np.random.Generator) -> None:
    (root / "kuka").mkdir(parents=True)
    plan = {0: (20, 45), 1: (20, 45), 2: (20, 45)}  # test shards: (S, F)
    plan.update({s: (45, 45) for s in range(3, 12)})  # train shards
    for sid, (n_s, n_f) in plan.items():
        E = n_s + n_f
        length = np.concatenate([rng.choice([9, 11, 13], n_s), rng.choice([11, 13, 21], n_f)])
        success = np.array([1] * n_s + [0] * n_f)
        feats = (rng.standard_normal((E, 24, 17, 384)) * 0.1).astype(np.float16)
        feats[:, :, 0, :] += success[:, None, None] * 0.5  # a weak success signal for the judge
        actions = rng.standard_normal((E, 24, 13)).astype(np.float32)
        for e in range(E):
            feats[e, length[e]:] = 0
            actions[e, length[e]:] = 0
        np.savez(root / "kuka" / f"shard_{sid:05d}.npz", feats=feats, actions=actions, proprio=np.zeros((E, 24, 2), np.float32),
                 length=length, success=success, proprio_ok=np.ones(E, bool))


def tiny_config() -> dict:
    c = copy.deepcopy(load_config())
    c.update(name="smoke", seeds=[100, 101], optimizer_steps=60)
    c["designs"]["O"]["arms"] = {"phi0.10": {"n_success": 90, "n_failure": 10}, "phi0.65": {"n_success": 35, "n_failure": 65}}
    c["designs"]["A"].update(transitions=1000)
    c["designs"]["B"].update(transitions=1000)
    c["designs"]["V"].update(base_successes=30, total_transitions=1000)
    c["evaluation"].update(min_eval_per_outcome_per_length=2)
    c["judge"].update(per_class=40)
    return c


def test_driver_end_to_end(tmp_path):
    from registered.run_registered import run_all

    rng = np.random.default_rng(0)
    tiny_cache(tmp_path, rng)
    m = build_manifest(tmp_path / "kuka")
    mhash = save_manifest(m, tmp_path / "manifest.jsonl")
    config = tiny_config()
    write_json(tmp_path / "config.json", config)
    sel = build_selection(m, mhash, config)
    assert sel["certificate"]["feasible"], sel["certificate"]["failed"]
    write_json(tmp_path / "selection.json", sel)
    sup = evaluation_support(m, config)
    sup["manifest_sha256"] = mhash
    write_json(tmp_path / "support.json", sup)
    only = ["A_phi0.10_s100", "A_phi0.65_s100"]
    logs: list[str] = []
    res = run_all(owner_repo=OWNER, npz_dir=tmp_path / "kuka", manifest_path=tmp_path / "manifest.jsonl", selection_path=tmp_path / "selection.json",
                  support_path=tmp_path / "support.json", config_path=tmp_path / "config.json", out=tmp_path / "runs", freeze_path=None, smoke=True,
                  only=only, retry_failed=False, allow_judge_below_gate=True, log=logs.append)
    assert [r["status"] for r in res] == ["completed", "completed"], [r.get("error") for r in res]
    rec = json.loads((tmp_path / "runs" / "A_phi0.10_s100.json").read_text())
    assert rec["train"]["distinct_transitions"] == 1000 and len(rec["p"]) == len(rec["eval_ids"]) == int(m.test_mask().sum())
    assert (tmp_path / "runs" / "judge.json").exists() and (tmp_path / "runs" / "normalization.json").exists()
    assert Path(rec["checkpoint"]).exists()
    # resume does not retrain; a stale file is refused
    res2 = run_all(owner_repo=OWNER, npz_dir=tmp_path / "kuka", manifest_path=tmp_path / "manifest.jsonl", selection_path=tmp_path / "selection.json",
                   support_path=tmp_path / "support.json", config_path=tmp_path / "config.json", out=tmp_path / "runs", freeze_path=None, smoke=True,
                   only=only, retry_failed=False, allow_judge_below_gate=True, log=logs.append)
    assert res2[0]["finished"] == res[0]["finished"]
    stale = json.loads((tmp_path / "runs" / "A_phi0.65_s100.json").read_text()); stale["config_sha256"] = "old"
    (tmp_path / "runs" / "A_phi0.65_s100.json").write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="different selection/config"):
        run_all(owner_repo=OWNER, npz_dir=tmp_path / "kuka", manifest_path=tmp_path / "manifest.jsonl", selection_path=tmp_path / "selection.json",
                support_path=tmp_path / "support.json", config_path=tmp_path / "config.json", out=tmp_path / "runs", freeze_path=None, smoke=True,
                only=only, retry_failed=False, allow_judge_below_gate=True, log=logs.append)
    (tmp_path / "runs" / "A_phi0.65_s100.json").unlink()
    # the analyzer accepts the partial run set and makes every decision inconclusive
    out = analyze(load_runs(tmp_path / "runs"), sel, sup, config, judge=json.loads((tmp_path / "runs" / "judge.json").read_text()),
                  truth=truth_from_manifest(tmp_path / "manifest.jsonl"))
    assert set(out["decisions"].values()) == {"inconclusive"}
    assert out["judge_on_real"]["auroc_unweighted"] is not None
    assert len(out["missing_runs"]) == sel["n_runs"] - 1


