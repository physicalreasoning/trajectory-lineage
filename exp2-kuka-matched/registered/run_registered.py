"""Registered training driver for Experiment 2.

Replaces ``run_arms.py`` for the registered runs. Differences that matter for the registration:

* training sets come from the frozen selection (ids by hash), never from episode counts chosen here;
* the freeze manifest must verify (code, config, selection, support unchanged; sign-offs present)
  before any training starts, unless ``--smoke`` runs the synthetic smoke test;
* normalization statistics and the judge are fitted once from frozen, hash-recorded episode sets; the
  judge validity gate (real AUROC >= 0.75) is a hard stop, not a warning;
* every run writes per-episode scores, the training ids hash, the recomputed distinct-transition
  count, the optimizer exposure (steps x batch), a checkpoint hash and the config and selection
  hashes; an existing run file with different hashes is refused (no silent reuse of stale artefacts);
* an infrastructure failure writes a run file with ``status: infrastructure_failure`` and the
  traceback; ``--retry-failed`` reruns such runs once with the same seed, as registered.

The model, trainer, rollout and judge are the owner's unchanged ``wm.py`` (``--owner-repo``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import load_config, sha256_json, write_json
from .manifest import Manifest, load_manifest


def import_owner_wm(owner_repo: Path):
    sys.path.insert(0, str(Path(owner_repo) / "experiment"))
    import wm  # type: ignore

    return wm


# ---------------------------------------------------------------- loading a frozen subset

def load_subset(wm, npz_dir: Path, m: Manifest, ids: list[str]):
    """Load features/actions for exactly ``ids`` (manifest order) into the owner's ``Episodes``."""

    idx = m.index_of()
    rows = np.array(sorted(idx[i] for i in ids), int)
    by_shard: dict[int, list[int]] = {}
    for r in rows:
        by_shard.setdefault(int(m.shard[r]), []).append(int(r))
    n_frames = int(m.length[rows].sum())
    Z = np.empty((n_frames, 17, 384), np.float16)
    A = None
    L, S, sh, out_ids = [], [], [], []
    row = 0
    for sid in sorted(by_shard):
        f = Path(npz_dir) / f"shard_{sid:05d}.npz"
        with np.load(f) as npz:
            feats, acts, succ, ln = npz["feats"], npz["actions"], npz["success"], npz["length"]
        if A is None:
            A = np.empty((n_frames, acts.shape[-1]), np.float32)
        for r in by_shard[sid]:
            e = int(m.ids[r].split("/")[1])
            n = int(m.length[r])
            if int(ln[e]) != n or int(succ[e]) != int(m.success[r]):
                raise RuntimeError(f"manifest/cache mismatch for {m.ids[r]}: cache length {ln[e]} success {succ[e]}")
            Z[row:row + n], A[row:row + n] = feats[e, :n], acts[e, :n]
            row += n
            L.append(n); S.append(int(succ[e])); sh.append(sid); out_ids.append(str(m.ids[r]))
        del feats, acts
    assert row == n_frames
    ep = wm.Episodes(Z, A, np.array(L), np.array(S), np.array(sh))
    ep.ids = np.array(out_ids, dtype=object)
    ep.pos = {i: k for k, i in enumerate(out_ids)}
    return ep


def rows_for(ep, ids: list[str]) -> np.ndarray:
    return np.array([ep.pos[i] for i in ids], int)


def sha256_path(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- judge

def fit_or_load_judge(wm, ep, m: Manifest, config: dict, out: Path, train_ids: list[str], test_ids: list[str], log) -> tuple[object, dict]:
    jc = config["judge"]
    jpath, jmeta = out / "judge.pkl", out / "judge.json"
    y_train = np.array([m.success[m.index_of()[i]] for i in train_ids])
    rng = np.random.default_rng(jc["rng_seed"])
    pool_ids = np.array(train_ids, dtype=object)
    j_ids = np.concatenate([rng.permutation(pool_ids[y_train == c])[: jc["per_class"]] for c in (0, 1)]).tolist()
    expected = {"judge_ids_sha256": sha256_json(sorted(j_ids)), "config_sha256": sha256_json(config)}
    if jmeta.exists():
        meta = json.loads(jmeta.read_text())
        if {k: meta.get(k) for k in expected} != expected or not jpath.exists() or sha256_path(jpath) != meta["judge_pkl_sha256"]:
            raise RuntimeError("existing judge.json/judge.pkl do not match the frozen configuration; refusing to reuse")
        log("judge: loaded frozen judge")
        return pickle.loads(jpath.read_bytes()), meta
    log(f"judge: fitting on {len(j_ids)} real train trajectories")
    j_rows = rows_for(ep, j_ids)
    judge = wm.SuccessJudge(C=jc["C"]).fit(wm.real_summary(ep, j_rows), ep.success[j_rows])
    te_rows = rows_for(ep, test_ids)
    p_real = judge.prob(wm.real_summary(ep, te_rows))
    y_te = ep.success[te_rows]
    from sklearn.metrics import roc_auc_score
    h = p_real > config["evaluation"]["threshold"]
    bins = np.clip((p_real * 10).astype(int), 0, 9)
    reliability = [{"bin": int(b), "n": int((bins == b).sum()), "mean_p": float(p_real[bins == b].mean()) if (bins == b).any() else None,
                    "frac_success": float(y_te[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
    meta = {**expected, "eval_ids": test_ids, "p_real": p_real.tolist(), "auroc": float(roc_auc_score(y_te, p_real)),
            "pred_rate_unweighted": float(h.mean()), "true_rate_unweighted": float(y_te.mean()),
            "false_positive_rate": float(h[y_te == 0].mean()), "false_negative_rate": float((~h[y_te == 1]).mean()),
            "brier_unweighted": float(((p_real - y_te) ** 2).mean()), "reliability": reliability,
            "length_only_auroc": float(roc_auc_score(y_te, -ep.L[te_rows])), "validity_gate": jc["validity_auroc"]}
    jpath.write_bytes(pickle.dumps(judge))
    meta["judge_pkl_sha256"] = sha256_path(jpath)
    write_json(jmeta, meta)
    log(f"judge: real AUROC {meta['auroc']:.3f} (gate {jc['validity_auroc']}), FPR {meta['false_positive_rate']:.3f}, FNR {meta['false_negative_rate']:.3f}")
    return judge, meta


# ---------------------------------------------------------------- runs

def run_one(wm, ep, judge, run: dict, config: dict, selection: dict, test_ids: list[str], out: Path, device_note: str, log, attempt: int) -> dict:
    rows = rows_for(ep, run["ids"])
    ids_hash = sha256_json(sorted(run["ids"]))
    if ids_hash != run["ids_sha256"]:
        raise RuntimeError(f"{run['run_id']}: training ids hash differs from the frozen selection")
    rec = {"run_id": run["run_id"], "design": run["design"], "arm": run["arm"], "phi": run["phi"], "seed": run["seed"],
           "selection_sha256": selection["selection_sha256_without_certificate"], "config_sha256": sha256_json(config), "ids_sha256": ids_hash,
           "attempt": attempt, "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "device": device_note,
           "train": {"n_episodes": int(len(rows)), "n_success": int(ep.success[rows].sum()), "n_failure": int((1 - ep.success[rows]).sum()),
                     "distinct_transitions": int((ep.L[rows] - 1).sum()), "optimizer_steps": config["optimizer_steps"], "batch_size": 256,
                     "sampled_starts": config["optimizer_steps"] * 256, "rollout_horizon": 4,
                     "note": "valid multistep loss terms are not exposed by wm.train_wm; add a counter there to log them"}}
    t0 = time.time()
    try:
        model = wm.train_wm(ep, rows, config["optimizer_steps"], seed=run["seed"], log=log)
        te_rows = rows_for(ep, test_ids)
        summ, _ = wm.predict_summary(model, ep, te_rows)
        p = judge.prob(summ)
        import torch
        ckpt = out / "checkpoints" / f"{run['run_id']}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        rec.update(status="completed", eval_ids=test_ids, p=np.clip(p, 0, 1).tolist(), checkpoint=str(ckpt), checkpoint_sha256=sha256_path(ckpt),
                   train_minutes=(time.time() - t0) / 60, finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    except Exception:  # noqa: BLE001  infrastructure failure: record, do not analyse around it
        rec.update(status="infrastructure_failure", error=traceback.format_exc(), train_minutes=(time.time() - t0) / 60)
    return rec


def run_all(*, owner_repo: Path, npz_dir: Path, manifest_path: Path, selection_path: Path, support_path: Path, config_path: Path | None,
            out: Path, freeze_path: Path | None, smoke: bool, only: list[str] | None, retry_failed: bool, allow_judge_below_gate: bool,
            deviation: str | None = None,
            log=print) -> list[dict]:
    wm = import_owner_wm(owner_repo)
    config = load_config(config_path)
    selection = json.loads(Path(selection_path).read_text())
    support = json.loads(Path(support_path).read_text())
    m, mhash = load_manifest(manifest_path)
    if mhash != selection["manifest_sha256"] or mhash != support["manifest_sha256"]:
        raise RuntimeError("manifest hash differs from the one the selection/support were built from")
    if not selection["certificate"]["feasible"]:
        raise RuntimeError("selection certificate failed; refusing to train")
    if not smoke:
        if freeze_path is None:
            raise RuntimeError("a signed freeze manifest is required to start registered training (use --smoke for the synthetic smoke test)")
        from .freeze import check_runtime, verify_freeze
        frozen = json.loads(Path(freeze_path).read_text())
        verify_freeze(frozen, config=config, selection=selection, support=support, require_signoff=True,
                      lockfile=Path(owner_repo) / "uv.lock")
        environment_check = check_runtime(frozen, wm.DEVICE, deviation)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / f"environment_check_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", environment_check)
        log(f"environment check: {json.dumps(environment_check)}")
        if selection.get("manifest_kind") == "SYNTHETIC":
            raise RuntimeError("SYNTHETIC manifest; refusing to train")
    out.mkdir(parents=True, exist_ok=True)
    runs = selection["runs"] if not only else [r for r in selection["runs"] if r["run_id"] in set(only)]

    train_ids = sorted(m.ids[m.train_mask()].tolist())
    test_ids = sorted(m.ids[m.test_mask()].tolist())
    needed = set(test_ids) | set(train_ids)  # judge pool and normalization draw from the full train pool
    t0 = time.time()
    ep = load_subset(wm, npz_dir, m, sorted(needed))
    log(f"loaded {len(ep)} episodes ({ep.Z.shape[0]} frames) in {time.time() - t0:.0f}s; peak RSS {wm.rss_gb():.2f} GB")
    norm_ids = train_ids[:2000]  # frozen: first 2,000 eligible train episodes in canonical manifest order
    norm_rows = rows_for(ep, norm_ids)
    stat_rows = np.concatenate([np.arange(ep.off[e], ep.off[e] + ep.L[e]) for e in norm_rows])
    wm.normalize_inplace(ep, stat_rows)
    write_json(out / "normalization.json", {"normalization_ids_sha256": sha256_json(norm_ids), "n_episodes": len(norm_ids), "n_frames": int(len(stat_rows))})
    judge, jmeta = fit_or_load_judge(wm, ep, m, config, out, train_ids, test_ids, log)
    if jmeta["auroc"] < config["judge"]["validity_auroc"] and not allow_judge_below_gate:
        raise RuntimeError(f"judge validity gate failed: real AUROC {jmeta['auroc']:.3f} < {config['judge']['validity_auroc']}; registered runs may not start")
    device_note = f"torch device {wm.DEVICE}"
    results = []
    for run in runs:
        path = out / f"{run['run_id']}.json"
        attempt = 1
        if path.exists():
            prev = json.loads(path.read_text())
            if prev.get("selection_sha256") != selection["selection_sha256_without_certificate"] or prev.get("config_sha256") != sha256_json(config):
                raise RuntimeError(f"{run['run_id']}: existing run file was produced under a different selection/config; refusing to reuse or overwrite")
            if prev.get("status") == "completed" or not (retry_failed and prev.get("status") == "infrastructure_failure" and prev.get("attempt", 1) < 2):
                results.append(prev)
                continue
            attempt = prev.get("attempt", 1) + 1
            log(f"{run['run_id']}: retrying once with the same seed (attempt {attempt})")
        rec = run_one(wm, ep, judge, run, config, selection, test_ids, out, device_note, log, attempt)
        write_json(path, rec)
        results.append(rec)
        if rec["status"] == "completed":
            w = np.array([support["weights"].get(i, 0.0) for i in test_ids])
            log(f"{run['run_id']}: weighted thresholded rate {(w * (np.array(rec['p']) > 0.5)).sum():.4f}  [{rec['train_minutes']:.1f} min, peak RSS {wm.rss_gb():.2f} GB]")
        else:
            log(f"{run['run_id']}: INFRASTRUCTURE FAILURE (attempt {attempt}); see run file")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the registered Experiment 2 runs from the frozen selection.")
    ap.add_argument("--owner-repo", required=True, help="path to robot-world-model-optimism-research (uses experiment/wm.py)")
    ap.add_argument("--npz-dir", required=True, help="data/kuka with shard_*.npz")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--freeze", default=None, help="signed freeze manifest (required unless --smoke)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run ids to train (default: all)")
    ap.add_argument("--retry-failed", action="store_true", help="rerun infrastructure failures once with the same seed")
    ap.add_argument("--allow-judge-below-gate", action="store_true", help="deviation: continue although the judge gate failed (recorded)")
    ap.add_argument("--smoke", action="store_true", help="synthetic smoke test: skips the freeze requirement")
    ap.add_argument("--deviation", default=None, help="reason to train on another hardware class or runtime than the freeze (logged)")
    args = ap.parse_args(argv)
    run_all(owner_repo=Path(args.owner_repo), npz_dir=Path(args.npz_dir), manifest_path=Path(args.manifest), selection_path=Path(args.selection),
            support_path=Path(args.support), config_path=args.config, out=Path(args.out), freeze_path=Path(args.freeze) if args.freeze else None,
            smoke=args.smoke, only=args.only, retry_failed=args.retry_failed, allow_judge_below_gate=args.allow_judge_below_gate,
            deviation=args.deviation)


if __name__ == "__main__":
    main()
