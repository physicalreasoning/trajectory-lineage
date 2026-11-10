from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from scipy import stats

from registered import sha256_json
from registered.analyze import AnalysisError, analyze, holm, paired_t, validate_runs
from registered.dryrun import synthetic_runs
from registered.freeze import (
    FreezeError,
    build_freeze,
    check_runtime,
    environment_block,
    owner_pipeline_block,
    sign,
    verify_freeze,
)


# ---------------------------------------------------------------- weights

def test_weights_sum_to_one_and_preserve_truth(support, manifest):
    assert support["weight_sum"] == pytest.approx(1.0, abs=1e-12)
    assert support["weighted_truth"] == pytest.approx(support["prevalence_pi"], abs=1e-12)
    idx = manifest.index_of()
    w = support["weights"]
    # both outcomes share the same length distribution q
    for y in (0, 1):
        tot = sum(v for i, v in w.items() if manifest.success[idx[i]] == y)
        for l in support["support_lengths"]:
            share = sum(v for i, v in w.items() if manifest.success[idx[i]] == y and manifest.length[idx[i]] == l) / tot
            assert share == pytest.approx(support["q_by_length"][str(l)], abs=1e-12)


def test_support_excludes_thin_lengths(support, manifest):
    te = manifest.test_mask()
    for l in support["support_lengths"]:
        for y in (0, 1):
            assert int(((manifest.length == l) & te & (manifest.success == y)).sum()) >= 20
    assert support["ess"] <= support["retained_test_episodes"]


# ---------------------------------------------------------------- inference

def test_paired_t_matches_scipy_and_tost():
    d = np.array([11.1071, 8.1071, 11.5714])
    t = paired_t(d, 3.0, 0.05, 0.90)
    assert t["mean"] == pytest.approx(10.2619, abs=1e-4) and t["sd"] == pytest.approx(1.8805, abs=1e-4)
    se = t["sd"] / np.sqrt(3)
    assert t["p_support"] == pytest.approx(stats.t.sf((t["mean"] - 3) / se, 2))
    assert t["lower95_one_sided"] == pytest.approx(t["mean"] - stats.t.ppf(0.95, 2) * se)
    null = paired_t(np.array([0.5, -0.4, 0.2, 0.1, -0.3, 0.0, 0.4, -0.2, 0.3, -0.1]), 3.0, 0.05, 0.90)
    assert null["p_equivalence"] < 0.001 and null["ci90"][0] > -3 and null["ci90"][1] < 3


def test_holm_step_down():
    adj = holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}, 0.05)
    assert adj["a"]["p_holm"] == pytest.approx(0.004) and adj["a"]["reject"]
    assert adj["b"]["p_holm"] == pytest.approx(0.06) and not adj["b"]["reject"]
    assert adj["c"]["p_holm"] == pytest.approx(0.06)  # monotone
    assert adj["d"]["p_holm"] == 0.5
    assert adj["a"]["step_alpha"] == pytest.approx(0.05 / 4) and adj["d"]["step_alpha"] == pytest.approx(0.05)


def test_holm_keeps_every_member_and_declares_at_the_level():
    # v3.4 section C: an untestable member enters with p = 1, so the family keeps its size
    adj = holm({"a": 0.01, "b": None, "c": 0.5}, 0.05)
    assert adj["b"]["p"] is None and adj["b"]["p_entered"] == 1.0 and not adj["b"]["reject"]
    assert adj["a"]["p_holm"] == pytest.approx(0.03)  # m = 3, not 2
    # "declared when its Holm-adjusted p is at most the familywise level"
    assert holm({"a": 0.025, "b": 0.9}, 0.05)["a"]["reject"]


# ---------------------------------------------------------------- analyzer end to end on synthetic scores

def _runs(selection, support, config, manifest, truth, sd=1.88, seed=0):
    te = manifest.test_mask()
    rng = np.random.default_rng(seed)
    return synthetic_runs(rng, selection, support, config, truth, manifest.ids[te].tolist(), manifest.success[te], sd)


def _truth_map(manifest):
    return {str(i): int(y) for i, y in zip(manifest.ids, manifest.success)}


def test_analyzer_supports_large_effect(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=10, A=10, B=10, V=10))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    assert res["decisions"]["A"] == "supported" and res["decisions"]["O"] == "supported"
    assert res["quantities"]["A"]["t"]["n"] == 10
    assert "Effect survives transition matching" in " ".join(res["interpretation"])


def test_analyzer_null_is_equivalent_not_supported(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=0, A=0, B=0, V=0))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    assert res["decisions"]["A"] == "equivalent"


def test_missing_run_makes_dependent_decisions_inconclusive(selection, support, config, manifest):
    missing = ["A_phi0.65_s104", "B_phi0.10_s101"]
    runs = _runs(selection, support, config, manifest, dict(O=10, A=10, B=10, V=10, missing=missing))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    for q in ("A", "O_minus_A", "B", "A_minus_B"):  # A_minus_B depends on B whether or not AB comparators exist
        assert res["decisions"][q] == "inconclusive"
        assert res["quantities"][q]["t"] is None
        lo, hi = res["quantities"][q]["worst_case_bounds_on_mean"]
        assert lo < hi and hi - lo >= 10.0 - 1e-9  # one missing run moves the mean by up to 100/10 points
    assert res["decisions"]["O"] == "supported" and res["decisions"]["V"] == "supported"
    assert res["failed_runs"] == {m: "infrastructure_failure" for m in missing}


def test_validation_rejects_stale_and_foreign_runs(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=0, A=0, B=0, V=0))
    stale = copy.deepcopy(runs[0]); stale["selection_sha256"] = "deadbeef"
    assert any("stale" in p for p in validate_runs([stale], selection, support, config))
    foreign = copy.deepcopy(runs[0]); foreign["run_id"] = "A_phi0.10_s0"
    assert any("not in the frozen selection" in p for p in validate_runs([foreign], selection, support, config))
    dup = [runs[0], copy.deepcopy(runs[0])]
    assert any("duplicate" in p for p in validate_runs(dup, selection, support, config))
    short = copy.deepcopy(runs[0]); short["eval_ids"] = short["eval_ids"][:10]; short["p"] = short["p"][:10]
    assert any("no score" in p for p in validate_runs([short], selection, support, config))
    bad = copy.deepcopy(runs[0]); bad["p"][0] = 1.5
    assert any("outside [0, 1]" in p for p in validate_runs([bad], selection, support, config))
    with pytest.raises(AnalysisError):
        analyze([stale], selection, support, config, truth=_truth_map(manifest), with_auroc=False)


def test_primary_can_be_refuted_and_secondary_intervals_use_holm_levels(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=0, A=-10, B=0, V=0))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    assert res["decisions"]["A"] == "refuted"  # one-sided 95% upper bound below -3
    assert res["quantities"]["A"]["t"]["upper95_one_sided"] < -3
    assert len(res["holm"]) == 10
    for q in config["decision"]["secondary"]:
        hi = res["quantities"][q]["holm_intervals"]
        for kind in ("support", "equivalence"):
            level = 1 - 2 * res["holm"][f"{q}:{kind}"]["step_alpha"]
            assert hi[kind]["level"] == pytest.approx(level)
            assert 0.89 <= level <= 0.99 + 1e-12  # from 1 - 2(0.05/10) to 1 - 2(0.05)
            lo, up = hi[kind]["ci"]
            t = res["quantities"][q]["t"]
            assert lo <= t["mean"] <= up


def test_missing_run_keeps_ten_members_in_the_holm_family(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=10, A=10, B=10, V=10, missing=["B_phi0.10_s101"]))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    assert len(res["holm"]) == 10
    assert res["holm"]["B:support"]["p_entered"] == 1.0 and not res["holm"]["B:support"]["reject"]


def test_mean_probability_sensitivity_is_reported(selection, support, config, manifest):
    runs = _runs(selection, support, config, manifest, dict(O=10, A=10, B=10, V=10))
    res = analyze(runs, selection, support, config, truth=_truth_map(manifest), with_auroc=False)
    t_thr, t_mp = res["quantities"]["A"]["t"], res["quantities"]["A"]["t_meanp_sensitivity"]
    assert t_mp["mean"] == pytest.approx(t_thr["mean"] / 2, abs=1.5)


# ---------------------------------------------------------------- freeze

def test_freeze_binds_and_rejects(selection, support, config, tmp_path, owner_repo):
    real_sel = copy.deepcopy(selection); real_sel["manifest_kind"] = "real"; real_sel["manifest_sha256"] = "testmanifest"
    with pytest.raises(FreezeError):
        build_freeze(config, tmp_path / "m.jsonl", selection, support, [], None)  # synthetic refused
    lock = tmp_path / "uv.lock"; lock.write_text("owner lock v1\n")
    env = environment_block("CPU", lock)
    fz = build_freeze(config, tmp_path / "m.jsonl", real_sel, support, [{"role": "paper_lead", "name": "Lead Name", "date": "2026-10-29"},
                                                                        {"role": "stats_check", "name": "Stats Name", "date": "2026-10-29"}], "osf.io/xxxx", env,
                      owner_pipeline_block(owner_repo))
    verify_freeze(fz, config=config, selection=real_sel, support=support, require_signoff=True, lockfile=lock)
    # environment: required, lockfile checked, runtime hash consistent, hardware named
    with pytest.raises(FreezeError, match="no environment block"):
        verify_freeze({**fz, "environment": None}, config=config, selection=real_sel, support=support, require_signoff=False)
    lock.write_text("owner lock v2\n")
    with pytest.raises(FreezeError, match="lockfile"):
        verify_freeze(fz, config=config, selection=real_sel, support=support, require_signoff=False, lockfile=lock)
    drift = copy.deepcopy(fz); drift["environment"]["runtime"]["numpy"] = "0.0"
    with pytest.raises(FreezeError, match="runtime_sha256"):
        verify_freeze(drift, config=config, selection=real_sel, support=support, require_signoff=False)
    assert check_runtime(fz, "cpu")["hardware_matches"]
    wrong = copy.deepcopy(fz); wrong["environment"]["hardware_class"] = "A100"
    with pytest.raises(FreezeError, match="environment differs"):
        check_runtime(wrong, "cpu")
    assert check_runtime(wrong, "cpu", deviation="no A100 available this week")["deviation"]
    with pytest.raises(FreezeError):
        verify_freeze(fz, config={**config, "optimizer_steps": 4000}, selection=real_sel, support=support, require_signoff=False)
    bad = copy.deepcopy(fz); bad["approvals"][0]["name"] = "TBD"
    with pytest.raises(FreezeError):
        verify_freeze(bad, config=config, selection=real_sel, support=support, require_signoff=True)
    tampered = copy.deepcopy(real_sel); tampered["runs"][0]["ids"] = tampered["runs"][0]["ids"][:-1]
    tampered["selection_sha256_without_certificate"] = sha256_json({k: v for k, v in tampered.items() if k != "certificate"})
    with pytest.raises(FreezeError):
        verify_freeze(fz, config=config, selection=tampered, support=support, require_signoff=False)



def test_draft_freeze_is_refused_until_both_internal_roles_sign(selection, support, config, tmp_path, owner_repo):
    real_sel = copy.deepcopy(selection); real_sel["manifest_kind"] = "real"; real_sel["manifest_sha256"] = "testmanifest"
    lock = tmp_path / "uv.lock"; lock.write_text("owner lock\n")
    draft = build_freeze(config, tmp_path / "m.jsonl", real_sel, support, [], None, environment_block("A100", lock),
                         owner_pipeline_block(owner_repo))
    assert draft["status"] == "DRAFT - unsigned"
    with pytest.raises(FreezeError, match="missing sign-off"):
        verify_freeze(draft, config=config, selection=real_sel, support=support, require_signoff=True)
    half = sign(draft, "paper_lead", "Paper Lead", "2026-11-02")
    assert half["status"] == "DRAFT - unsigned"
    with pytest.raises(FreezeError):
        sign(half, "stats_check", "TBD", "2026-11-02")
    full = sign(half, "stats_check", "Stats Check", "2026-11-02")
    assert full["status"] == "signed"
    verify_freeze(full, config=config, selection=real_sel, support=support, require_signoff=True, lockfile=lock)


def test_freeze_cli_sign_and_verify(selection, support, config, tmp_path, owner_repo):
    """The CLI path: create (draft) -> sign twice -> verify with sign-offs; sign takes no config."""

    from registered.freeze import main as freeze_main

    real_sel = copy.deepcopy(selection); real_sel["manifest_kind"] = "real"; real_sel["manifest_sha256"] = "testmanifest"
    lock = tmp_path / "uv.lock"; lock.write_text("owner lock\n")
    fz = build_freeze(config, tmp_path / "m.jsonl", real_sel, support, [], None, environment_block("CPU", lock),
                      owner_pipeline_block(owner_repo))
    path = tmp_path / "freeze.json"; path.write_text(json.dumps(fz))
    freeze_main(["sign", "--freeze", str(path), "--role", "paper_lead", "--name", "Paper Lead", "--date", "2026-11-02"])
    freeze_main(["sign", "--freeze", str(path), "--role", "stats_check", "--name", "Stats Check", "--date", "2026-11-02"])
    signed = json.loads(path.read_text())
    assert signed["status"] == "signed"
    verify_freeze(signed, config=config, selection=real_sel, support=support, require_signoff=True, lockfile=lock)


def test_freeze_binds_the_owner_pipeline(selection, support, config, tmp_path, owner_repo):
    """v3.4 section A names the owner's pipeline commit; the freeze records it and the wm.py hash."""

    import subprocess

    real_sel = copy.deepcopy(selection); real_sel["manifest_kind"] = "real"; real_sel["manifest_sha256"] = "testmanifest"
    lock = owner_repo / "uv.lock"
    fz = build_freeze(config, tmp_path / "m.jsonl", real_sel, support, [], None, environment_block("CPU", lock),
                      owner_pipeline_block(owner_repo))
    assert len(fz["owner_pipeline"]["commit"]) == 40
    verify_freeze(fz, config=config, require_signoff=False, owner_repo=owner_repo)
    with pytest.raises(FreezeError, match="owner's pipeline"):
        verify_freeze({**fz, "owner_pipeline": None}, config=config, require_signoff=False)
    wm = owner_repo / "experiment" / "wm.py"
    original = wm.read_text()
    wm.write_text(original.replace('"cpu"', '"cuda"'))  # an edited model file is refused
    with pytest.raises(FreezeError, match="differs from the frozen file"):
        verify_freeze(fz, config=config, require_signoff=False, owner_repo=owner_repo)
    wm.write_text(original)
    (owner_repo / "README.md").write_text("later change\n")  # same wm.py, another commit
    git = ["git", "-C", str(owner_repo), "-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "later"], check=True)
    with pytest.raises(FreezeError, match="not at the frozen commit"):
        verify_freeze(fz, config=config, require_signoff=False, owner_repo=owner_repo)


def test_freeze_cli_create_reuses_a_recorded_environment(selection, support, config, tmp_path, owner_repo):
    """``create --environment`` reuses the block recorded on the registered hardware."""

    from registered.freeze import main as freeze_main

    real_sel = copy.deepcopy(selection); real_sel["manifest_kind"] = "real"; real_sel["manifest_sha256"] = "testmanifest"
    recorded = build_freeze(config, tmp_path / "m.jsonl", real_sel, support, [], None,
                            environment_block("A100", owner_repo / "uv.lock"), owner_pipeline_block(owner_repo))
    (tmp_path / "old.json").write_text(json.dumps(recorded))
    (tmp_path / "selection.json").write_text(json.dumps(real_sel)); (tmp_path / "support.json").write_text(json.dumps(support))
    out = tmp_path / "new.json"
    freeze_main(["create", "--manifest", str(tmp_path / "m.jsonl"), "--selection", str(tmp_path / "selection.json"),
                 "--support", str(tmp_path / "support.json"), "--owner-repo", str(owner_repo),
                 "--environment", str(tmp_path / "old.json"), "--out", str(out)])
    new = json.loads(out.read_text())
    assert new["environment"] == recorded["environment"] and new["environment"]["hardware_class"] == "A100"
    assert new["owner_pipeline"] == recorded["owner_pipeline"]
    with pytest.raises(SystemExit):  # exactly one source of the environment block
        freeze_main(["create", "--manifest", str(tmp_path / "m.jsonl"), "--selection", str(tmp_path / "selection.json"),
                     "--support", str(tmp_path / "support.json"), "--owner-repo", str(owner_repo), "--out", str(out)])
