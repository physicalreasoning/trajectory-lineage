"""Predictor capacity as a registered block factor: schedule, runner records, analyzer, dry run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_2d import analyze_2d, welch  # noqa: E402
from analyze_confirmation import RecordError  # noqa: E402
from dryrun_2d import SCENARIOS, dispersion_from, level_truth, synthetic_runs  # noqa: E402

from trajectory_lineage.exposure import capacity_levels, capacity_schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments" / "exposure2d_confirmation.json").read_text())
PILOT_REPORT = ROOT / "results" / "exposure2d_pilot_v1_blinded.json"


def test_schedule_counts_interleaving_and_determinism():
    config = {"blocks": 160, "hidden_channels": 16, "capacity_levels": [16, 64, 256], "capacity_blocks": {"16": 80, "64": 40, "256": 40}}
    schedule = capacity_schedule(config)
    assert len(schedule) == 160 and schedule == capacity_schedule(config)
    assert {level: schedule.count(level) for level in (16, 64, 256)} == {16: 80, 64: 40, 256: 40}
    # any prefix is balanced in proportion (within one block per level)
    for cut in (40, 80, 120):
        prefix = schedule[:cut]
        assert abs(prefix.count(16) - cut / 2) <= 1 and abs(prefix.count(64) - cut / 4) <= 1
    assert capacity_schedule({"blocks": 7, "hidden_channels": 16}) == [16] * 7
    assert capacity_levels({"hidden_channels": 32}) == [32]


def test_schedule_rejects_inconsistent_configuration():
    with pytest.raises(ValueError):
        capacity_schedule({"blocks": 10, "hidden_channels": 16, "capacity_levels": [16, 64], "capacity_blocks": {"16": 4, "64": 4}})
    with pytest.raises(ValueError):
        capacity_schedule({"blocks": 10, "hidden_channels": 16, "capacity_levels": [16, 64], "capacity_blocks": {"16": 10}})
    with pytest.raises(ValueError):
        capacity_levels({"hidden_channels": 16, "capacity_levels": [16, 16]})


def test_registered_confirmation_schedule_matches_counts():
    schedule = capacity_schedule(CONFIG)
    assert len(schedule) == CONFIG["blocks"] == 1600
    assert {level: schedule.count(level) for level in CONFIG["capacity_levels"]} == {16: 800, 64: 400, 256: 400}
    assert CONFIG["decision_contract"]["primary_capacity"] == 16


def _small_config(blocks=120):
    cfg = dict(CONFIG, blocks=blocks, capacity_blocks={"16": blocks // 2, "64": blocks // 4, "256": blocks // 4})
    cfg["decision_contract"] = dict(CONFIG["decision_contract"], minimum_screened_blocks=int(0.9 * blocks // 2),
                                    minimum_screened_blocks_by_capacity={"64": int(0.9 * blocks // 4), "256": int(0.9 * blocks // 4)})
    return cfg


@pytest.mark.skipif(not PILOT_REPORT.exists(), reason="pilot report not available")
def test_analyzer_detects_capacity_dependence_and_independence():
    sds = dispersion_from(PILOT_REPORT)
    cfg = _small_config(240)
    rng = np.random.default_rng(7)
    dep = synthetic_runs(rng, cfg, SCENARIOS["capacity_dependent_x2_x3"], sds, 240, 1.0)
    result = analyze_2d(dep, cfg, check_records=True)
    assert result["primary_capacity"] == 16 and result["screened"] == 120
    assert result["capacity"]["label"] == "capacity_dependent"
    assert result["capacity"]["trend"]["ci95"]["lower"] > 0
    assert set(result["primary"]["secondary_holm_rejections"]) >= {"identity_specific", "detectable", "consequential_harm", "capacity_increase"}
    flat = synthetic_runs(np.random.default_rng(7), cfg, SCENARIOS["capacity_flat_consequential"], sds, 240, 1.0)
    result = analyze_2d(flat, cfg, check_records=True)
    cap = result["capacity"]
    assert cap["label"] != "capacity_dependent" and not result["primary"]["secondary_holm_rejections"]["capacity_increase"]
    assert abs(cap["trend"]["ci95"]["mean"]) < 0.002  # flat truth: no trend beyond the negligible bound
    assert result["primary"]["axes"]["magnitude"] == "consequential"  # primary level unaffected by the factor


@pytest.mark.skipif(not PILOT_REPORT.exists(), reason="pilot report not available")
def test_primary_claim_uses_primary_level_only_and_failures_suppress_capacity():
    sds = dispersion_from(PILOT_REPORT)
    cfg = _small_config(120)
    runs = synthetic_runs(np.random.default_rng(1), cfg, SCENARIOS["capacity_dependent_x2_x3"], sds, 120, 1.0)
    result = analyze_2d(runs, cfg, check_records=True)
    assert result["primary"]["blocks"] == 60  # only 16-channel blocks
    runs[5] = {"block_index": 5, "status": "infrastructure_failure", "error": "cuda", "attempts": ["cuda"]}
    result = analyze_2d(runs, cfg, check_records=True)
    assert result["primary_claim"].startswith("inconclusive")
    assert result["capacity"]["label"].startswith("unresolved")


@pytest.mark.skipif(not PILOT_REPORT.exists(), reason="pilot report not available")
def test_record_with_wrong_capacity_is_rejected():
    sds = dispersion_from(PILOT_REPORT)
    cfg = _small_config(120)
    runs = synthetic_runs(np.random.default_rng(2), cfg, SCENARIOS["no_reuse_effect"], sds, 120, 1.0)
    runs[0]["hidden_channels"] = 999
    with pytest.raises(RecordError):
        analyze_2d(runs, cfg, check_records=True)


def test_level_truth_and_welch():
    truth = dict(S=0.006, H=0.0, Q=0.0, E=0.004, capacity_multipliers={"64": 2.0}, capacity_add={"256": 0.01})
    assert level_truth(truth, 16, 16)["S"] == 0.006
    assert level_truth(truth, 64, 16)["S"] == pytest.approx(0.012)
    assert level_truth(truth, 256, 16)["S"] == pytest.approx(0.016)
    a, b = np.array([1.0, 1.2, 0.9, 1.1, 1.0]), np.array([0.5, 0.4, 0.6, 0.5, 0.5])
    w = welch(a, b, 0.025)
    assert w["mean"] == pytest.approx(0.54) and w["lower"] > 0 and w["p_one_sided"] < 0.01
    same = welch(a, a, 0.025)
    assert same["mean"] == 0 and same["lower"] < 0 < same["upper"]


def test_checkpoint_round_trip_and_refusals(tmp_path):
    from exposure_experiment import append_checkpoint, load_checkpoint

    path = tmp_path / "raw.jsonl"
    assert load_checkpoint(path, "abc") == []
    append_checkpoint(path, "abc", {"block_index": 0, "status": "completed"})
    append_checkpoint(path, "abc", {"block_index": 2, "status": "infrastructure_failure"})
    assert [run["block_index"] for run in load_checkpoint(path, "abc")] == [0, 2]
    with pytest.raises(SystemExit):
        load_checkpoint(path, "other-config")
    append_checkpoint(path, "abc", {"block_index": 0, "status": "completed"})
    with pytest.raises(SystemExit):
        load_checkpoint(path, "abc")


def test_blinded_record_hides_locations_but_keeps_dispersion_and_flag(tmp_path):
    from exposure_experiment import blind_record, blinded_report, load_blinding_key, ARMS, POPULATIONS
    from trajectory_lineage.exposure import reuse_contrasts

    zero = {arm: dict.fromkeys(POPULATIONS, 0.0) for arm in ARMS}
    key = load_blinding_key(tmp_path / "key.json", sorted(reuse_contrasts(zero)))
    assert key == load_blinding_key(tmp_path / "key.json", sorted(reuse_contrasts(zero)))  # stable on reload
    rng = np.random.default_rng(0)
    runs = []
    for b in range(6):
        errors = {arm: {pop: float(0.05 + 0.01 * rng.random()) for pop in POPULATIONS} for arm in ARMS}
        errors["E"]["exact"] = 0.01  # arm E clearly wins on its own renders
        runs.append({"block_index": b, "status": "completed", "errors": errors, "item_losses": {}, "source_family_of_render": [],
                     "contrasts": reuse_contrasts(errors), "base_errors": {"source": 0.05, "independent": 0.05}, "base_competent": True,
                     "presentations_per_source": [12, 13], "nearby_band": [0.01, 0.02], "nearby_distance_median": 0.015,
                     "closer_distance_median": 0.005, "base_nearest_distance_median": 0.014, "closer_box_offset_max": 0.5,
                     "replacement_distance_median": 0.1, "total_seconds": 1.0, "hidden_channels": 16, "model_parameters": 1})
    blinded = [blind_record(run, key) for run in runs]
    assert all("errors" not in r and "item_losses" not in r and r["blinded"] for r in blinded)
    for name in runs[0]["contrasts"]:
        raw = np.array([r["contrasts"][name] for r in runs])
        shifted = np.array([r["contrasts"][name] for r in blinded])
        assert np.allclose(shifted - raw, key[name])  # locations are offset by the secret key
    config = {"sd_upper_limit_confidence": 0.8}
    a, b = blinded_report(config, runs), blinded_report(config, blinded)
    assert a["dispersion"] == b["dispersion"] and a["positive_control_passed"] == b["positive_control_passed"] is True


def test_closer_neighbors_are_worker_count_invariant():
    import json as _json
    from exposure_experiment import spec_from
    from trajectory_lineage.exposure import nearest_distances, sample_closer_neighbors, sample_event

    config = _json.loads((ROOT / "experiments" / "exposure2d_capacity_pilot_v1.json").read_text())
    spec = spec_from(config)
    rng = np.random.default_rng(3)
    base = [sample_event(rng, spec) for _ in range(64)]
    sources = [sample_event(rng, spec) for _ in range(3)]
    ref = nearest_distances(sources, base)
    kw = dict(min_distances=[0.25 * r for r in ref], max_distances=[0.5 * r for r in ref], box_multiplier=2.5)
    one = sample_closer_neighbors(np.random.default_rng(11), sources, spec, workers=1, **kw)
    two = sample_closer_neighbors(np.random.default_rng(11), sources, spec, workers=3, **kw)
    assert [n.state for n, _, _ in one] == [n.state for n, _, _ in two]
    assert all(lo <= d <= hi for (_, d, _), lo, hi in zip(one, kw["min_distances"], kw["max_distances"], strict=True))


def test_parallel_workers_do_not_change_results(tmp_path):
    """Blocks are seeded independently, so the blinded dispersion is identical for any worker count."""

    import json as _json
    import subprocess, sys as _sys

    cfg = _json.loads((ROOT / "experiments" / "exposure2d_pilot_v1.json").read_text())
    cfg.update(blocks=3, base_train_examples=16, source_trajectories=4, heldout_renders_per_source=2, independent_eval_examples=8,
               batch_size=8, base_training_steps=3, continuation_steps=2, arm_examples_per_batch=2, replay_examples_per_batch=6,
               hidden_channels=4, screen_examples=8)
    cfg["nearby"] = dict(cfg["nearby"], reservoir_size=4000)
    path = tmp_path / "cfg.json"; path.write_text(_json.dumps(cfg))
    outs = []
    for workers in (1, 2):
        out = tmp_path / f"out{workers}.json"
        subprocess.run([_sys.executable, str(ROOT / "scripts" / "exposure_experiment.py"), "--config", str(path), "--out", str(out), "--device", "cpu", "--workers", str(workers)],
                       check=True, capture_output=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        outs.append(_json.loads(out.read_text()))
    a, b = outs
    assert a["blocks_completed"] == b["blocks_completed"] == 3
    for name in a["dispersion"]:
        assert abs(a["dispersion"][name]["standard_deviation"] - b["dispersion"][name]["standard_deviation"]) < 1e-9


@pytest.mark.skipif(not PILOT_REPORT.exists(), reason="pilot report not available")
def test_joint_axes_analysis_labels_and_family(tmp_path):
    """Joint Holm family across the main confirmation and the architecture/dose axes; axis failures never touch the primary."""

    from analyze_confirmation import validate_records
    from analyze_exp3_axes import AXES, analyze_axes
    from trajectory_lineage.exposure import scaled_capacity_config

    main_cfg = scaled_capacity_config(CONFIG, 80)
    main_cfg["decision_contract"] = dict(CONFIG["decision_contract"], minimum_screened_blocks=30, minimum_screened_blocks_by_capacity={"64": 15, "256": 15})
    sds = dispersion_from(PILOT_REPORT)
    rng = np.random.default_rng(5)
    main_runs = validate_records(synthetic_runs(rng, main_cfg, dict(S=0.015, H=0.0, Q=0.0, E=0.004), sds, 80, 1.0), 80, main_cfg)
    main_result = analyze_2d(main_runs, main_cfg, check_records=False)
    axes = {}
    truths = {"latent_dinov2": 0.004, "latent_vjepa2": 0.0, "dose16": 0.015, "dose256": 0.004}
    for name in AXES:
        cfg = json.loads((ROOT / "experiments" / f"exposure2d_confirm_{name}.json").read_text())
        cfg = dict(cfg, blocks=40); cfg["decision_contract"] = dict(cfg["decision_contract"], minimum_screened_blocks=30)
        axis_sds = dispersion_from(ROOT / "results" / f"{name}_pilot_v1_blinded.json")
        runs = synthetic_runs(rng, cfg, dict(S=truths[name], H=0.0, Q=0.0, E=0.004), axis_sds, 40, 1.0)
        axes[name] = (validate_records(runs, 40, cfg), cfg)
    joint = analyze_axes({"result": main_result, "config": main_cfg}, axes)
    assert set(joint["joint_family_p_values"]) == {"identity_specific", "detectable", "consequential_harm", "capacity_increase",
                                                   "detectable_latent_dinov2", "detectable_latent_vjepa2", "dose_increase"}
    assert joint["labels"]["latent_dinov2"] == "reuse_detectable_under_recipe"
    assert joint["labels"]["latent_vjepa2"] == "negligible_under_recipe"
    assert joint["labels"]["dose"] == "dose_dependent"
    assert joint["main_primary_claim"] == main_result["primary_claim"] == "consequential reuse advantage"
    # an axis with a failed block drops out of the family and is unresolved; the primary is untouched
    broken = dict(axes); runs, cfg = broken["dose16"]; runs = list(runs); runs[0] = {"block_index": 0, "status": "infrastructure_failure", "error": "x", "attempts": ["x"]}
    broken["dose16"] = (runs, cfg)
    joint2 = analyze_axes({"result": main_result, "config": main_cfg}, broken)
    assert "dose_increase" not in joint2["joint_family_p_values"] and joint2["labels"]["dose"] == "unresolved"
    assert joint2["main_primary_claim"] == main_result["primary_claim"]
