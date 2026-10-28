import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exposure_experiment as experiment  # noqa: E402

TINY = {
    "experiment_seed": 11,
    "blocks": 2,
    "observed_frames": 8,
    "future_frames": 16,
    "image_size": 64,
    "frame_dt": 0.04,
    "base_train_examples": 16,
    "source_trajectories": 4,
    "heldout_renders_per_source": 2,
    "independent_eval_examples": 8,
    "batch_size": 8,
    "base_training_steps": 3,
    "continuation_steps": 2,
    "arm_examples_per_batch": 2,
    "replay_examples_per_batch": 6,
    "learning_rate": 0.001,
    "hidden_channels": 4,
    "error_cap": 1.0,
    "nearby": {"reservoir_size": 4000, "low_quantile": 0.001, "high_quantile": 0.01},
    "closer_nearby": {"low_fraction": 0.25, "high_fraction": 0.5, "box_multiplier": 3.0},
    "screen_examples": 8,
    "sd_upper_limit_confidence": 0.8,
}


def test_identical_content_control_gives_identical_arms() -> None:
    spec = experiment.spec_from(TINY)
    run = experiment.run_block(TINY, 0, spec, torch.device("cpu"), arm_content_override="B")
    reference = run["errors"]["B"]
    for arm in ("C", "D", "D2", "E"):
        assert run["errors"][arm] == pytest.approx(reference, abs=1e-6)


def test_blocks_use_distinct_data() -> None:
    spec = experiment.spec_from(TINY)
    first = experiment.run_block(TINY, 0, spec, torch.device("cpu"))
    second = experiment.run_block(TINY, 1, spec, torch.device("cpu"))
    assert first["base_errors"] != second["base_errors"]
    assert first["presentations_per_source"] == [1, 1]


def test_bounded_errors_cap_large_and_nonfinite_predictions() -> None:
    target = torch.zeros(2, 3, 2, 2)
    prediction = target.clone()
    prediction[0] = 5.0
    prediction[1, 0, 0, 0] = float("nan")
    errors = experiment.bounded_errors(prediction, target, cap=1.0)
    assert errors[0].item() == pytest.approx(1.0)
    assert errors[1].item() == pytest.approx(1.0 / 6.0)


def _fake_runs(shift: float) -> list[dict]:
    runs = []
    for index, value in enumerate([0.1, 0.3, 0.2, 0.4]):
        contrasts = {"C_reuse_gap": value + shift, "E_exact_gain": 1.0}
        runs.append(
            {
                "block_index": index,
                "status": "completed",
                "contrasts": contrasts,
                "base_errors": {"source": 0.03, "independent": 0.03},
                "presentations_per_source": [12, 13],
                "nearby_band": [0.01, 0.02],
                "nearby_distance_median": 0.015,
                "closer_distance_median": 0.005,
                "base_nearest_distance_median": 0.014,
                "closer_box_offset_max": 0.5,
                "replacement_distance_median": 0.1,
                "total_seconds": 1.0,
            }
        )
    return runs


def test_full_report_hides_contrast_locations() -> None:
    config = {"sd_upper_limit_confidence": 0.8}
    assert experiment.blinded_report(config, _fake_runs(0.0)) == experiment.blinded_report(
        config, _fake_runs(5.0)
    )


def test_failed_blocks_count_against_positive_control() -> None:
    runs = _fake_runs(0.0)[:2] + [
        {"block_index": 2, "status": "infrastructure_failure", "error": "boom"},
        {"block_index": 3, "status": "infrastructure_failure", "error": "boom"},
    ]
    report = experiment.blinded_report({"sd_upper_limit_confidence": 0.8}, runs)
    assert report["positive_control_passed"] is False
    assert len(report["infrastructure_failures"]) == 2


def test_unblinding_requires_complete_manifest(tmp_path: Path, monkeypatch) -> None:
    config = dict(
        TINY, role="confirmatory", required_signoff_roles=["paper_lead", "stats_check"]
    )
    config["excluded_pilot_seeds"] = [5]
    config_path = tmp_path / "exposure_confirmation.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "exposure_pilot_v9.json").write_text(json.dumps({"experiment_seed": 7}))
    monkeypatch.setattr(experiment, "git_state", lambda: ("abc123", True))
    monkeypatch.setattr(experiment, "changed_since", lambda commit: None)
    import freeze

    monkeypatch.setattr(freeze, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(
        freeze.validate,
        "__kwdefaults__",
        dict(freeze.validate.__kwdefaults__, experiments=tmp_path),
    )
    valid = {
        "config_sha256": experiment.config_hash(config_path),
        "code_commit": "abc123",
        "analyzer_sha256": experiment.config_hash(experiment.ANALYZER),
        "signoffs": {"paper_lead": "Paper Lead", "stats_check": "Internal Statistician"},
        "signoff_date": "2026-10-20",
        "decision_statement": "A 0.001 shift changes whether we report one model as better.",
        "excluded_pilot_seeds": [5, 7],
    }
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps(valid))
    experiment.check_freeze(manifest, config_path)
    for key, bad in (
        ("config_sha256", "wrong"),
        ("code_commit", "other"),
        ("analyzer_sha256", "wrong"),
        ("signoffs", {"paper_lead": "full name", "stats_check": "Internal Statistician"}),
        ("signoff_date", "soon"),
        ("decision_statement", "ok"),
        ("excluded_pilot_seeds", [5]),
    ):
        manifest.write_text(json.dumps(dict(valid, **{key: bad})))
        with pytest.raises(SystemExit):
            experiment.check_freeze(manifest, config_path)
    monkeypatch.setattr(experiment, "git_state", lambda: ("abc123", False))
    manifest.write_text(json.dumps(valid))
    with pytest.raises(SystemExit):
        experiment.check_freeze(manifest, config_path)
    monkeypatch.setattr(experiment, "git_state", lambda: ("abc123", True))
    config_path.write_text(json.dumps(dict(config, role="discarded_effect_blinded_variance_pilot")))
    manifest.write_text(json.dumps(dict(valid, config_sha256=experiment.config_hash(config_path))))
    with pytest.raises(SystemExit):
        experiment.check_freeze(manifest, config_path)


def test_failure_policy_classifies_and_retries(monkeypatch) -> None:
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(experiment, "run_block", flaky)
    run = experiment.run_with_policy(TINY, 0, None, None)
    assert run["status"] == "infrastructure_failure" and calls["count"] == 2

    def broken(*args, **kwargs):
        raise ValueError("bad sampler")

    monkeypatch.setattr(experiment, "run_block", broken)
    run = experiment.run_with_policy(TINY, 0, None, None)
    assert run["status"] == "algorithmic_failure" and len(run["attempts"]) == 1


def test_item_losses_are_retained_per_render() -> None:
    spec = experiment.spec_from(TINY)
    run = experiment.run_block(TINY, 0, spec, torch.device("cpu"))
    renders = TINY["source_trajectories"] * TINY["heldout_renders_per_source"]
    assert len(run["item_losses"]["C"]["source"]) == renders
    assert len(run["item_losses"]["B"]["independent"]) == TINY["independent_eval_examples"]
    assert run["source_family_of_render"][:2] == [0, 0]
    assert run["errors"]["C"]["source"] == pytest.approx(
        sum(run["item_losses"]["C"]["source"]) / renders, rel=1e-5
    )
