import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import freeze  # noqa: E402

ANALYZER = Path(__file__).resolve().parents[1] / "scripts" / "analyze_confirmation.py"


def setup(tmp_path: Path, **config_overrides) -> tuple[Path, dict]:
    config = {"role": "confirmatory", "experiment_seed": 99, "excluded_pilot_seeds": [5]}
    config.update(config_overrides)
    config_path = tmp_path / "elsewhere" / "exposure_confirmation.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config))
    registry = tmp_path / "experiments"
    registry.mkdir()
    (registry / "exposure_pilot_v1.json").write_text(json.dumps({"experiment_seed": 7}))
    manifest = {
        "config_sha256": freeze.sha256(config_path),
        "code_commit": "frozen",
        "analyzer_sha256": freeze.sha256(ANALYZER),
        "signoffs": {"paper_lead": "Paper Lead", "stats_check": "Internal Statistician"},
        "signoff_date": "2026-10-20",
        "decision_statement": "A 0.001 shift decides whether a model is reported as better.",
        "excluded_pilot_seeds": [5, 7],
    }
    return config_path, manifest


def check(manifest, config_path, tmp_path, head="frozen", changed=None):
    freeze.validate(
        manifest,
        config_path,
        ANALYZER,
        state=lambda: (head, True),
        changes=lambda commit: changed,
        experiments=tmp_path / "experiments",
    )


def test_manifest_may_be_committed_under_approvals(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path)
    check(manifest, config_path, tmp_path)
    check(manifest, config_path, tmp_path, head="later", changed=["approvals/freeze.json"])
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path, head="later", changed=["scripts/analyze.py"])
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path, head="later", changed=None)


def test_required_roles_cannot_be_removed_by_configuration(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path, required_signoff_roles=["someone_else"])
    manifest["config_sha256"] = freeze.sha256(config_path)
    manifest["signoffs"] = {"someone_else": "Some Person"}
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path)


def test_pilot_registry_does_not_depend_on_configuration_location(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path, excluded_pilot_seeds=[])
    manifest["config_sha256"] = freeze.sha256(config_path)
    manifest["excluded_pilot_seeds"] = []
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path)


def test_confirmation_seed_cannot_reuse_a_pilot_seed(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path, experiment_seed=7)
    manifest["config_sha256"] = freeze.sha256(config_path)
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path)


def test_pilot_configuration_cannot_be_unblinded(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path, role="discarded_effect_blinded_variance_pilot")
    manifest["config_sha256"] = freeze.sha256(config_path)
    with pytest.raises(freeze.FreezeError):
        check(manifest, config_path, tmp_path)


def test_provisional_mode_requires_owner_and_label(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path)
    provisional = dict(
        manifest,
        mode="provisional",
        signoffs={"workstream_owner": "Lovejesh (workstream owner)"},
        decision_statement="Provisional freeze by the owner; collaborator review pending.",
    )
    check(provisional, config_path, tmp_path)
    with pytest.raises(freeze.FreezeError):
        check(
            dict(provisional, decision_statement="Frozen by owner before unblinding today."),
            config_path,
            tmp_path,
        )
    with pytest.raises(freeze.FreezeError):
        check(dict(provisional, signoffs={}), config_path, tmp_path)
    with pytest.raises(freeze.FreezeError):
        check(dict(provisional, mode="other"), config_path, tmp_path)


# --- environment block, several configurations per manifest, axis role ---------------------------


def env_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock v1\n")
    return repo


def check_env(manifest, config_path, tmp_path, repo):
    freeze.validate(
        manifest,
        config_path,
        ANALYZER,
        state=lambda: ("frozen", True),
        changes=lambda commit: None,
        experiments=tmp_path / "experiments",
        repository=repo,
    )


def test_environment_block_is_required_and_checked(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path, require_environment=True)
    repo = env_repo(tmp_path)
    with pytest.raises(freeze.FreezeError, match="environment block"):
        check_env(manifest, config_path, tmp_path, repo)
    good = freeze.environment_block("A100", lockfile=repo / "uv.lock")
    check_env({**manifest, "environment": good}, config_path, tmp_path, repo)
    (repo / "uv.lock").write_text("lock v2\n")
    with pytest.raises(freeze.FreezeError, match="lockfile"):
        check_env({**manifest, "environment": good}, config_path, tmp_path, repo)
    (repo / "uv.lock").write_text("lock v1\n")
    tampered = {**good, "runtime": {**good["runtime"], "numpy": "0.0"}}
    with pytest.raises(freeze.FreezeError, match="runtime_sha256"):
        check_env({**manifest, "environment": tampered}, config_path, tmp_path, repo)
    with pytest.raises(freeze.FreezeError, match="hardware class"):
        check_env(
            {**manifest, "environment": {**good, "hardware_class": "tbd"}},
            config_path,
            tmp_path,
            repo,
        )
    # provisional (1D) manifests and configurations without the flag are unaffected
    (tmp_path / "plain").mkdir()
    plain_path, plain = setup(tmp_path / "plain")
    check(plain, plain_path, tmp_path / "plain")


def test_one_manifest_binds_several_configurations_including_axes(tmp_path: Path) -> None:
    config_path, manifest = setup(tmp_path)
    axis = config_path.parent / "exposure2d_confirm_dose16.json"
    axis.write_text(json.dumps({"role": "confirmatory_axis", "experiment_seed": 101}))
    multi = {k: v for k, v in manifest.items() if k != "config_sha256"}
    multi["configs"] = {
        config_path.stem: freeze.sha256(config_path),
        axis.stem: freeze.sha256(axis),
    }
    check(multi, config_path, tmp_path)
    check(multi, axis, tmp_path)
    other = config_path.parent / "exposure2d_confirm_dose256.json"
    other.write_text(json.dumps({"role": "confirmatory_axis", "experiment_seed": 102}))
    with pytest.raises(freeze.FreezeError, match="does not bind"):
        check(multi, other, tmp_path)


def test_runtime_check_refuses_other_hardware_unless_a_deviation_is_logged() -> None:
    env = freeze.environment_block("CPU")
    record = freeze.check_runtime({"environment": env}, "cpu")
    assert record["checked"] and record["hardware_matches"] and not record["runtime_differences"]
    wrong = {**env, "hardware_class": "A100"}
    with pytest.raises(freeze.FreezeError, match="frozen class"):
        freeze.check_runtime({"environment": wrong}, "cpu")
    logged = freeze.check_runtime(
        {"environment": wrong}, "cpu", deviation="Colab had no A100 for a week"
    )
    assert logged["deviation"] and not logged["hardware_matches"]
    drift = {**env, "runtime": {**env["runtime"], "numpy": "0.0"}}
    with pytest.raises(freeze.FreezeError, match="runtime differs"):
        freeze.check_runtime({"environment": drift}, "cpu")
    assert freeze.check_runtime({}, "cpu")["checked"] is False


def test_environment_command_writes_a_valid_block(tmp_path: Path) -> None:
    out = tmp_path / "approvals" / "environment.json"
    freeze.main(["environment", "--hardware-class", "A100", "--out", str(out)])
    block = json.loads(out.read_text())
    assert block["hardware_class"] == "A100" and block["lockfile"] == "uv.lock"
    freeze.validate_environment(block)


def test_draft_binds_all_five_configurations_and_is_refused_until_signed() -> None:
    paths = [freeze.REPOSITORY / c for c in freeze.EXP3_CONFIGS]
    env = freeze.environment_block("CPU")
    draft = freeze.build_draft(paths, env, code_commit="abc123")
    assert draft["status"] == "DRAFT - unsigned" and len(draft["configs"]) == 5
    assert draft["analyzer"] == "scripts/analyze_2d.py"
    every_pilot = freeze.pilot_registry(freeze.EXPERIMENTS, {})
    assert every_pilot <= set(draft["excluded_pilot_seeds"])

    def run(manifest, path):
        freeze.validate(
            manifest,
            path,
            freeze.REPOSITORY / draft["analyzer"],
            state=lambda: ("abc123", True),
            changes=lambda commit: [],
        )

    for path in paths:
        with pytest.raises(freeze.FreezeError, match="sign-off"):
            run(draft, path)
    half = freeze.sign(draft, "paper_lead", "Paper Lead", "2026-11-02")
    assert half["status"] == "DRAFT - unsigned"
    with pytest.raises(freeze.FreezeError):
        freeze.sign(half, "stats_check", "tbd", "2026-11-02")
    signed = freeze.sign(half, "stats_check", "Stats Check", "2026-11-03")
    assert signed["status"] == "signed" and signed["signoff_date"] == "2026-11-03"
    for path in paths:
        run(signed, path)
    with pytest.raises(freeze.FreezeError, match="analyzers"):
        freeze.build_draft(
            [*paths, freeze.REPOSITORY / "experiments/exposure_confirmation.json"], env, "abc123"
        )
