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
        "signoffs": {"paper_lead": "Jiankai Sun", "evaluation_science": "Yining Hua"},
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
