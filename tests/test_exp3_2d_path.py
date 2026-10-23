"""Integration test of the registered 2D path: freeze manifest -> export identity -> analysis."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_2d import analyze_2d, check_identity  # noqa: E402
from dryrun_2d import SCENARIOS, synthetic_runs  # noqa: E402
from exposure_experiment import analyzer_for  # noqa: E402
from freeze import FreezeError, run_identity, sha256, validate  # noqa: E402

SDS = {"S": 0.005, "H": 0.001, "G": 0.0055, "Q": 0.004, "Q2": 0.005, "E": 0.0075}


def _setup(tmp_path, mode="confirmatory"):
    config = json.loads((ROOT / "experiments/exposure2d_confirmation.json").read_text())
    config["blocks"] = 40
    config["decision_contract"]["minimum_screened_blocks"] = 30
    cpath = tmp_path / "config.json"
    cpath.write_text(json.dumps(config))
    analyzer = analyzer_for(config)
    manifest = {
        "mode": mode,
        "config_sha256": sha256(cpath),
        "analyzer_sha256": sha256(analyzer),
        "code_commit": "abc123",
        "signoffs": {
            "paper_lead": "Jiankai Sun",
            "evaluation_science": "Yining Hua",
            "workstream_owner": "Lovejesh Bangar",
        },
        "signoff_date": "2026-10-29",
        "decision_statement": "Thresholds approved as registered for the 2D confirmation "
        "(synthetic test)." + (" provisional" if mode == "provisional" else ""),
        "excluded_pilot_seeds": sorted(config["excluded_pilot_seeds"]),
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    return config, cpath, manifest, mpath, analyzer


def _validate(manifest, cpath, analyzer, tmp_path):
    validate(
        manifest,
        cpath,
        analyzer,
        state=lambda: ("abc123", True),
        changes=lambda c: [],
        experiments=tmp_path,
    )


def test_runner_binds_the_2d_analyzer():
    config = json.loads((ROOT / "experiments/exposure2d_confirmation.json").read_text())
    assert analyzer_for(config).name == "analyze_2d.py"
    assert analyzer_for({}).name == "analyze_confirmation.py"


def test_provisional_freeze_is_rejected_for_2d(tmp_path):
    _, cpath, manifest, _, analyzer = _setup(tmp_path, mode="provisional")
    with pytest.raises(FreezeError, match="confirmatory approvals"):
        _validate(manifest, cpath, analyzer, tmp_path)


def test_full_path_synthetic(tmp_path):
    config, cpath, manifest, mpath, analyzer = _setup(tmp_path)
    _validate(manifest, cpath, analyzer, tmp_path)  # approvals present -> passes
    runs = synthetic_runs(
        np.random.default_rng(0), config, SCENARIOS["consequential_0.015"], SDS, 40, 1.0
    )
    export = {"identity": run_identity(mpath, cpath), "runs": runs}
    checked = check_identity(export, mpath, cpath)
    result = analyze_2d(checked, config, check_records=True)
    assert result["primary_claim"] == "consequential reuse advantage"
    # a tampered export is refused
    export["identity"]["config_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        check_identity(export, mpath, cpath)


def test_any_missing_block_makes_every_claim_inconclusive(tmp_path):
    config, *_ = _setup(tmp_path)
    truth = SCENARIOS["consequential_but_harmful_0.002"]
    runs = synthetic_runs(np.random.default_rng(1), config, truth, SDS, 40, 1.0)
    runs[3] = {"block_index": 3, "status": "algorithmic_failure", "error": "synthetic"}
    result = analyze_2d(runs, config, check_records=True)
    assert result["primary_claim"] == "inconclusive (unresolved block failures)"
    nested = result["primary"]
    assert nested["primary_claim"] == "inconclusive (unresolved block failures)"
    assert not any(nested["secondary_holm_rejections"].values())
    assert nested["axes"]["event_identity"] == "unresolved"


def test_confirmatory_2d_run_requires_approval():
    config = json.loads((ROOT / "experiments/exposure2d_confirmation.json").read_text())
    assert config["require_approval_to_run"] is True
    assert config["allow_provisional_freeze"] is False


def test_identity_label_follows_holm(tmp_path):
    config, *_ = _setup(tmp_path)
    truth = dict(SCENARIOS["consequential_0.015"], Q=0.0115)  # near the threshold
    for seed in range(20):
        runs = synthetic_runs(np.random.default_rng(seed), config, truth, SDS, 40, 1.0)
        p = analyze_2d(runs, config, check_records=False)["primary"]
        assert (p["axes"]["event_identity"] == "event_specific") == p["secondary_holm_rejections"][
            "identity_specific"
        ]


def test_too_few_screened_blocks_suppresses_every_decision(tmp_path):
    config, *_ = _setup(tmp_path)
    runs = synthetic_runs(
        np.random.default_rng(2), config, SCENARIOS["consequential_0.015"], SDS, 40, 1.0
    )
    for run in runs[:15]:  # 25 of 40 screened < minimum 30
        run["screen_base_error"] = 0.09
    result = analyze_2d(runs, config, check_records=True)
    assert result["primary_claim"] == "inconclusive (too few screened blocks)"
    nested = result["primary"]
    assert nested["primary_claim"] == "inconclusive (too few screened blocks)"
    assert not any(nested["secondary_holm_rejections"].values())
