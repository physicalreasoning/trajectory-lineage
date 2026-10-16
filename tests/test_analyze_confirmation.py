import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_confirmation as analyzer  # noqa: E402

REAL_CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "experiments" / "exposure_confirmation.json").read_text()
)


def config_for(blocks: int, minimum: int = 2) -> dict:
    config = copy.deepcopy(REAL_CONFIG)
    config["blocks"] = blocks
    config["decision_contract"]["minimum_screened_blocks"] = minimum
    return config


def records(
    *,
    s: float,
    h: float,
    q: float = 0.02,
    exact: float = 0.01,
    n: int = 64,
    sd: float = 0.0005,
    competent: bool = True,
    seed: int = 0,
    start: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    runs = []
    for offset in range(n):
        source = s + rng.normal(0, sd)
        harm = h + rng.normal(0, sd)
        b_src, b_ind = 0.05 + rng.normal(0, 0.001), 0.04 + rng.normal(0, 0.001)
        errors = {
            arm: {"source": 0.04, "exact": 0.04, "independent": 0.04} for arm in analyzer.ARMS
        }
        errors["B"] = {"source": b_src, "exact": b_src, "independent": b_ind}
        errors["C"] = {
            "source": b_src - source,
            "exact": b_src - source,
            "independent": b_ind + harm,
        }
        runs.append(
            {
                "block_index": start + offset,
                "status": "completed",
                "base_competent": competent,
                "errors": errors,
                "contrasts": {
                    "C_source_gain": source,
                    "C_independent_gain": -harm,
                    "C_independent_harm": harm,
                    "C_reuse_gap": (b_src - (b_src - source)) - (b_ind - (b_ind + harm)),
                    "C_minus_D_gap": q + rng.normal(0, sd),
                    "C_minus_D2_gap": q + rng.normal(0, sd),
                    "C_apparent_improvement": source,
                    "B_apparent_improvement": 0.0,
                    "E_exact_gain": exact + rng.normal(0, sd),
                    "E_transfer_loss": 0.001,
                },
            }
        )
    return runs


def test_consequential_reuse_advantage() -> None:
    result = analyzer.analyze(records(s=0.015, h=0.0), config_for(64))
    assert result["primary_claim"] == "consequential reuse advantage"
    assert result["primary"]["secondary_holm_rejections"]["identity_specific"]
    assert "B" in result["primary"]["arm_population_errors"]


def test_large_gap_driven_by_harm_is_not_relabeled() -> None:
    result = analyzer.analyze(records(s=-0.005, h=0.020), config_for(64))
    assert result["primary"]["gap_magnitude"] == "consequential"
    assert result["primary_claim"] == "differential gap without safeguards"
    assert result["primary"]["secondary_holm_rejections"]["consequential_harm"]


def test_negligible_requires_assay_sensitivity() -> None:
    sensitive = analyzer.analyze(records(s=0.0, h=0.0), config_for(64))
    insensitive = analyzer.analyze(records(s=0.0, h=0.0, exact=-0.01), config_for(64))
    assert sensitive["primary_claim"] == "negligible differential gap"
    assert insensitive["primary_claim"] == "assay insensitive"


def test_prevalence_dependent_and_break_even_interval() -> None:
    result = analyzer.analyze(records(s=0.005, h=0.0), config_for(64))
    assert result["primary_claim"] == "prevalence_dependent reuse advantage"
    interval = result["primary"]["break_even_prevalence"]
    assert interval["status"] == "finite"
    upper = result["primary"]["contrasts"]["G"]["upper"]
    assert interval["interval"][0] == pytest.approx(0.001 / upper)


def test_reverse_gap_and_no_positive_crossing() -> None:
    result = analyzer.analyze(records(s=-0.006, h=0.0), config_for(64))
    assert result["primary_claim"] == "reverse gap"
    assert result["primary"]["break_even_prevalence"]["status"] == "no positive crossing"


def test_execution_failures_make_result_inconclusive_but_keep_estimates() -> None:
    runs = records(s=0.015, h=0.0, n=40)
    for run in runs[:3]:
        run.update({"status": "algorithmic_failure", "error": "x"})
    result = analyzer.analyze(runs, config_for(40))
    assert result["primary_claim"].startswith("inconclusive (execution")
    assert result["all_completed_blocks"]["blocks"] == 37


def test_failure_sensitive_claim_is_downgraded() -> None:
    runs = records(s=0.0108, h=0.0, n=60, sd=0.0002)
    runs[1]["contrasts"]["C_source_gain"] = 0.0
    runs[1]["errors"]["C"]["source"] = runs[1]["errors"]["B"]["source"]
    runs[1]["contrasts"]["C_reuse_gap"] = runs[1]["contrasts"]["C_independent_harm"]
    runs[0].update({"status": "algorithmic_failure", "error": "x"})
    complete = [run for run in runs if run["status"] == "completed"]
    without_failure = analyzer.analyze_blocks(
        complete, config_for(60)["decision_contract"], np.random.default_rng(0)
    )
    result = analyzer.analyze(runs, config_for(60))
    assert result["primary"]["gap_magnitude"] == without_failure["gap_magnitude"]
    assert result["failure_robustness"] != [result["primary"]["primary_claim"]]
    assert result["primary_claim"] == "inconclusive (failure-sensitive)"


def test_screen_failures_are_excluded_from_primary_but_reported() -> None:
    runs = records(s=0.015, h=0.0, n=40) + records(
        s=0.015, h=0.0, n=2, competent=False, seed=3, start=40
    )
    result = analyzer.analyze(runs, config_for(42))
    assert result["primary"]["blocks"] == 40
    assert result["all_completed_blocks"]["blocks"] == 42


def test_duplicated_or_missing_blocks_are_rejected() -> None:
    runs = records(s=0.015, h=0.0, n=10)
    duplicated = runs + [copy.deepcopy(runs[0])]
    with pytest.raises(analyzer.RecordError):
        analyzer.analyze(duplicated, config_for(11))
    with pytest.raises(analyzer.RecordError):
        analyzer.analyze(runs[:9], config_for(10))


def test_inconsistent_or_nonfinite_records_are_rejected() -> None:
    runs = records(s=0.015, h=0.0, n=10)
    runs[2]["contrasts"]["C_reuse_gap"] += 0.01
    with pytest.raises(analyzer.RecordError):
        analyzer.analyze(runs, config_for(10))
    runs = records(s=0.015, h=0.0, n=10)
    runs[3]["contrasts"]["C_minus_D_gap"] = float("nan")
    with pytest.raises(analyzer.RecordError):
        analyzer.analyze(runs, config_for(10))


def test_real_contract_enforces_minimum_screened_blocks() -> None:
    config = copy.deepcopy(REAL_CONFIG)
    config["blocks"] = 64
    result = analyzer.analyze(records(s=0.015, h=0.0, n=64), config)
    assert result["primary_claim"] == "inconclusive (too few screened blocks)"


def test_fieller_interval_types() -> None:
    rng = np.random.default_rng(1)
    gap = 0.015 + rng.normal(0, 0.0005, 200)
    harm = 0.003 + rng.normal(0, 0.0005, 200)
    assert analyzer.fieller(harm, gap, 0.025)["type"] == "bounded"
    flat_gap = rng.normal(0, 0.0005, 200)
    assert analyzer.fieller(harm, flat_gap, 0.025)["type"] in {"two rays", "entire real line"}


def test_holm_is_step_down() -> None:
    decisions = analyzer.holm({"a": 0.001, "b": 0.02, "c": 0.03}, alpha=0.025)
    assert decisions == {"a": True, "b": False, "c": False}
