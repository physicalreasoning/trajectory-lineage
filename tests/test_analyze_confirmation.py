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
CONFIG = json.loads(json.dumps(REAL_CONFIG))
CONFIG["decision_contract"]["minimum_screened_blocks"] = 2


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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    runs = []
    for index in range(n):
        source = s + rng.normal(0, sd)
        harm = h + rng.normal(0, sd)
        runs.append(
            {
                "block_index": index,
                "status": "completed",
                "base_competent": competent,
                "contrasts": {
                    "C_source_gain": source,
                    "C_independent_gain": -harm,
                    "C_independent_harm": harm,
                    "C_reuse_gap": source + harm,
                    "C_minus_D_gap": q + rng.normal(0, sd),
                    "C_minus_D2_gap": q + rng.normal(0, sd),
                    "C_apparent_improvement": source,
                    "E_exact_gain": exact + rng.normal(0, sd),
                },
            }
        )
    return runs


def test_consequential_reuse_advantage() -> None:
    result = analyzer.analyze(records(s=0.015, h=0.0), CONFIG)
    assert result["primary_claim"] == "consequential reuse advantage"
    assert result["primary"]["secondary_holm_rejections"]["identity_specific"]


def test_large_gap_driven_by_harm_is_not_relabeled() -> None:
    result = analyzer.analyze(records(s=-0.005, h=0.020), CONFIG)
    assert result["primary"]["gap_magnitude"] == "consequential"
    assert result["primary_claim"] == "differential gap without safeguards"
    assert result["primary"]["secondary_holm_rejections"]["consequential_harm"]


def test_negligible_requires_assay_sensitivity() -> None:
    sensitive = analyzer.analyze(records(s=0.0, h=0.0), CONFIG)
    insensitive = analyzer.analyze(records(s=0.0, h=0.0, exact=-0.01), CONFIG)
    assert sensitive["primary_claim"] == "negligible differential gap"
    assert insensitive["primary_claim"] == "assay insensitive"


def test_prevalence_dependent_and_break_even_interval() -> None:
    result = analyzer.analyze(records(s=0.005, h=0.0), CONFIG)
    assert result["primary_claim"] == "prevalence_dependent reuse advantage"
    interval = result["primary"]["break_even_prevalence"]
    assert interval["status"] == "finite"
    assert interval["interval"][0] == pytest.approx(
        0.001 / result["primary"]["contrasts"]["G"]["upper"]
    )


def test_reverse_gap_and_no_positive_crossing() -> None:
    result = analyzer.analyze(records(s=-0.006, h=0.0), CONFIG)
    assert result["primary_claim"] == "reverse gap"
    assert result["primary"]["break_even_prevalence"]["status"] == "no positive crossing"


def test_execution_failures_make_result_inconclusive() -> None:
    runs = records(s=0.015, h=0.0, n=40)
    for run in runs[:3]:
        run.update({"status": "algorithmic_failure", "error": "x"})
    result = analyzer.analyze(runs, CONFIG)
    assert result["primary_claim"].startswith("inconclusive (execution")


def test_screen_failures_are_excluded_from_primary_but_kept_in_itt() -> None:
    runs = records(s=0.015, h=0.0, n=40) + records(s=0.015, h=0.0, n=2, competent=False, seed=3)
    result = analyzer.analyze(runs, CONFIG)
    assert result["primary"]["blocks"] == 40
    assert result["intent_to_treat"]["blocks"] == 42


def test_holm_is_step_down() -> None:
    decisions = analyzer.holm({"a": 0.001, "b": 0.02, "c": 0.03}, alpha=0.025)
    assert decisions == {"a": True, "b": False, "c": False}


def test_real_contract_enforces_minimum_screened_blocks() -> None:
    result = analyzer.analyze(records(s=0.015, h=0.0, n=64), REAL_CONFIG)
    assert result["primary_claim"] == "inconclusive (too few screened blocks)"
