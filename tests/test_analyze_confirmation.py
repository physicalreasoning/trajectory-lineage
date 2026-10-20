import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_confirmation as analyzer  # noqa: E402

LENIENT = {"strict_items": False}

REAL_CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "experiments" / "exposure_confirmation.json").read_text()
)


def analyze(runs, config):
    return analyzer.analyze(runs, config, strict_items=False)


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
    """Synthetic blocks whose component errors imply the requested contrasts."""

    rng = np.random.default_rng(seed)
    runs = []
    for offset in range(n):
        source = s + rng.normal(0, sd)
        harm = h + rng.normal(0, sd)
        identity = q + rng.normal(0, sd)
        b_src, b_ind = 0.06 + rng.normal(0, 0.001), 0.05 + rng.normal(0, 0.001)
        c_src, c_ind = b_src - source, b_ind + harm
        errors = {
            "B": {"source": b_src, "exact": b_src, "independent": b_ind},
            "C": {"source": c_src, "exact": c_src, "independent": c_ind},
            "D": {"source": c_src + identity, "exact": c_src, "independent": c_ind},
            "D2": {"source": c_src + identity, "exact": c_src, "independent": c_ind},
            "E": {
                "source": b_src,
                "exact": b_src - (exact + rng.normal(0, sd)),
                "independent": b_ind,
            },
        }
        runs.append(
            {
                "block_index": start + offset,
                "status": "completed",
                "errors": errors,
                "screen_base_error": 0.03 if competent else 0.06,
                "screen_constant_velocity_error": 0.05,
                "contrasts": {},
            }
        )
    return runs


def test_consequential_reuse_advantage() -> None:
    result = analyze(records(s=0.015, h=0.0), config_for(64))
    assert result["primary_claim"] == "consequential reuse advantage"
    assert result["primary"]["secondary_holm_rejections"]["identity_specific"]
    assert "B" in result["primary"]["arm_population_errors"]


def test_large_gap_driven_by_harm_is_not_relabeled() -> None:
    result = analyze(records(s=-0.005, h=0.020), config_for(64))
    assert result["primary"]["gap_magnitude"] == "consequential"
    assert result["primary_claim"] == "differential gap without safeguards"
    assert result["primary"]["secondary_holm_rejections"]["consequential_harm"]


def test_negligible_requires_assay_sensitivity() -> None:
    sensitive = analyze(records(s=0.0, h=0.0), config_for(64))
    insensitive = analyze(records(s=0.0, h=0.0, exact=-0.01), config_for(64))
    assert sensitive["primary_claim"] == "negligible differential gap"
    assert insensitive["primary_claim"] == "assay insensitive"


def test_prevalence_dependent_and_break_even_interval() -> None:
    result = analyze(records(s=0.005, h=0.0), config_for(64))
    assert result["primary_claim"] == "prevalence_dependent reuse advantage"
    interval = result["primary"]["break_even_prevalence"]
    assert interval["status"] == "finite"
    upper = result["primary"]["contrasts"]["G"]["upper"]
    assert interval["interval"][0] == pytest.approx(0.001 / upper)


def test_reverse_gap_and_no_positive_crossing() -> None:
    result = analyze(records(s=-0.006, h=0.0), config_for(64))
    assert result["primary_claim"] == "reverse gap"
    assert result["primary"]["break_even_prevalence"]["status"] == "no positive crossing"


def test_any_unresolved_failure_blocks_decisive_claims() -> None:
    runs = records(s=0.015, h=0.0, n=40)
    runs[0] = {"block_index": 0, "status": "algorithmic_failure", "error": "x"}
    result = analyze(runs, config_for(40))
    assert result["primary"]["primary_claim"] == "consequential reuse advantage"
    assert result["primary_claim"] == "inconclusive (unresolved block failures)"
    assert result["all_completed_blocks"]["blocks"] == 39


def test_tampered_contrasts_and_screen_flags_are_recomputed() -> None:
    runs = records(s=-0.005, h=0.020, n=64)
    for run in runs:
        run["contrasts"] = {"C_source_gain": 0.02, "C_independent_harm": 0.0}
        run["base_competent"] = "false"
    result = analyze(runs, config_for(64))
    assert result["primary_claim"] == "differential gap without safeguards"
    assert result["screened"] == 64


def test_screen_failures_are_excluded_from_primary_but_reported() -> None:
    runs = records(s=0.015, h=0.0, n=40) + records(
        s=0.015, h=0.0, n=2, competent=False, seed=3, start=40
    )
    result = analyze(runs, config_for(42))
    assert result["primary"]["blocks"] == 40
    assert result["all_completed_blocks"]["blocks"] == 42


def test_duplicated_or_missing_blocks_are_rejected() -> None:
    runs = records(s=0.015, h=0.0, n=10)
    duplicated = runs + [copy.deepcopy(runs[0])]
    with pytest.raises(analyzer.RecordError):
        analyze(duplicated, config_for(11))
    with pytest.raises(analyzer.RecordError):
        analyze(runs[:9], config_for(10))


def test_invalid_errors_and_item_losses_are_rejected() -> None:
    runs = records(s=0.015, h=0.0, n=10)
    runs[2]["errors"]["C"]["source"] = 1.5
    with pytest.raises(analyzer.RecordError):
        analyze(runs, config_for(10))
    runs = records(s=0.015, h=0.0, n=10)
    runs[3]["errors"]["D"]["source"] = float("nan")
    with pytest.raises(analyzer.RecordError):
        analyze(runs, config_for(10))
    runs = records(s=0.015, h=0.0, n=10)
    runs[4]["item_losses"] = {"C": {"source": [0.9, 0.9]}}
    with pytest.raises(analyzer.RecordError):
        analyze(runs, config_for(10))


def test_real_contract_enforces_minimum_screened_blocks() -> None:
    config = copy.deepcopy(REAL_CONFIG)
    config["blocks"] = 64
    result = analyze(records(s=0.015, h=0.0, n=64), config)
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


def strict_records(n: int = 4) -> tuple[list[dict], dict]:
    config = config_for(n)
    config.update(source_trajectories=2, heldout_renders_per_source=2, independent_eval_examples=3)
    runs = records(s=0.015, h=0.0, n=n)
    for run in runs:
        run["item_losses"] = {
            arm: {
                "source": [run["errors"][arm]["source"]] * 4,
                "exact": [run["errors"][arm]["exact"]] * 2,
                "independent": [run["errors"][arm]["independent"]] * 3,
            }
            for arm in analyzer.ARMS
        }
        run["source_family_of_render"] = [0, 0, 1, 1]
    return runs, config


def test_strict_item_schema_is_enforced() -> None:
    runs, config = strict_records()
    analyzer.analyze(runs, config)
    for mutate in (
        lambda run: run["item_losses"]["C"].pop("source"),
        lambda run: run["item_losses"]["C"]["source"].append(0.0),
        lambda run: run["item_losses"]["B"]["exact"].__setitem__(0, float("nan")),
        lambda run: run.__setitem__("source_family_of_render", [0, 1, 0, 1]),
        lambda run: run.__setitem__("screen_base_error", -0.1),
    ):
        broken, _ = strict_records()
        mutate(broken[1])
        with pytest.raises(analyzer.RecordError):
            analyzer.analyze(broken, config)


def test_secondary_claims_are_qualified_after_any_failure() -> None:
    runs = records(s=-0.005, h=0.020, n=40)
    runs[0] = {"block_index": 0, "status": "algorithmic_failure", "error": "x"}
    result = analyze(runs, config_for(40))
    assert result["primary_claim"] == "differential gap without safeguards"
    assert "secondary_qualified" in result


def test_fieller_degenerate_denominator() -> None:
    harm = np.full(50, 0.002)
    zero = np.zeros(50)
    assert analyzer.fieller(harm, zero, 0.025)["type"].startswith("undefined")
