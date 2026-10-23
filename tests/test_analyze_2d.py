import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_2d import classify_gap_2d, primary_claim  # noqa: E402

C = {"consequential_threshold": 0.010, "negligible_bound": 0.002}


def g(lo, hi):
    return {"lower": lo, "upper": hi}


def test_prevalence_dependent_requires_whole_interval_below_threshold():
    assert classify_gap_2d(g(0.004, 0.009), g(0.005, 0.008), C) == "prevalence_dependent"
    # the review's counterexample: [0.004, 0.020] is not prevalence-dependent
    assert classify_gap_2d(g(0.004, 0.020), g(0.005, 0.018), C) == "detectable_magnitude_unresolved"


def test_consequential_negligible_reverse_inconclusive():
    assert classify_gap_2d(g(0.011, 0.02), g(0.012, 0.019), C) == "consequential"
    assert (
        classify_gap_2d(g(-0.0025, 0.0025), g(-0.0015, 0.0015), C) == "negligible"
    )  # 90% decides absence
    assert classify_gap_2d(g(-0.01, -0.003), g(-0.009, -0.004), C) == "reverse"
    assert classify_gap_2d(g(-0.001, 0.004), g(-0.0005, 0.0035), C) == "inconclusive"


def test_claims_combine_axes():
    assert primary_claim("consequential", True, True, True) == "consequential reuse advantage"
    assert (
        primary_claim("consequential", True, False, True) == "differential gap without safeguards"
    )
    assert primary_claim("negligible", True, True, False) == "positive control not detected"
    assert primary_claim("negligible", False, False, True) == "negligible differential gap"
