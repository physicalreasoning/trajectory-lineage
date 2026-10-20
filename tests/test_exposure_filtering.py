import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exposure_filtering as filtering  # noqa: E402


def run() -> dict:
    return {
        "item_losses": {
            "B": {"source": [0.05, 0.05, 0.05, 0.05], "independent": [0.04, 0.02, 0.04, 0.02]},
            "C": {"source": [0.01, 0.01, 0.03, 0.03], "independent": [0.04, 0.02, 0.04, 0.02]},
        }
    }


def test_unfiltered_gap_and_prevalence_weighting() -> None:
    result = filtering.residual(run(), np.ones(4, bool), np.ones(4, bool))
    assert result["G"] == pytest.approx(0.03)
    assert result["pi_tau"] == pytest.approx(0.10)
    assert result["D"] == pytest.approx(0.003)


def test_partial_filtering_changes_prevalence_and_gap() -> None:
    keep_source = np.array([False, False, True, True])
    result = filtering.residual(run(), keep_source, np.ones(4, bool))
    assert result["G"] == pytest.approx(0.02)
    expected_pi = 0.1 * 0.5 / (0.1 * 0.5 + 0.9)
    assert result["pi_tau"] == pytest.approx(expected_pi)
    assert result["D"] == pytest.approx(expected_pi * 0.02)


def test_no_retained_source_contributes_zero_distortion() -> None:
    result = filtering.residual(run(), np.zeros(4, bool), np.ones(4, bool))
    assert result["G"] is None
    assert result["D"] == 0.0


def test_difficulty_shift_tracks_removed_hard_items() -> None:
    keep_independent = np.array([False, True, False, True])
    result = filtering.residual(run(), np.ones(4, bool), keep_independent)
    assert result["difficulty_shift"] == pytest.approx(0.02 - 0.03)
