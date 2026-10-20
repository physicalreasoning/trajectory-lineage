import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plan_exposure_power.py"
_SPEC = importlib.util.spec_from_file_location("plan_exposure_power", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_required_blocks_increases_with_variance() -> None:
    small = _MODULE.required_blocks(
        standard_deviation=0.0025,
        detectable_excess=0.002,
        alpha=0.05,
        power=0.9,
    )
    large = _MODULE.required_blocks(
        standard_deviation=0.005,
        detectable_excess=0.002,
        alpha=0.05,
        power=0.9,
    )

    assert large > small


def test_required_blocks_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        _MODULE.required_blocks(
            standard_deviation=0.0,
            detectable_excess=0.002,
            alpha=0.05,
            power=0.9,
        )
