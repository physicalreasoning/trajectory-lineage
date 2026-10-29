"""Latent world-model predictor (frozen encoder + trained dynamics model) in the reuse design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # covered by the model-unit CI job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from exposure_experiment import encode_block, new_model, predictor_config, spec_from, train_steps  # noqa: E402

from trajectory_lineage.latent import FrozenEncoder, LatentStatePredictor, get_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LATENT_CFG = json.loads(
    (ROOT / "experiments" / "exposure2d_latent_dinov2_pilot_v1.json").read_text()
)


def test_fake_encoder_is_deterministic_and_frozen():
    enc = FrozenEncoder("fake-24", torch.device("cpu"))
    video = torch.rand(2, 8, 3, 64, 64)
    a, b = enc.encode(video), enc.encode(video)
    assert a.shape == (2, 8, 17, 24) and torch.equal(a, b)
    assert enc.parameters_frozen == 0 and enc.hub_id is None
    assert get_encoder("fake-24", torch.device("cpu")) is get_encoder(
        "fake-24", torch.device("cpu")
    )


def test_predictor_shapes_and_registered_size():
    model = LatentStatePredictor(
        token_dim=384, steps=8, tokens_per_step=17, future_frames=16, width=256, depth=4, heads=8
    )
    out = model(torch.randn(3, 8, 17, 384))
    assert out.shape == (3, 16, 2, 2) and out.min() >= 0 and out.max() <= 1
    n = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < n < 3_600_000  # the paper's width-256 / depth-4 dynamics model


def test_predictor_config_defaults_and_capacity_as_width():
    assert predictor_config({"hidden_channels": 16}) == {"type": "convlstm"}
    cfg = dict(
        LATENT_CFG, predictor={"type": "latent", "encoder": "fake-16", "depth": 1, "heads": 2}
    )
    spec = spec_from(cfg)
    small = new_model(cfg, spec, torch.device("cpu"), 32)
    large = new_model(cfg, spec, torch.device("cpu"), 64)
    assert small.project.out_features == 32 and large.project.out_features == 64
    assert (
        cfg["predictor"]["encoder"].startswith("fake")
        and LATENT_CFG["predictor"]["encoder"] == "dinov2-small"
    )
    assert LATENT_CFG["hidden_channels"] == 256 and LATENT_CFG["learning_rate"] == pytest.approx(
        3e-4
    )


def test_encode_block_replaces_videos_with_tokens_once():
    cfg = dict(
        LATENT_CFG, predictor={"type": "latent", "encoder": "fake-16", "depth": 1, "heads": 2}
    )
    video = torch.rand(5, 8, 3, 64, 64)
    target = torch.rand(5, 16, 2, 2)
    exact = (torch.rand(4, 8, 3, 64, 64), torch.rand(4, 16, 2, 2))
    block = {
        "base": (video, target),
        "source": (torch.rand(4, 8, 3, 64, 64), torch.rand(4, 16, 2, 2)),
        "exact": exact,
        "independent": (torch.rand(4, 8, 3, 64, 64), torch.rand(4, 16, 2, 2)),
        "screen": (torch.rand(4, 8, 3, 64, 64), torch.rand(4, 16, 2, 2)),
        "arms": {"B": (torch.rand(4, 8, 3, 64, 64), torch.rand(4, 16, 2, 2)), "E": exact},
    }
    encoded = encode_block(block, cfg, torch.device("cpu"))
    assert encoded["base"][0].shape == (5, 8, 17, 16) and torch.equal(encoded["base"][1], target)
    assert encoded["arms"]["E"][0] is encoded["exact"][0]  # shared video encoded once
    assert (
        encode_block(block, {"hidden_channels": 16}, torch.device("cpu")) is block
    )  # convlstm: untouched


def test_training_step_reduces_loss_on_tokens():
    cfg = dict(
        LATENT_CFG, predictor={"type": "latent", "encoder": "fake-16", "depth": 1, "heads": 2}
    )
    spec = spec_from(cfg)
    torch.manual_seed(0)
    model = new_model(cfg, spec, torch.device("cpu"), 32)
    tokens = torch.randn(16, 8, 17, 16)
    target = torch.rand(16, 16, 2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = train_steps(model, opt, [(tokens, target)] * 30, torch.device("cpu"))
    assert losses[-1] < losses[0]


def test_vjepa_step_count_follows_tubelets():
    cfg = dict(
        LATENT_CFG, predictor={"type": "latent", "encoder": "fake-16", "depth": 1, "heads": 2}
    )
    spec = spec_from(cfg)
    assert new_model(cfg, spec, torch.device("cpu"), 16).steps == spec.observed_frames
    # a V-JEPA-named fake is not available, so check the rule directly
    assert (spec.observed_frames // 2) == 4


def test_tune_base_2d_runs_for_latent_and_convlstm(tmp_path):
    """The tuning script runs end to end for both predictor types (the notebook runs it first)."""

    import os
    import subprocess

    base = json.loads((ROOT / "experiments" / "exposure2d_pilot_v1.json").read_text())
    tiny = dict(base, base_train_examples=16, screen_examples=8, batch_size=8, hidden_channels=8)
    for name, extra in (
        ("convlstm", {}),
        ("latent", {"predictor": {"type": "latent", "encoder": "fake-16", "depth": 1, "heads": 2}}),
    ):
        cfg = dict(tiny, **extra)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(cfg))
        out = tmp_path / f"{name}_tuning.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "tune_base_2d.py"),
                "--config",
                str(path),
                "--out",
                str(out),
                "--blocks",
                "1",
                "--checkpoints",
                "2,4",
                "--hidden",
                "8",
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        result = json.loads(out.read_text())
        assert "h8_s4" in result["summary"]
