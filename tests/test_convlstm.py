import pytest

torch = pytest.importorskip("torch")
ConvLSTMStatePredictor = pytest.importorskip("trajectory_lineage.convlstm").ConvLSTMStatePredictor


def test_convlstm_state_predictor_shape_and_range() -> None:
    model = ConvLSTMStatePredictor(future_frames=5, hidden_channels=4, image_size=16)
    video = torch.rand(2, 3, 3, 16, 16)

    prediction = model(video)

    assert prediction.shape == (2, 5, 2, 2)
    assert torch.isfinite(prediction).all()
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))


def test_convlstm_rejects_wrong_image_size() -> None:
    model = ConvLSTMStatePredictor(future_frames=5, hidden_channels=4, image_size=16)

    with pytest.raises(ValueError, match="spatial dimensions"):
        model(torch.rand(1, 3, 3, 20, 20))
