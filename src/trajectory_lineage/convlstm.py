"""A compact ConvLSTM visual state predictor for the feasibility pilot."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with a spatial hidden state."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, value: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((value, hidden), dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell


class ConvLSTMStatePredictor(nn.Module):
    """Predict future object centres from an RGB video prefix.

    The convolutional encoder and ConvLSTM consume pixels. A fixed-size head predicts all future
    centre coordinates at once. This is a visual state predictor, not a pixel generator.
    """

    def __init__(
        self,
        *,
        future_frames: int = 16,
        hidden_channels: int = 16,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size % 4:
            raise ValueError("image_size must be divisible by four")
        self.future_frames = future_frames
        self.hidden_channels = hidden_channels
        self.spatial_size = image_size // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.recurrent = ConvLSTMCell(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * self.spatial_size * self.spatial_size, 256),
            nn.ReLU(),
            nn.Linear(256, future_frames * 4),
            nn.Sigmoid(),
        )

    def forward(self, video_prefix: Tensor) -> Tensor:
        """Return ``[batch, future_frames, 2, 2]`` normalized centre coordinates."""

        if video_prefix.ndim != 5 or video_prefix.shape[2] != 3:
            raise ValueError("video_prefix must have shape [batch, frames, 3, height, width]")
        batch, _, _, height, width = video_prefix.shape
        if height != width or height // 4 != self.spatial_size:
            raise ValueError("video spatial dimensions do not match the configured image size")

        hidden = torch.zeros(
            batch,
            self.hidden_channels,
            self.spatial_size,
            self.spatial_size,
            device=video_prefix.device,
            dtype=video_prefix.dtype,
        )
        cell = torch.zeros_like(hidden)
        for frame_index in range(video_prefix.shape[1]):
            encoded = self.encoder(video_prefix[:, frame_index])
            hidden, cell = self.recurrent(encoded, (hidden, cell))
        return self.head(hidden).view(batch, self.future_frames, 2, 2)
