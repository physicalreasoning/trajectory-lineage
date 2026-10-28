"""Latent world-model predictor: a frozen pretrained encoder plus a small trained dynamics model.

This is the recipe of current world models (DINO-WM, V-JEPA 2-AC, the Kuka open-loop study and the
paper's Experiment 1): a large pretrained vision encoder turns each frame into tokens and is never
trained on the study's data; a small transformer trained on the block's trajectories reads the
observed tokens and predicts the future state. In the reuse study the encoder therefore never sees a
block's training trajectories, so any reuse advantage must come from the dynamics model memorizing
trajectories in latent space.

Encoders (``FrozenEncoder``):
  dinov2-small   facebook/dinov2-small, per frame; class token + the 16x16 patch grid pooled to 4x4
                 -> 17 tokens x 384 per frame (the Kuka study's representation).
  vjepa2-vitl    facebook/vjepa2-vitl-fpc64-256, on the 8-frame observed clip; 4 tubelets x 256 tokens
                 pooled to 4x4 + mean -> 17 tokens x 1024 per tubelet.
  fake-<dim>     deterministic random projection of downsampled pixels, for tests and dry runs.

Renders stay at the registered 64x64; frames are upsampled to the encoder's input size at encoding
time, so the blocks (events, appearances, arms) are identical to the ConvLSTM levels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenEncoder:
    """Pretrained encoder applied with no gradient; returns tokens ``[batch, steps, tokens, dim]``."""

    def __init__(self, name: str, device, batch_size: int = 64) -> None:
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.hub_id = None
        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        self._mean, self._std = mean.to(device), std.to(device)
        if name == "dinov2-small":
            from transformers import AutoModel

            self.hub_id = "facebook/dinov2-small"
            self.model = AutoModel.from_pretrained(self.hub_id).to(device).eval()
            self.input_size, self.token_dim, self.tokens_per_step = 224, 384, 17
        elif name == "vjepa2-vitl":
            from transformers import VJEPA2Model

            self.hub_id = "facebook/vjepa2-vitl-fpc64-256"
            self.model = VJEPA2Model.from_pretrained(self.hub_id).to(device).eval()
            self.input_size, self.token_dim, self.tokens_per_step = 256, 1024, 17
        elif name.startswith("fake-"):
            self.model = None
            self.token_dim = int(name.split("-", 1)[1])
            self.input_size, self.tokens_per_step = 16, 17
            generator = torch.Generator().manual_seed(0)
            self._projection = torch.randn(3 * 16 * 16, self.token_dim, generator=generator).to(device) / 48.0
        else:
            raise ValueError(f"unknown encoder {name!r}")
        for parameter in getattr(self.model, "parameters", lambda: [])():
            parameter.requires_grad_(False)

    @property
    def parameters_frozen(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters())) if self.model is not None else 0

    def _prepare(self, video: Tensor) -> Tensor:
        """``[B, T, 3, h, w]`` in [0, 1] -> upsampled, normalized ``[B, T, 3, S, S]`` on the device."""

        batch, frames = video.shape[:2]
        x = video.to(self.device, non_blocking=True).float().flatten(0, 1)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = x.view(batch, frames, 3, self.input_size, self.input_size)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def encode(self, video: Tensor) -> Tensor:
        outputs = []
        for start in range(0, video.shape[0], self.batch_size):
            chunk = self._prepare(video[start : start + self.batch_size])
            outputs.append(self._encode_chunk(chunk).float().cpu())
        return torch.cat(outputs)

    def _encode_chunk(self, x: Tensor) -> Tensor:
        batch, frames = x.shape[:2]
        if self.name == "dinov2-small":
            hidden = self.model(pixel_values=x.flatten(0, 1)).last_hidden_state  # [B*T, 257, 384]
            cls, patches = hidden[:, :1], hidden[:, 1:]
            side = int(round(patches.shape[1] ** 0.5))
            grid = patches.view(-1, side, side, patches.shape[-1]).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)  # [B*T, 16, 384]
            tokens = torch.cat([cls, pooled], 1)
            return tokens.view(batch, frames, self.tokens_per_step, self.token_dim)
        if self.name == "vjepa2-vitl":
            hidden = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state  # [B, tubelets*256, 1024]
            tubelets = frames // 2
            side = int(round((hidden.shape[1] // tubelets) ** 0.5))
            grid = hidden.view(batch * tubelets, side, side, hidden.shape[-1]).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)  # [B*tub, 16, 1024]
            mean = grid.flatten(2).mean(-1, keepdim=True).transpose(1, 2)  # [B*tub, 1, 1024]
            tokens = torch.cat([mean, pooled], 1)
            return tokens.view(batch, tubelets, self.tokens_per_step, self.token_dim)
        # fake: deterministic projection of the downsampled frame, repeated to 17 tokens with a token index offset
        flat = x.flatten(0, 1).flatten(1)  # [B*T, 3*16*16]
        base = flat @ self._projection  # [B*T, dim]
        offsets = torch.linspace(-1.0, 1.0, self.tokens_per_step, device=x.device).view(1, -1, 1)
        tokens = base[:, None, :] * (1.0 + 0.1 * offsets)
        return tokens.view(batch, frames, self.tokens_per_step, self.token_dim)


class LatentStatePredictor(nn.Module):
    """Small transformer over frozen-encoder tokens that predicts future object centres.

    Observed tokens ``[batch, steps, tokens, dim]`` are projected to ``width``, given learned
    position embeddings, and attended together with ``future_frames`` learned query tokens; the
    queries are read out as ``[batch, future_frames, 2, 2]`` normalized coordinates. Width 256 and
    depth 4 are the paper's world-model size (Kuka study, Experiment 1).
    """

    def __init__(
        self,
        *,
        token_dim: int,
        steps: int,
        tokens_per_step: int,
        future_frames: int = 16,
        width: int = 256,
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.steps, self.tokens_per_step = steps, tokens_per_step
        self.project = nn.Linear(token_dim, width)
        self.position = nn.Parameter(torch.zeros(1, steps * tokens_per_step, width))
        self.queries = nn.Parameter(torch.zeros(1, future_frames, width))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.queries, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4), nn.Sigmoid())

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, steps, tokens, dim]")
        batch = tokens.shape[0]
        x = self.project(tokens.flatten(1, 2)) + self.position[:, : tokens.shape[1] * tokens.shape[2]]
        x = torch.cat([self.queries.expand(batch, -1, -1), x], 1)
        out = self.transformer(x)[:, : self.future_frames]
        return self.head(out).view(batch, self.future_frames, 2, 2)


_ENCODERS: dict[tuple[str, str], FrozenEncoder] = {}


def get_encoder(name: str, device) -> FrozenEncoder:
    """One encoder instance per (name, device) per process; weights load once."""

    key = (name, str(device))
    if key not in _ENCODERS:
        _ENCODERS[key] = FrozenEncoder(name, device)
    return _ENCODERS[key]
