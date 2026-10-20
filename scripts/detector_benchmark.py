"""Benchmark exact, perceptual, and motion duplicate detectors against trajectory lineage.

Requires torch, torchvision, and Pillow (see docs/REPRODUCE.md). Pretrained weights are fetched
from their public release locations on first use.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from trajectory_lineage.detectors import (
    centroid_motion_descriptor,
    silhouette_descriptor,
)
from trajectory_lineage.exposure import (
    EventSpec,
    nearest_distances,
    sample_closer_neighbor,
    sample_event,
)
from trajectory_lineage.simulator import render

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TRANSFORMS = (
    "copy_reencode",
    "appearance",
    "texture",
    "compression",
    "crop",
    "mirror",
    "time_shift",
    "distractor",
)


def radius_pixels(event, size: int) -> int:
    return max(2, round(event.state.radius * size))


def render_clip(
    event, appearance_seed: int, transform: str | None, rng: np.random.Generator, config
) -> np.ndarray:
    size = int(config["image_size"])
    frames = int(config["clip_frames"])
    pixels = render(
        event.trajectory,
        appearance_seed=appearance_seed,
        size=size,
        radius_pixels=radius_pixels(event, size),
    )
    if transform == "time_shift":
        shift = int(config["time_shift_frames"])
        clip = pixels[shift : shift + frames].copy()
    else:
        clip = pixels[:frames].copy()
    if transform in (None, "appearance", "time_shift"):
        return clip
    if transform == "copy_reencode":
        out = np.empty_like(clip)
        for index, frame in enumerate(clip):
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=75)
            out[index] = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        return out
    if transform == "texture":
        background = clip[:, :1, :1, :]
        mask = np.all(clip == background, axis=-1)
        texture = rng.integers(120, 256, size=(size, size, 3), dtype=np.uint8)
        clip[mask] = np.broadcast_to(texture, clip.shape)[mask]
        return clip
    if transform == "compression":
        out = np.empty_like(clip)
        for index, frame in enumerate(clip):
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=25)
            out[index] = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        return out
    if transform == "crop":
        keep = int(round(size * 0.8))
        start = (size - keep) // 2
        out = np.empty_like(clip)
        for index, frame in enumerate(clip):
            cropped = Image.fromarray(frame[start : start + keep, start : start + keep])
            out[index] = np.asarray(cropped.resize((size, size), Image.BILINEAR))
        return out
    if transform == "mirror":
        return clip[:, :, ::-1].copy()
    if transform == "distractor":
        scale = size / 64.0
        radius = 3.0 * scale
        yy, xx = np.ogrid[:size, :size]
        start = rng.uniform([8 * scale, 8 * scale], [size - 8 * scale, size - 8 * scale])
        velocity = rng.uniform(-1.2, 1.2, size=2) * scale
        color = rng.integers(30, 205, size=3, dtype=np.uint8)
        for index in range(frames):
            center = np.clip(start + velocity * index, radius, size - 1 - radius)
            mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
            clip[index][mask] = color
        return clip
    raise ValueError(f"unknown transform {transform}")


def build_split(config: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    spec = EventSpec(
        observed_frames=8,
        future_frames=int(config["simulated_frames"]) - 8,
        frame_dt=float(config["frame_dt"]),
        image_size=int(config["image_size"]),
    )
    gallery = [sample_event(rng, spec) for _ in range(int(config["gallery_trajectories"]))]
    gallery_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in gallery]
    positives = list(range(int(config["positive_queries"])))
    hard_targets = list(
        range(len(positives), len(positives) + int(config["hard_negative_queries"]))
    )
    band = config["hard_negative_band"]
    hard, hard_distances, references = [], [], []
    for index in hard_targets:
        others = gallery[:index] + gallery[index + 1 :]
        reference = float(nearest_distances([gallery[index]], others)[0])
        neighbor, distance, _ = sample_closer_neighbor(
            rng,
            gallery[index],
            spec,
            min_distance=float(band["low_fraction"]) * reference,
            max_distance=float(band["high_fraction"]) * reference,
            box_multiplier=float(band["box_multiplier"]),
        )
        hard.append(neighbor)
        hard_distances.append(distance)
        references.append(reference)
    easy = [sample_event(rng, spec) for _ in range(int(config["easy_negative_queries"]))]
    background = [sample_event(rng, spec) for _ in range(int(config["background_clips"]))]
    return {
        "hard_targets": hard_targets,
        "background": background,
        "rng": rng,
        "gallery": gallery,
        "gallery_seeds": gallery_seeds,
        "positive_sources": positives,
        "hard": hard,
        "hard_distances": hard_distances,
        "hard_references": references,
        "easy": easy,
    }


class LearnedEncoders:
    """Lazily loaded pretrained frame encoders and the RAFT flow model."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._sscd = None
        self._dino = None
        self._raft = None

    def sscd(self):
        if self._sscd is None:
            path = Path(torch.hub.get_dir()) / "sscd_disc_mixup.torchscript.pt"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(SSCD_URL, str(path))
            self._sscd = torch.jit.load(str(path), map_location="cpu").eval().to(self.device)
        return self._sscd

    def dino(self):
        if self._dino is None:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                verbose=False,
                trust_repo=True,
                skip_validation=True,
            )
            self._dino = model.eval().to(self.device)
        return self._dino

    def raft(self):
        if self._raft is None:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self._raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(self.device)
        return self._raft


def frame_embeddings(
    encoder: Callable, frames: np.ndarray, device: torch.device, batch: int = 256
) -> np.ndarray:
    """Embed ``[n, h, w, 3]`` uint8 frames at 224 px with ImageNet normalization."""

    outputs = []
    with torch.no_grad():
        for start in range(0, len(frames), batch):
            chunk = torch.from_numpy(frames[start : start + batch]).permute(0, 3, 1, 2).float()
            chunk = functional.interpolate(chunk / 255.0, size=224, mode="bilinear")
            chunk = ((chunk - IMAGENET_MEAN) / IMAGENET_STD).to(device)
            embedding = encoder(chunk).float()
            outputs.append(functional.normalize(embedding, dim=-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def flow_descriptors(
    raft, clips: np.ndarray, indices: list[int], device: torch.device, batch: int = 64
) -> np.ndarray:
    """RAFT-small flow between sampled frames, pooled to an 8x8 grid, in native pixels."""

    native = clips.shape[2]
    flow_size = max(128, int(math.ceil(native / 8) * 8))

    sampled = clips[:, indices]
    first = sampled[:, :-1].reshape(-1, *sampled.shape[2:])
    second = sampled[:, 1:].reshape(-1, *sampled.shape[2:])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(first), batch):
            left = torch.from_numpy(first[start : start + batch]).permute(0, 3, 1, 2).float()
            right = torch.from_numpy(second[start : start + batch]).permute(0, 3, 1, 2).float()
            left = functional.interpolate(left / 127.5 - 1.0, size=flow_size, mode="bilinear")
            right = functional.interpolate(right / 127.5 - 1.0, size=flow_size, mode="bilinear")
            flow = raft(left.to(device), right.to(device))[-1] * (native / flow_size)
            outputs.append(functional.adaptive_avg_pool2d(flow, 8).cpu().numpy())
    pooled = np.concatenate(outputs).astype(np.float32)
    return pooled.reshape(len(clips), len(indices) - 1, 2, 8, 8)


def ofs_flows(raft, clips: np.ndarray, device: torch.device, batch: int = 64) -> np.ndarray:
    """Unit flow vectors between all consecutive frames on a pooled grid, for OFS-k.

    Follows Chen et al. (2024): RAFT flow for every consecutive frame pair and per-pixel cosine
    similarity between time-aligned flows. Flow is computed at 128 px and average-pooled to a
    16 x 16 grid before normalization (a compute adaptation); zero-flow cells contribute zero.
    """

    frames = clips.shape[1]
    first = clips[:, :-1].reshape(-1, *clips.shape[2:])
    second = clips[:, 1:].reshape(-1, *clips.shape[2:])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(first), batch):
            left = torch.from_numpy(first[start : start + batch]).permute(0, 3, 1, 2).float()
            right = torch.from_numpy(second[start : start + batch]).permute(0, 3, 1, 2).float()
            left = functional.interpolate(left / 127.5 - 1.0, size=128, mode="bilinear")
            right = functional.interpolate(right / 127.5 - 1.0, size=128, mode="bilinear")
            flow = raft(left.to(device), right.to(device))[-1]
            pooled = functional.adaptive_avg_pool2d(flow, 16)
            norm = pooled.norm(dim=1, keepdim=True)
            unit = torch.where(norm > 1e-3, pooled / norm.clamp(min=1e-6), torch.zeros_like(pooled))
            outputs.append(unit.cpu().numpy())
    flows = np.concatenate(outputs).astype(np.float32)
    return flows.reshape(len(clips), frames - 1, 2, 16, 16)


def ofs_matrix(query: np.ndarray, gallery: np.ndarray, k: int) -> torch.Tensor:
    """OFS-k: max over k-frame windows of mean time-aligned per-cell flow cosine."""

    steps = query.shape[1]
    cells = query.shape[3] * query.shape[4]
    q = torch.from_numpy(query).flatten(2)
    g = torch.from_numpy(gallery).flatten(2)
    per_step = torch.einsum("qtd,gtd->tqg", q, g) / cells
    windows = per_step.unfold(0, k, 1).mean(dim=-1) if steps >= k else per_step.mean(0, True)
    return windows.max(dim=0).values


def describe_ofs(clips: np.ndarray, config, encoders: LearnedEncoders) -> dict[str, Any]:
    hashes = [hashlib.sha256(np.ascontiguousarray(clip).tobytes()).hexdigest() for clip in clips]
    return {"hash": hashes, "ofs": ofs_flows(encoders.raft(), clips, encoders.device)}


def describe(clips: np.ndarray, config, encoders: LearnedEncoders) -> dict[str, Any]:
    """Compute every detector descriptor for a batch of clips."""

    indices = list(config["learned_frame_indices"])
    hashes = [hashlib.sha256(np.ascontiguousarray(clip).tobytes()).hexdigest() for clip in clips]
    silhouettes = np.stack([silhouette_descriptor(clip).ravel() for clip in clips]).astype(
        np.float32
    )
    centroids = np.full((len(clips), clips.shape[1] * 4), np.nan, dtype=np.float32)
    for row, clip in enumerate(clips):
        try:
            centroids[row] = centroid_motion_descriptor(clip).ravel()
        except ValueError:
            pass
    frames = clips[:, indices].reshape(-1, *clips.shape[2:])
    sscd = frame_embeddings(encoders.sscd(), frames, encoders.device)
    dino = frame_embeddings(encoders.dino(), frames, encoders.device)
    flows = flow_descriptors(encoders.raft(), clips, indices, encoders.device)
    count = len(indices)
    return {
        "hash": hashes,
        "silhouette": silhouettes,
        "centroid": centroids,
        "sscd": sscd.reshape(len(clips), count, -1),
        "dino": dino.reshape(len(clips), count, -1),
        "flow": flows.reshape(len(clips), -1),
        "flow_steps": flows,
    }


def concat(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in parts[0]:
        if key == "hash":
            out[key] = [value for part in parts for value in part[key]]
        else:
            out[key] = np.concatenate([part[key] for part in parts])
    return out


DESCRIBE = {"all": None, "ofs": None}


def describe_in_chunks(clip_iter, config, encoders, chunk: int | None = None) -> dict[str, Any]:
    if chunk is None:
        chunk = max(25, int(200 * (64 / int(config["image_size"])) ** 2))
    parts, buffer = [], []
    for clip in clip_iter:
        buffer.append(clip)
        if len(buffer) == chunk:
            parts.append(_describer(config)(np.stack(buffer), config, encoders))
            buffer = []
    if buffer:
        parts.append(_describer(config)(np.stack(buffer), config, encoders))
    return concat(parts)


def _describer(config):
    return describe_ofs if config.get("_detector_mode") == "ofs" else describe


def negative_rms_matrix(query: np.ndarray, gallery: np.ndarray) -> torch.Tensor:
    """Negative RMS distance matrix; NaN query rows score -inf, NaN gallery rows never match."""

    q = torch.from_numpy(np.nan_to_num(query, nan=0.0)).double()
    g = torch.from_numpy(np.nan_to_num(gallery, nan=1e6)).double()
    squared = (q.square().sum(1, keepdim=True) - 2 * q @ g.T + g.square().sum(1)[None]).clamp(min=0)
    similarity = -(squared / query.shape[1]).sqrt()
    similarity[torch.from_numpy(np.isnan(query).any(axis=1))] = -math.inf
    return similarity


def track_variants(tracks: np.ndarray, mirror: bool, width: int) -> np.ndarray:
    """Centroid tracks [n, frames, 2, 2] in units of pixel/width.

    A horizontal mirror maps pixel column c to width - 1 - c, i.e. x to (width - 1)/width - x;
    objects are then re-sorted by x.
    """

    if not mirror:
        return tracks
    flipped = tracks.copy()
    flipped[..., 0] = (width - 1) / width - flipped[..., 0]
    return flipped[:, :, ::-1, :]


def invariant_track_matrix(
    query: np.ndarray, gallery: np.ndarray, max_shift: int, width: int
) -> torch.Tensor:
    """Best negative RMS over mirror and integer time offsets on overlapping frames."""

    frames = query.shape[1] // 4
    q_tracks = query.reshape(len(query), frames, 2, 2)
    g_tracks = gallery.reshape(len(gallery), frames, 2, 2)
    best = None
    for mirror in (False, True):
        variant = track_variants(q_tracks, mirror, width)
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                left, right = variant[:, shift:], g_tracks[:, : frames - shift]
            else:
                left, right = variant[:, :shift], g_tracks[:, -shift:]
            matrix = negative_rms_matrix(
                left.reshape(len(left), -1), np.ascontiguousarray(right).reshape(len(right), -1)
            )
            best = matrix if best is None else torch.maximum(best, matrix)
    return best


def flow_variants(flows: np.ndarray, mirror: bool) -> np.ndarray:
    """Flow descriptors [n, steps, 2, 8, 8]; mirroring flips the grid and negates x-flow."""

    if not mirror:
        return flows
    flipped = flows[..., ::-1].copy()
    flipped[:, :, 0] *= -1
    return flipped


def invariant_flow_matrix(query: np.ndarray, gallery: np.ndarray, max_shift: int) -> torch.Tensor:
    best = None
    for mirror in (False, True):
        variant = flow_variants(query, mirror)
        steps = variant.shape[1]
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                left, right = variant[:, shift:], gallery[:, : steps - shift]
            else:
                left, right = variant[:, :shift], gallery[:, -shift:]
            q = functional.normalize(
                torch.from_numpy(np.ascontiguousarray(left)).flatten(1), dim=-1
            )
            g = functional.normalize(
                torch.from_numpy(np.ascontiguousarray(right)).flatten(1), dim=-1
            )
            matrix = q @ g.T
            best = matrix if best is None else torch.maximum(best, matrix)
    return best


def cosine_pooled_matrix(query: np.ndarray, gallery: np.ndarray) -> torch.Tensor:
    q = functional.normalize(torch.from_numpy(query).float().mean(1), dim=-1)
    g = functional.normalize(torch.from_numpy(gallery).float().mean(1), dim=-1)
    return q @ g.T


def cosine_aligned_matrix(query: np.ndarray, gallery: np.ndarray) -> torch.Tensor:
    q = torch.from_numpy(query).float()
    g = torch.from_numpy(gallery).float()
    return torch.einsum("qtd,gtd->qg", q, g) / q.shape[1]


def cosine_flat_matrix(query: np.ndarray, gallery: np.ndarray) -> torch.Tensor:
    q = functional.normalize(torch.from_numpy(query).float(), dim=-1)
    g = functional.normalize(torch.from_numpy(gallery).float(), dim=-1)
    return q @ g.T


def exact_matrix(query: list[str], gallery: list[str]) -> torch.Tensor:
    lookup = np.array(gallery, dtype=object)
    return torch.from_numpy(
        np.array([[1.0 if value == other else 0.0 for other in lookup] for value in query])
    )


def oracle_matrix(query_events, gallery_events, frames: int) -> torch.Tensor:
    q = np.stack([event.trajectory.positions[:frames].ravel() for event in query_events])
    g = np.stack([event.trajectory.positions[:frames].ravel() for event in gallery_events])
    return negative_rms_matrix(q.astype(np.float32), g.astype(np.float32))


def background_normalized(
    matrix: torch.Tensor, background_matrix: torch.Tensor, top_k: int
) -> torch.Tensor:
    """Subtract each query's mean top-k similarity to an unrelated background set."""

    k = min(top_k, background_matrix.shape[1])
    return matrix - background_matrix.topk(k, dim=1).values.mean(dim=1, keepdim=True)


def score_all(
    query: dict,
    gallery: dict,
    max_shift: int,
    flow_shift: int,
    background: dict | None = None,
    top_k: int = 10,
    width: int = 64,
) -> dict[str, Any]:
    matrices = _raw_scores(query, gallery, max_shift, flow_shift, width)
    if background is not None:
        for name, key, function in (
            ("sscd_aligned", "sscd", cosine_aligned_matrix),
            ("dinov2_aligned", "dino", cosine_aligned_matrix),
            ("raft_flow", "flow", cosine_flat_matrix),
        ):
            reference = function(query[key], background[key])
            matrices[f"{name}_norm"] = background_normalized(matrices[name], reference, top_k)
    return matrices


def _raw_scores(
    query: dict, gallery: dict, max_shift: int, flow_shift: int, width: int
) -> dict[str, Any]:
    return {
        "exact_hash": exact_matrix(query["hash"], gallery["hash"]),
        "silhouette": negative_rms_matrix(query["silhouette"], gallery["silhouette"]),
        "centroid_motion": negative_rms_matrix(query["centroid"], gallery["centroid"]),
        "centroid_invariant": invariant_track_matrix(
            query["centroid"], gallery["centroid"], max_shift, width
        ),
        "sscd_pooled": cosine_pooled_matrix(query["sscd"], gallery["sscd"]),
        "sscd_aligned": cosine_aligned_matrix(query["sscd"], gallery["sscd"]),
        "dinov2_pooled": cosine_pooled_matrix(query["dino"], gallery["dino"]),
        "dinov2_aligned": cosine_aligned_matrix(query["dino"], gallery["dino"]),
        "raft_flow": cosine_flat_matrix(query["flow"], gallery["flow"]),
        "raft_flow_invariant": invariant_flow_matrix(
            query["flow_steps"], gallery["flow_steps"], flow_shift
        ),
    }


def expected_tracks(event, transform: str, config: dict[str, Any]) -> np.ndarray:
    """Ground-truth disk centers in the centroid descriptor's units (pixel / width), by x."""

    size = int(config["image_size"])
    frames = int(config["clip_frames"])
    start = int(config["time_shift_frames"]) if transform == "time_shift" else 0
    pixels = event.trajectory.positions[start : start + frames] * (size - 1)
    if transform == "mirror":
        pixels = pixels.copy()
        pixels[..., 0] = size - 1 - pixels[..., 0]
    if transform == "crop":
        keep = int(round(size * 0.8))
        offset = (size - keep) // 2
        pixels = (pixels - offset + 0.5) * size / keep - 0.5
    order = np.argsort(pixels[..., 0], axis=1)
    sorted_pixels = np.take_along_axis(pixels, order[..., None], axis=1)
    return sorted_pixels / size


def track_rmse_pixels(descriptors: np.ndarray, events, transform: str, config) -> dict[str, Any]:
    size = int(config["image_size"])
    errors, failures = [], 0
    for row, event in zip(descriptors, events, strict=True):
        if np.isnan(row).any():
            failures += 1
            continue
        tracks = row.reshape(-1, 2, 2)
        truth = expected_tracks(event, transform, config)
        errors.append(float(np.sqrt(np.square((tracks - truth) * size).mean())))
    return {
        "median_rmse_px": float(np.median(errors)) if errors else None,
        "fraction_within_2px": float(np.mean(np.array(errors) <= 2.0)) if errors else None,
        "extraction_failures": failures,
    }


def run_split(config: dict[str, Any], seed: int, encoders: LearnedEncoders) -> dict[str, Any]:
    split = build_split(config, seed)
    rng = split["rng"]
    gallery_events = split["gallery"]
    gallery_clips = (
        render_clip(event, seed_value, None, rng, config)
        for event, seed_value in zip(gallery_events, split["gallery_seeds"], strict=True)
    )
    gallery = describe_in_chunks(gallery_clips, config, encoders)
    background_clips = (
        render_clip(event, int(rng.integers(0, 2**31 - 1)), None, rng, config)
        for event in split["background"]
    )
    if config.get("_detector_mode") == "ofs":
        for _ in background_clips:  # render to keep the random stream identical to full mode
            pass
        background = None
    else:
        background = describe_in_chunks(background_clips, config, encoders)
    frames = int(config["clip_frames"])
    max_shift = int(config["invariant_max_shift_frames"])
    flow_shift = int(config["invariant_max_shift_flow_steps"])
    positive_events = [gallery_events[index] for index in split["positive_sources"]]
    roles = {"positive": positive_events, "hard": split["hard"], "easy": split["easy"]}
    targets = np.array(split["positive_sources"])
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for transform in TRANSFORMS:
        scores[transform] = {}
        for role, events in roles.items():
            gallery_seeds = split["gallery_seeds"]
            if transform == "copy_reencode" and role == "positive":
                seeds = [gallery_seeds[index] for index in split["positive_sources"]]
            elif transform == "copy_reencode" and role == "hard":
                # Same appearance as the target gallery clip: only the trajectory differs.
                seeds = [gallery_seeds[index] for index in split["hard_targets"]]
            elif transform == "copy_reencode":
                # Easy negatives borrow a gallery palette so appearance alone cannot match.
                seeds = [
                    int(gallery_seeds[int(rng.integers(0, len(gallery_seeds)))]) for _ in events
                ]
            else:
                seeds = [int(rng.integers(0, 2**31 - 1)) for _ in events]
            clips = (
                render_clip(event, seed_value, transform, rng, config)
                for event, seed_value in zip(events, seeds, strict=True)
            )
            query = describe_in_chunks(clips, config, encoders)
            if config.get("_detector_mode") == "ofs":
                matrices = {
                    f"ofs_{k}": ofs_matrix(query["ofs"], gallery["ofs"], k) for k in (3, 5, 10)
                }
            else:
                matrices = score_all(
                    query,
                    gallery,
                    max_shift,
                    flow_shift,
                    background,
                    int(config["background_top_k"]),
                    int(config["image_size"]),
                )
            matrices["oracle_state"] = oracle_matrix(events, gallery_events, frames)
            entry: dict[str, Any] = {}
            for name, matrix in matrices.items():
                best, argbest = matrix.max(dim=1)
                record = {"score": best.double().numpy(), "argbest": argbest.numpy()}
                if role == "positive":
                    target_scores = matrix[torch.arange(len(targets)), torch.from_numpy(targets)]
                    greater = (matrix > target_scores[:, None]).sum(dim=1).double()
                    ties = (matrix == target_scores[:, None]).sum(dim=1).double() - 1
                    record["rank"] = (1 + greater + 0.5 * ties).numpy()
                entry[name] = record
            if "centroid" in query:
                entry["_centroid_failures"] = int(np.isnan(query["centroid"]).any(axis=1).sum())
                entry["_centroid_track_validation"] = track_rmse_pixels(
                    query["centroid"], events, transform, config
                )
            scores[transform][role] = entry
            print(f"  {transform:13s} {role:9s} n={len(events)}", flush=True)
    return {
        "scores": scores,
        "positive_sources": split["positive_sources"],
        "hard_distance_median": float(np.median(split["hard_distances"])),
        "hard_reference_median": float(np.median(split["hard_references"])),
    }


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return [center - half, center + half]


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate([positive, negative])
    finite_floor = np.nanmin(values[np.isfinite(values)]) - 1.0 if np.isfinite(values).any() else 0
    values = np.where(np.isfinite(values), values, finite_floor)
    ranks = np.empty(len(values), dtype=np.float64)
    # Average ranks for ties (Mann-Whitney formulation).
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0
        start = end + 1
    positive_ranks = ranks[: len(positive)] + 1
    n_pos, n_neg = len(positive), len(negative)
    return float((positive_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(config, calibration, test) -> dict[str, Any]:
    detectors = list(test["scores"]["appearance"]["positive"].keys())
    detectors = [name for name in detectors if not name.startswith("_")]
    summary: dict[str, Any] = {}
    for name in detectors:
        pooled = np.concatenate(
            [calibration["scores"][t]["easy"][name]["score"] for t in TRANSFORMS]
        )
        pooled = np.where(np.isfinite(pooled), pooled, -1e30)
        detector: dict[str, Any] = {"thresholds": {}, "transforms": {}}
        for alpha in config["operating_points"]:
            detector["thresholds"][str(alpha)] = float(
                np.quantile(pooled, 1 - alpha, method="higher")
            )
        for transform in TRANSFORMS:
            roles = test["scores"][transform]
            row: dict[str, Any] = {
                "median_true_source_rank": float(np.median(roles["positive"][name]["rank"])),
                "true_source_top1": float(np.mean(roles["positive"][name]["rank"] == 1)),
                "auroc_vs_easy": auroc(
                    roles["positive"][name]["score"], roles["easy"][name]["score"]
                ),
                "auroc_vs_hard": auroc(
                    roles["positive"][name]["score"], roles["hard"][name]["score"]
                ),
            }
            for alpha, threshold in detector["thresholds"].items():
                removed = {
                    role: roles[role][name]["score"] > threshold
                    for role in ("positive", "hard", "easy")
                }
                correct = roles["positive"][name]["argbest"] == np.array(test["positive_sources"])
                n_pos = len(removed["positive"])
                row[alpha] = {
                    "recall": float(removed["positive"].mean()),
                    "recall_ci": wilson(int(removed["positive"].sum()), n_pos),
                    "correct_source_among_removed": float(correct[removed["positive"]].mean())
                    if removed["positive"].any()
                    else None,
                    "hard_false_removal": float(removed["hard"].mean()),
                    "hard_false_removal_ci": wilson(
                        int(removed["hard"].sum()), len(removed["hard"])
                    ),
                    "easy_false_removal": float(removed["easy"].mean()),
                    "easy_false_removal_ci": wilson(
                        int(removed["easy"].sum()), len(removed["easy"])
                    ),
                }
            detector["transforms"][transform] = row
        summary[name] = detector
    if "_centroid_failures" not in test["scores"][TRANSFORMS[0]]["positive"]:
        return {"detectors": summary}
    failures = {
        transform: {
            role: test["scores"][transform][role]["_centroid_failures"]
            for role in ("positive", "hard", "easy")
        }
        for transform in TRANSFORMS
    }
    validation = {
        transform: {
            role: test["scores"][transform][role]["_centroid_track_validation"]
            for role in ("positive", "hard", "easy")
        }
        for transform in TRANSFORMS
    }
    return {
        "detectors": summary,
        "centroid_extraction_failures_test": failures,
        "centroid_track_validation_test": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/detector_benchmark_v4.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/detector_benchmark_v4.json"))
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Shrink query counts for smoke runs."
    )
    parser.add_argument(
        "--detectors",
        choices=("all", "ofs"),
        default="all",
        help="'ofs' scores only OFS-k on the identical clips (same seeds and rendering).",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["_detector_mode"] = args.detectors
    if args.scale != 1.0:
        for key in (
            "gallery_trajectories",
            "positive_queries",
            "hard_negative_queries",
            "easy_negative_queries",
            "background_clips",
        ):
            config[key] = max(4, int(config[key] * args.scale))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import subprocess

    def git(*command: str) -> str:
        try:
            return subprocess.check_output(["git", *command], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    provenance = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--untracked-files=no") != "",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "sscd_url": SSCD_URL,
        "dinov2": "facebookresearch/dinov2 dinov2_vits14 (torch.hub)",
        "raft": "torchvision Raft_Small_Weights.DEFAULT",
    }
    encoders = LearnedEncoders(device)
    started = time.perf_counter()
    print("calibration split", flush=True)
    calibration = run_split(config, int(config["splits"]["calibration"]), encoders)
    print("test split", flush=True)
    test = run_split(config, int(config["splits"]["test"]), encoders)
    result = {
        "config": config,
        "provenance": provenance,
        "device": str(device),
        "torch": torch.__version__,
        "seconds": time.perf_counter() - started,
        "hard_reference_median": {
            "calibration": calibration["hard_reference_median"],
            "test": test["hard_reference_median"],
        },
        "hard_distance_median": {
            "calibration": calibration["hard_distance_median"],
            "test": test["hard_distance_median"],
        },
        **summarize(config, calibration, test),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    scores_path = args.out.with_name(args.out.stem + "_scores.json.gz")
    export = {
        "config": config,
        "splits": {
            name: {
                "positive_sources": split["positive_sources"],
                "scores": {
                    transform: {
                        role: {
                            detector: {
                                key: np.asarray(value).tolist() for key, value in rec.items()
                            }
                            for detector, rec in entry.items()
                            if not detector.startswith("_")
                        }
                        for role, entry in roles.items()
                    }
                    for transform, roles in split["scores"].items()
                },
            }
            for name, split in (("calibration", calibration), ("test", test))
        },
    }
    with gzip.open(scores_path, "wt") as handle:
        json.dump(export, handle)
    print(f"done in {result['seconds']:.0f}s -> {args.out}, {scores_path}")


if __name__ == "__main__":
    main()
