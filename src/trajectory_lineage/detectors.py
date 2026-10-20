"""Classical overlap-detector baselines for the trajectory-lineage pilot."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


def exact_video_hash(video: UInt8Array) -> str:
    """Hash shape, dtype, and bytes for an exact-render baseline."""

    contiguous = np.ascontiguousarray(video)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def foreground_masks(video: UInt8Array, *, tolerance: float = 2.0) -> NDArray[np.bool_]:
    """Return foreground masks for synthetic renders with a uniform background."""

    frames = np.asarray(video, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video must have shape [frames, height, width, 3]")
    corners = np.stack(
        (
            frames[:, 0, 0],
            frames[:, 0, -1],
            frames[:, -1, 0],
            frames[:, -1, -1],
        ),
        axis=1,
    )
    background = np.median(corners, axis=1)
    distance = np.linalg.norm(frames - background[:, None, None, :], axis=-1)
    return distance > tolerance


def silhouette_descriptor(video: UInt8Array, *, cells: int = 8) -> FloatArray:
    """Downsample binary silhouettes into a color-invariant perceptual descriptor."""

    masks = foreground_masks(video)
    frames, height, width = masks.shape
    if height % cells or width % cells:
        raise ValueError("video dimensions must be divisible by cells")
    pooled = masks.reshape(
        frames,
        cells,
        height // cells,
        cells,
        width // cells,
    ).mean(axis=(2, 4))
    return pooled.astype(np.float64)


def descriptor_distance(left: FloatArray, right: FloatArray) -> float:
    """Root-mean-square distance between equal-shape descriptors."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("descriptor shapes differ")
    return float(np.sqrt(np.square(left_array - right_array).mean()))


def centroid_motion_descriptor(video: UInt8Array) -> FloatArray:
    """Extract unordered two-object center tracks from solid-color synthetic renders.

    Centers are sorted by horizontal position independently in every frame. This intentionally
    describes visible motion without relying on object color identity.
    """

    frames = np.asarray(video, dtype=np.uint8)
    masks = foreground_masks(frames)
    tracks: list[np.ndarray] = []
    for frame, mask in zip(frames, masks, strict=True):
        colors, counts = np.unique(frame[mask], axis=0, return_counts=True)
        if len(colors) < 2:
            raise ValueError("expected two foreground colors in every frame")
        selected = colors[np.argsort(counts)[-2:]]
        centers = []
        for color in selected:
            pixels = np.argwhere(np.all(frame == color, axis=-1))
            centers.append(
                pixels[:, [1, 0]].mean(axis=0) / np.array([frame.shape[1], frame.shape[0]])
            )
        tracks.append(np.array(sorted(centers, key=lambda point: point[0])))
    return np.stack(tracks)


def state_trajectory_distance(left: FloatArray, right: FloatArray) -> float:
    """Oracle root-mean-square state distance, minimized over a global object swap."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.shape[-2:] != (2, 2):
        raise ValueError("state arrays must share shape [frames, 2, 2]")
    direct = np.sqrt(np.square(left_array - right_array).mean())
    swapped = np.sqrt(np.square(left_array - right_array[:, ::-1]).mean())
    return float(min(direct, swapped))
