"""Validation and metadata helpers for the public PhyWorld HDF5 datasets."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

_REQUIRED_GROUPS = ("init_streams", "position_streams", "video_streams")


@dataclass(frozen=True)
class PhyWorldSummary:
    """Validated structural summary of one PhyWorld HDF5 file."""

    path: str
    sha256: str
    file_bytes: int
    groups: list[str]
    examples_per_group: dict[str, int]
    examples_total: int
    frames_per_example: int
    position_shape: list[int]
    init_width: int
    init_ranges: dict[str, list[float]]
    video_bytes_min: int
    video_bytes_max: int
    video_bytes_mean: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_phyworld(path: str | Path) -> PhyWorldSummary:
    """Validate group alignment and summarize a PhyWorld collision dataset."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    with h5py.File(dataset_path, "r") as handle:
        missing = [name for name in _REQUIRED_GROUPS if name not in handle]
        if missing:
            raise ValueError(f"missing required HDF5 groups: {missing}")

        keys_by_group = {name: sorted(handle[name].keys()) for name in _REQUIRED_GROUPS}
        expected_keys = keys_by_group[_REQUIRED_GROUPS[0]]
        for name, keys in keys_by_group.items():
            if keys != expected_keys:
                raise ValueError(f"group keys differ for {name}")
        if not expected_keys:
            raise ValueError("dataset contains no example groups")

        examples_per_group: dict[str, int] = {}
        frames_per_example: int | None = None
        position_tail: tuple[int, ...] | None = None
        init_width: int | None = None
        init_min: np.ndarray | None = None
        init_max: np.ndarray | None = None
        video_sizes: list[int] = []

        for key in expected_keys:
            init = handle["init_streams"][key]
            positions = handle["position_streams"][key]
            videos = handle["video_streams"][key]
            count = len(init)
            if len(positions) != count or len(videos) != count:
                raise ValueError(f"example counts differ in group {key}")
            if init.ndim != 2:
                raise ValueError(f"init_streams/{key} must be rank 2")
            if positions.ndim != 4 or tuple(positions.shape[-2:]) != (2, 2):
                raise ValueError(f"position_streams/{key} must have shape [examples, frames, 2, 2]")

            current_frames = int(positions.shape[1])
            current_tail = tuple(int(value) for value in positions.shape[1:])
            current_init_width = int(init.shape[1])
            if frames_per_example is None:
                frames_per_example = current_frames
                position_tail = current_tail
                init_width = current_init_width
            elif (
                current_frames != frames_per_example
                or current_tail != position_tail
                or current_init_width != init_width
            ):
                raise ValueError(f"tensor shapes differ in group {key}")

            init_values = init[:]
            if not np.isfinite(init_values).all() or not np.isfinite(positions[:]).all():
                raise ValueError(f"non-finite values found in group {key}")
            group_min = init_values.min(axis=0)
            group_max = init_values.max(axis=0)
            init_min = group_min if init_min is None else np.minimum(init_min, group_min)
            init_max = group_max if init_max is None else np.maximum(init_max, group_max)
            video_sizes.extend(len(payload) for payload in videos[:])
            examples_per_group[key] = count

    assert frames_per_example is not None
    assert position_tail is not None
    assert init_width is not None
    assert init_min is not None
    assert init_max is not None

    labels = ["radius_1", "radius_2", "velocity_1", "velocity_2"]
    init_ranges = {
        labels[index] if index < len(labels) else f"init_{index}": [
            float(init_min[index]),
            float(init_max[index]),
        ]
        for index in range(init_width)
    }

    return PhyWorldSummary(
        path=str(dataset_path),
        sha256=file_sha256(dataset_path),
        file_bytes=dataset_path.stat().st_size,
        groups=expected_keys,
        examples_per_group=examples_per_group,
        examples_total=sum(examples_per_group.values()),
        frames_per_example=frames_per_example,
        position_shape=[-1, *position_tail],
        init_width=init_width,
        init_ranges=init_ranges,
        video_bytes_min=min(video_sizes),
        video_bytes_max=max(video_sizes),
        video_bytes_mean=float(np.mean(video_sizes)),
    )
