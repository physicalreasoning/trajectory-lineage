from pathlib import Path

import h5py
import numpy as np
import pytest

from trajectory_lineage.phyworld import inspect_phyworld


def write_fixture(path: Path, *, omit: str | None = None, misalign: bool = False) -> None:
    with h5py.File(path, "w") as handle:
        for group_name in ("init_streams", "position_streams", "video_streams"):
            if group_name == omit:
                continue
            group = handle.create_group(group_name)
            count = 3 if misalign and group_name == "video_streams" else 2
            if group_name == "init_streams":
                group.create_dataset(
                    "00000",
                    data=np.array([[0.3, 0.4, 1.0, 2.0], [0.5, 0.6, 3.0, 4.0]]),
                )
            elif group_name == "position_streams":
                group.create_dataset("00000", data=np.zeros((2, 32, 2, 2)))
            else:
                values = np.empty(count, dtype=object)
                for index in range(count):
                    values[index] = np.arange(index + 3, dtype=np.uint8)
                group.create_dataset(
                    "00000",
                    data=values,
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )


def test_inspect_phyworld_validates_and_summarizes(tmp_path: Path) -> None:
    dataset = tmp_path / "collision_eval.hdf5"
    write_fixture(dataset)

    summary = inspect_phyworld(dataset)

    assert summary.examples_total == 2
    assert summary.frames_per_example == 32
    assert summary.position_shape == [-1, 32, 2, 2]
    assert summary.init_ranges["radius_1"] == [0.3, 0.5]
    assert summary.video_bytes_min == 3
    assert summary.video_bytes_max == 4
    assert len(summary.sha256) == 64


def test_inspect_phyworld_rejects_missing_group(tmp_path: Path) -> None:
    dataset = tmp_path / "missing.hdf5"
    write_fixture(dataset, omit="video_streams")

    with pytest.raises(ValueError, match="missing required HDF5 groups"):
        inspect_phyworld(dataset)


def test_inspect_phyworld_rejects_misaligned_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "misaligned.hdf5"
    write_fixture(dataset, misalign=True)

    with pytest.raises(ValueError, match="example counts differ"):
        inspect_phyworld(dataset)
