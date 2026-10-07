"""Immutable lineage records for trajectories, renders, clips, and exposures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Relationship(StrEnum):
    """Relationship between an evaluation clip and a consumed training example."""

    EXACT_RENDER = "exact_render"
    SAME_TRAJECTORY = "same_trajectory"
    NEARBY_TRAJECTORY = "nearby_trajectory"
    INDEPENDENT = "independent"


def hash_array(value: NDArray[Any]) -> str:
    """Hash dtype, shape, and contiguous bytes instead of bytes alone."""

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LineageRecord:
    trajectory_id: str
    trajectory_hash: str
    render_id: str
    render_hash: str
    appearance_seed: int
    parent_trajectory_id: str | None = None
    related_pair_id: str | None = None

    def relationship_to(self, other: "LineageRecord") -> Relationship:
        if self.render_hash == other.render_hash:
            return Relationship.EXACT_RENDER
        if self.trajectory_hash == other.trajectory_hash:
            return Relationship.SAME_TRAJECTORY
        if self.related_pair_id is not None and self.related_pair_id == other.related_pair_id:
            return Relationship.NEARBY_TRAJECTORY
        return Relationship.INDEPENDENT


def write_manifest(path: str | Path, records: list[LineageRecord]) -> None:
    """Write deterministic JSON so the same records always produce the same manifest hash."""

    payload = [asdict(record) for record in sorted(records, key=lambda record: record.render_id)]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
