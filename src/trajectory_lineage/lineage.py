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
    """Provenance relationship between an evaluation clip and a consumed training example.

    Relationships come from recorded identities and derivation edges, never from content
    equality: rasterization is many-to-one, so equal pixels do not imply equal states, and equal
    state hashes between records with different identities do not imply derivation.
    """

    EXACT_RENDER = "exact_render"
    SAME_TRAJECTORY = "same_trajectory"
    DERIVED_TRAJECTORY = "derived_trajectory"
    NEARBY_TRAJECTORY = "nearby_trajectory"
    INDEPENDENT = "independent"
    UNRESOLVED = "unresolved"


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
    """Identities, content hashes, and provenance for one rendered clip.

    ``independently_generated`` defaults to false and must be asserted by a generator that
    documents independent sampling (as the simulator does); records without that assertion (for
    example, third-party video) resolve to ``UNRESOLVED`` rather than ``INDEPENDENT``.
    """

    trajectory_id: str
    trajectory_hash: str
    render_id: str
    render_hash: str
    appearance_seed: int
    parent_trajectory_id: str | None = None
    related_pair_id: str | None = None
    independently_generated: bool = False

    def relationship_to(self, other: "LineageRecord") -> Relationship:
        if self.render_id == other.render_id:
            return Relationship.EXACT_RENDER
        if self.trajectory_id == other.trajectory_id:
            return Relationship.SAME_TRAJECTORY
        if other.trajectory_id == self.parent_trajectory_id or (
            self.trajectory_id == other.parent_trajectory_id
        ):
            return Relationship.DERIVED_TRAJECTORY
        if self.parent_trajectory_id is not None and (
            self.parent_trajectory_id == other.parent_trajectory_id
        ):
            return Relationship.DERIVED_TRAJECTORY
        if self.related_pair_id is not None and self.related_pair_id == other.related_pair_id:
            return Relationship.NEARBY_TRAJECTORY
        if self.independently_generated and other.independently_generated:
            return Relationship.INDEPENDENT
        return Relationship.UNRESOLVED

    def content_equality(self, other: "LineageRecord") -> dict[str, bool]:
        """Content properties reported separately from provenance."""

        return {
            "render_bytes_equal": self.render_hash == other.render_hash,
            "state_equal": self.trajectory_hash == other.trajectory_hash,
        }


def write_manifest(path: str | Path, records: list[LineageRecord]) -> None:
    """Write deterministic JSON so the same records always produce the same manifest hash."""

    payload = [asdict(record) for record in sorted(records, key=lambda record: record.render_id)]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
