"""Generate one trajectory, two rerenders, and a lineage manifest."""

from __future__ import annotations

import json
from pathlib import Path

from trajectory_lineage import InitialState, LineageRecord, hash_array, render, simulate
from trajectory_lineage.lineage import write_manifest


def main() -> None:
    state = InitialState(x_left=0.2, x_right=0.7, vx_left=0.55, vx_right=-0.10)
    trajectory = simulate(state, frames=24, dt=0.04)
    first = render(trajectory, appearance_seed=7)
    second = render(trajectory, appearance_seed=19)
    state_hash = hash_array(trajectory.positions)

    records = [
        LineageRecord(
            trajectory_id="trajectory-0001",
            trajectory_hash=state_hash,
            render_id="trajectory-0001-render-0007",
            render_hash=hash_array(first),
            appearance_seed=7,
        ),
        LineageRecord(
            trajectory_id="trajectory-0001",
            trajectory_hash=state_hash,
            render_id="trajectory-0001-render-0019",
            render_hash=hash_array(second),
            appearance_seed=19,
        ),
    ]

    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    write_manifest(result_dir / "smoke_manifest.json", records)
    result = {
        "collision_time": trajectory.collision_time,
        "frames": len(trajectory.frame_times),
        "same_trajectory_hash": records[0].trajectory_hash == records[1].trajectory_hash,
        "different_render_hash": records[0].render_hash != records[1].render_hash,
        "relationship": records[0].relationship_to(records[1]).value,
    }
    (result_dir / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
