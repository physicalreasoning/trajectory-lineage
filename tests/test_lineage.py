from trajectory_lineage import LineageRecord, Relationship


def record(
    render_id: str,
    render_hash: str,
    trajectory_hash: str,
    related_pair_id: str | None = None,
) -> LineageRecord:
    return LineageRecord(
        trajectory_id=f"trajectory-{trajectory_hash}",
        trajectory_hash=trajectory_hash,
        render_id=render_id,
        render_hash=render_hash,
        appearance_seed=1,
        related_pair_id=related_pair_id,
    )


def test_relationship_precedence_is_exact_then_trajectory_then_related() -> None:
    base = record("r1", "pixels-a", "state-a", "pair-a")
    assert base.relationship_to(record("r2", "pixels-a", "state-b")) == Relationship.EXACT_RENDER
    assert base.relationship_to(record("r3", "pixels-b", "state-a")) == Relationship.SAME_TRAJECTORY
    assert (
        base.relationship_to(record("r4", "pixels-c", "state-c", "pair-a"))
        == Relationship.NEARBY_TRAJECTORY
    )
    assert (
        base.relationship_to(record("r5", "pixels-d", "state-d", "pair-b"))
        == Relationship.INDEPENDENT
    )
