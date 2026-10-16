from trajectory_lineage import LineageRecord, Relationship


def record(
    trajectory_id: str,
    render_id: str,
    *,
    render_hash: str = "pixels",
    trajectory_hash: str = "state",
    parent: str | None = None,
    pair: str | None = None,
    independent: bool = True,
) -> LineageRecord:
    return LineageRecord(
        trajectory_id=trajectory_id,
        trajectory_hash=trajectory_hash,
        render_id=render_id,
        render_hash=render_hash,
        appearance_seed=1,
        parent_trajectory_id=parent,
        related_pair_id=pair,
        independently_generated=independent,
    )


def test_relationships_follow_provenance() -> None:
    base = record("t1", "t1-r1", pair="pair-a")
    assert base.relationship_to(record("t1", "t1-r1")) == Relationship.EXACT_RENDER
    assert base.relationship_to(record("t1", "t1-r2")) == Relationship.SAME_TRAJECTORY
    assert base.relationship_to(record("t2", "t2-r1", parent="t1")) == (
        Relationship.DERIVED_TRAJECTORY
    )
    assert base.relationship_to(record("t3", "t3-r1", pair="pair-a")) == (
        Relationship.NEARBY_TRAJECTORY
    )
    assert base.relationship_to(record("t4", "t4-r1")) == Relationship.INDEPENDENT
    assert base.relationship_to(record("t5", "t5-r1", independent=False)) == (
        Relationship.UNRESOLVED
    )


def test_content_equality_does_not_imply_lineage() -> None:
    """Round-3 counterexamples: equal hashes between unrelated identities are not ancestry."""

    base = record("t1", "t1-r1", render_hash="pixels-a", trajectory_hash="state-a")
    same_pixels = record("t2", "t2-r1", render_hash="pixels-a", trajectory_hash="state-b")
    same_state = record("t3", "t3-r1", render_hash="pixels-c", trajectory_hash="state-a")
    assert base.relationship_to(same_pixels) == Relationship.INDEPENDENT
    assert base.content_equality(same_pixels)["render_bytes_equal"]
    assert base.relationship_to(same_state) == Relationship.INDEPENDENT
    assert base.content_equality(same_state)["state_equal"]
