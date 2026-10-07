# Specification

## State and rendering boundary

One physical trajectory is an immutable array of frame times, positions, and velocities. A renderer consumes that array and an appearance seed. It must not rerun or perturb physics.

## Identifiers

- `trajectory_id`: stable logical identifier for one physical state sequence;
- `trajectory_hash`: hash of dtype, shape, and state bytes;
- `render_id`: stable identifier for one rendering;
- `render_hash`: hash of rendered dtype, shape, and bytes;
- `related_pair_id`: declared similarity relation between independently generated trajectories;
- `source_id`: acquisition or procedural origin;
- `split_id`: frozen role assignment; and
- `exposure_id`: consumed-example and successful-update record.

## Relationship precedence

1. Identical render hash → `exact_render`;
2. Identical trajectory hash → `same_trajectory`;
3. Shared declared pair ID → `nearby_trajectory`;
4. Otherwise → `independent`.

Similarity does not create a derivation edge. A nearby trajectory is not a descendant.

## Simulator contract

- Unit-width arena;
- Two equal-radius and equal-mass disks;
- Exact continuous collision time;
- Elastic velocity exchange;
- No wall contact in accepted trajectories;
- Deterministic states for fixed inputs; and
- Conservation of horizontal momentum and kinetic energy.

This initial one-line collision model is a contract test, not the final research population.

## Metric contract

Center-position arrays use shape `[frames, disks, 2]` in arena-width units. Errors are capped per disk-frame. Invalid frames receive the cap and remain in the denominator.

## Reproducibility contract

Every result should record:

- Git SHA;
- Dependency lock hash;
- Full configuration;
- Data and manifest hashes;
- Random seeds;
- Requested and successful exposure counts;
- Model/checkpoint parentage;
- Evaluator version; and
- Failures and exclusions by reason.
