# specification

## state and rendering boundary

one physical trajectory is an immutable array of frame times, positions, and velocities. a renderer consumes that array and an appearance seed. it must not rerun or perturb physics.

## identifiers

- `trajectory_id`: stable logical identifier for one physical state sequence;
- `trajectory_hash`: hash of dtype, shape, and state bytes;
- `render_id`: stable identifier for one rendering;
- `render_hash`: hash of rendered dtype, shape, and bytes;
- `related_pair_id`: declared similarity relation between independently generated trajectories;
- `source_id`: acquisition or procedural origin;
- `split_id`: frozen role assignment; and
- `exposure_id`: consumed-example and successful-update record.

## relationship precedence

1. identical render hash → `exact_render`;
2. identical trajectory hash → `same_trajectory`;
3. shared declared pair id → `nearby_trajectory`;
4. otherwise → `independent`.

similarity does not create a derivation edge. a nearby trajectory is not a descendant.

## simulator contract

- unit-width arena;
- two equal-radius and equal-mass disks;
- exact continuous collision time;
- elastic velocity exchange;
- no wall contact in accepted trajectories;
- deterministic states for fixed inputs; and
- conservation of horizontal momentum and kinetic energy.

this initial one-line collision model is a contract test, not the final research population.

## metric contract

center-position arrays use shape `[frames, disks, 2]` in arena-width units. errors are capped per disk-frame. invalid frames receive the cap and remain in the denominator.

## reproducibility contract

every result should record:

- git sha;
- dependency lock hash;
- full configuration;
- data and manifest hashes;
- random seeds;
- requested and successful exposure counts;
- model/checkpoint parentage;
- evaluator version; and
- failures and exclusions by reason.
