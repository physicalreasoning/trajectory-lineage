# Overlap-Detector Evaluation Plan

## Goal

Measure which training–evaluation relationships are detected by increasingly strong overlap safeguards, and what independent evaluation coverage each safeguard removes.

## Ground-Truth Relationships

The controlled benchmark contains:

- Exact renders
- Appearance-only rerenders of one stored trajectory
- Nearby but independently generated trajectories
- Unrestricted independently sampled events

Ground truth comes from the simulator’s provenance records. Similarity scores are detector outputs, not lineage labels.

## Detector Ladder

### 1. Exact Matching

- File hash
- Rendered-tensor hash
- Per-frame hash

### 2. Classical Perceptual Matching

- Color-invariant silhouette descriptor
- Perceptual hash over sampled frames
- Temporally aligned frame distance

### 3. Learned Perceptual Matching

- SSCD frame embeddings
- DINOv2 frame embeddings
- Video-level pooled embeddings

### 4. Motion Matching

- Foreground-centroid trajectories
- Optical-flow descriptors
- Dynamic time warping over motion features

### 5. Oracle Simulation Matching

- Stored-state trajectory distance
- Initial-state distance
- Event-class match

The oracle is used to measure detector recall and false removal; it is not available for ordinary third-party video.

## Threshold Selection

Split source families into detector-development and detector-test sets.

For each detector:

1. Fit preprocessing and thresholds on development families only.
2. Freeze an operating point at a declared independent-event false-removal rate.
3. Evaluate once on detector-test families.
4. Report sensitivity across several frozen operating points.

## Metrics

- Exact-render recall
- Same-trajectory rerender recall
- Nearby-trajectory false-positive rate
- Independent-event false-removal rate
- Precision at the declared evaluation prevalence
- Retained event-class coverage
- Retained difficulty coverage
- Residual source-specific generalization gap after filtering

## Interpretation

A detector is useful when it removes consequential same-trajectory reuse while preserving independent events and the benchmark’s physical coverage. High recall without false-removal and coverage accounting is insufficient.

## Current Groundwork

Implemented locally:

- Exact video hashing
- Color-invariant silhouette descriptors
- Foreground-centroid motion descriptors
- Oracle state-trajectory distance with object-swap symmetry

Planned next:

- Development/test lineage benchmark generation
- Threshold calibration
- SSCD and DINOv2 baselines
- Optical-flow and dynamic-time-warping baselines
- Detector-effect analysis on the learned ConvLSTM exposure pilot
