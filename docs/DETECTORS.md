# Duplicate-Detector Audit

Responds to Jiankai Sun’s request to “include perceptual and motion-based duplicate detectors as baselines.”

## Question

Do exact, perceptual, and motion-based duplicate detectors recognize evaluation clips that replay a training trajectory under a new rendering, while sparing independent events—including independent events closer to a training trajectory than any other training clip?

Ground truth is provenance from the simulator, never similarity. A thresholded state distance is a privileged *reference*, not a provenance oracle.

## Observation Contract

Detectors compare full 24-frame evaluation clips with full training clips, rendered natively at 224 px from stored states and appearance seeds. This is the curation setting: curators see whole videos. It is not the 64 px, 8-frame prefix the ConvLSTM consumes, so thresholds from this audit do not transfer automatically to the exposure assay; the planned filtering analysis recalibrates on separate calibration blocks with the confirmation gallery (base corpus plus continuation events).

## Protocol (`experiments/detector_benchmark_v3.json`)

Per independently generated split (calibration, test):

- **Gallery:** 500 training trajectories.
- **Positives:** 200 gallery trajectories re-rendered under a transformation.
- **Hard negatives:** 200 independent events closer to a target gallery trajectory than its nearest other gallery trajectory (0.25–0.5×), drawn with the validated closer-neighbor sampler.
- **Easy negatives:** 600 independent events (the same 600 families appear under every transformation).
- **Background:** 500 unrelated clips for VSC-style score normalization.

Transformations: `copy_reencode` (the identical gallery render re-encoded as JPEG q75 — the conventional-copy positive control), appearance, texture, compression (JPEG q25), crop (80%), mirror, time shift (3 frames), distractor.

Detectors: exact hash; SSCD and DINOv2 (pooled, time-aligned, background-normalized); color-invariant silhouettes; centroid tracks with and without mirror and time-offset search; RAFT-small flow with and without search and normalization; state-distance reference.

Query score: maximum similarity over the whole gallery. One threshold per detector and operating point (1%, 5%): the higher-method quantile of pooled calibration easy-negative scores across transformations. This targets the pooled transformation mixture, not each transformation, so per-transformation false removal is always reported.

## Interpretive Gate

A detector’s results support conclusions about trajectory reuse only if it passes the copy control: AUROC versus easy negatives ≥ 0.95 and recall ≥ 0.80 at 5%. The gate applies to motion detectors as well as perceptual ones. Failing detectors are reported as not functional in this rendering domain.

## Analyses (`scripts/analyze_detectors.py`, from saved per-query scores)

- Recall and false removal per transformation, with Wilson intervals and counts.
- Family-bootstrap intervals (1,000 draws) that resample calibration easy families jointly across transformations *through threshold estimation*, and test families per role.
- Recall when hard-negative removal is also constrained to the operating rate (threshold = max of easy and hard calibration quantiles).
- The state-distance reference at a zero threshold, separating operating-point effects from genuine inseparability.
- AUROC versus easy and hard negatives, tie-aware median rank, and strict top-1 retrieval.

## History

| Run | Outcome |
|---|---|
| v1 (64 px) | Engineering run; pairwise-band hard negatives were no closer than each query’s nearest gallery item |
| v2 (64 px) | Copy positive control failed: upsampled JPEG blocks dominated SSCD (copy cosine 0.64 vs rerender 0.94); also a 1/64 mirror-mapping offset and untied ranks. Perceptual and RAFT results uninterpretable; preserved as `results/detector_benchmark_v2_failed_control.json` |
| v3 (224 px) | Registered protocol above; first launch stopped and restarted with identical configuration to save per-query scores |

## Planned Next

- Faithful reimplementations of published GSSCD/VSSCD and OFS-k pipelines and the VSC 2022 temporal matcher.
- A deployed curation embedding (InternVideo2 or Cosmos-Embed1).
- Larger independent negative pools for precise 1% claims.
- Filtering analysis on the confirmation populations after unblinding: common masks across arms, retained prevalence `π_τ = π r_S / (π r_S + (1 − π) r_I)`, residual distortion `π_τ G_τ`, component errors, retained event and difficulty coverage, compared with equally sized random removal and provenance-family exclusion.
- DROID multi-view detector validation (real same-event positives).
