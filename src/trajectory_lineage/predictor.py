"""Cheap predictors used to validate the evaluation assay before model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class NearestTrajectoryPredictor:
    """Return the future belonging to the nearest observed state trajectory.

    This is a deliberate memory baseline, not a learned world model. It should be maximally
    sensitive to exact trajectory reuse and is therefore useful as a positive control.
    """

    prefixes: FloatArray | None = None
    futures: FloatArray | None = None

    def fit(self, prefixes: FloatArray, futures: FloatArray) -> "NearestTrajectoryPredictor":
        prefixes = np.asarray(prefixes, dtype=np.float64)
        futures = np.asarray(futures, dtype=np.float64)
        if prefixes.ndim != 4 or prefixes.shape[-2:] != (2, 2):
            raise ValueError("prefixes must have shape [examples, frames, 2, 2]")
        if futures.ndim != 4 or futures.shape[-2:] != (2, 2):
            raise ValueError("futures must have shape [examples, frames, 2, 2]")
        if prefixes.shape[0] != futures.shape[0] or prefixes.shape[0] == 0:
            raise ValueError("prefixes and futures must contain the same nonzero example count")
        self.prefixes = prefixes.copy()
        self.futures = futures.copy()
        return self

    def predict(self, prefixes: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self.prefixes is None or self.futures is None:
            raise RuntimeError("predictor must be fit before prediction")
        queries = np.asarray(prefixes, dtype=np.float64)
        if queries.ndim != 4 or queries.shape[1:] != self.prefixes.shape[1:]:
            raise ValueError("query prefix shape does not match fitted prefixes")

        query_flat = queries.reshape(len(queries), -1)
        bank_flat = self.prefixes.reshape(len(self.prefixes), -1)
        squared = np.square(query_flat[:, None, :] - bank_flat[None, :, :]).mean(axis=-1)
        nearest = np.argmin(squared, axis=1)
        return self.futures[nearest].copy(), np.sqrt(squared[np.arange(len(queries)), nearest])
