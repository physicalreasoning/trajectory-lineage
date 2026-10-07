import numpy as np

from trajectory_lineage import NearestTrajectoryPredictor


def test_nearest_predictor_recovers_seen_future() -> None:
    prefixes = np.array(
        [
            [[[0.0, 0.0], [1.0, 0.0]]],
            [[[0.2, 0.0], [0.8, 0.0]]],
        ]
    )
    futures = np.array(
        [
            [[[0.1, 0.0], [0.9, 0.0]]],
            [[[0.3, 0.0], [0.7, 0.0]]],
        ]
    )
    predictor = NearestTrajectoryPredictor().fit(prefixes, futures)
    predicted, distance = predictor.predict(prefixes[[1]])

    assert np.array_equal(predicted, futures[[1]])
    assert distance[0] == 0.0
