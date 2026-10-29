from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from registered import load_config  # noqa: E402
from registered.design import build_selection  # noqa: E402
from registered.manifest import LENGTH_MAX, LENGTH_MIN, Manifest  # noqa: E402
from registered.weights import evaluation_support  # noqa: E402


def make_manifest(rng: np.random.Generator, train_s=3989, train_f=6900, test_s=996, test_f=1804, failure_timeout_spike=0.45) -> Manifest:
    """Small synthetic manifest with a Kuka-like shape: successes short, failures long with a spike at 21."""

    lengths = np.arange(LENGTH_MIN, LENGTH_MAX + 1)
    p_s = np.exp(-0.5 * ((lengths - 10) / 3.5) ** 2); p_s /= p_s.sum()
    p_f = np.exp(-0.5 * ((lengths - 14) / 4.5) ** 2); p_f[lengths == 21] += failure_timeout_spike * p_f.sum() / (1 - failure_timeout_spike); p_f /= p_f.sum()
    rows = []
    for n, y, split, p in ((train_s, 1, "train", p_s), (train_f, 0, "train", p_f), (test_s, 1, "test", p_s), (test_f, 0, "test", p_f)):
        L = rng.choice(lengths, size=n, p=p)
        for l in L:
            rows.append((y, split, int(l)))
    shards_train = [s for s in range(600) if (s // 3) % 5 != 0]
    shards_test = [s for s in range(600) if (s // 3) % 5 == 0]
    ids, shard, length, success, split = [], [], [], [], []
    counters: dict[int, int] = {}
    for y, sp, l in rows:
        s = int(rng.choice(shards_test if sp == "test" else shards_train))
        ids.append(f"{s:05d}/{counters.get(s, 0)}")
        counters[s] = counters.get(s, 0) + 1
        shard.append(s); length.append(l); success.append(y); split.append(sp)
    return Manifest(np.array(ids, dtype=object), np.array(shard), np.array(length), np.array(success), np.array(split, dtype=object),
                    np.ones(len(ids), bool), {"kind": "SYNTHETIC"})


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def manifest():
    return make_manifest(np.random.default_rng(1))


@pytest.fixture(scope="session")
def selection(manifest, config):
    return build_selection(manifest, "testmanifest", config)


@pytest.fixture(scope="session")
def support(manifest, config):
    s = evaluation_support(manifest, config)
    s["manifest_sha256"] = "testmanifest"
    return s
