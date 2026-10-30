"""Episode manifest: the metadata-only view of the Kuka cache that every registered step is built from.

An episode is identified by ``"<shard:05d>/<index>"`` (its position in the shard's npz arrays). The
manifest records length (frames), transitions (frames - 1), recorded success, shard, split and the
proprioception flag. Building it reads only the small ``length``/``success``/``proprio_ok`` arrays, so
it takes seconds and needs no features in memory.

Two constructors:

* ``build_manifest(npz_dir)``   from the real cache ``data/kuka/shard_*.npz`` (owner's machine);
* ``synthetic_manifest(...)``   a stand-in with the real pool sizes and the audited length-by-outcome
  shape, used only for dry runs and for checking the samplers at scale. It is labelled as synthetic
  in its header and must never be used to train or to sign a freeze.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sha256_bytes, write_json

LENGTH_MIN, LENGTH_MAX = 3, 24


def is_test_shard(shard: np.ndarray | int, divisor: int = 3, every: int = 5) -> np.ndarray | bool:
    """The owner's split rule: test = every 5th sampled shard (``shard_split`` in ``wm.py``)."""

    return (np.asarray(shard) // divisor) % every == 0


@dataclass
class Manifest:
    ids: np.ndarray  # str
    shard: np.ndarray  # int
    length: np.ndarray  # int frames
    success: np.ndarray  # int 0/1
    split: np.ndarray  # "train" | "test"
    proprio_ok: np.ndarray  # bool
    header: dict

    @property
    def transitions(self) -> np.ndarray:
        return self.length - 1

    def __len__(self) -> int:
        return len(self.ids)

    def eligible(self, lo: int = LENGTH_MIN, hi: int = LENGTH_MAX) -> np.ndarray:
        return (self.length >= lo) & (self.length <= hi)

    def train_mask(self) -> np.ndarray:
        return (self.split == "train") & self.eligible()

    def test_mask(self) -> np.ndarray:
        return (self.split == "test") & self.eligible()

    def index_of(self) -> dict[str, int]:
        return {i: k for k, i in enumerate(self.ids.tolist())}

    def to_rows(self) -> list[dict]:
        return [
            {"id": i, "shard": int(s), "length": int(l), "transitions": int(l) - 1, "success": int(y),
             "split": sp, "proprio_ok": bool(p)}
            for i, s, l, y, sp, p in zip(self.ids, self.shard, self.length, self.success, self.split, self.proprio_ok)
        ]

    def summary(self) -> dict:
        tr, te = self.train_mask(), self.test_mask()
        return {
            "episodes": int(len(self)),
            "eligible": int(self.eligible().sum()),
            "train": {"episodes": int(tr.sum()), "successes": int(self.success[tr].sum()),
                      "failures": int((1 - self.success[tr]).sum()),
                      "transitions": int(self.transitions[tr].sum())},
            "test": {"episodes": int(te.sum()), "successes": int(self.success[te].sum()),
                     "prevalence": float(self.success[te].mean()) if te.any() else None},
            "length_range": [int(self.length.min()), int(self.length.max())] if len(self) else None,
        }


def _canonical_bytes(header: dict, rows: list[dict]) -> bytes:
    lines = [json.dumps({"header": header}, sort_keys=True, separators=(",", ":"))]
    lines += [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def save_manifest(m: Manifest, path: Path) -> str:
    """Write JSONL (header line first, one episode per line). Returns the SHA-256 of the bytes."""

    data = _canonical_bytes(m.header, m.to_rows())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def load_manifest(path: Path) -> tuple[Manifest, str]:
    data = Path(path).read_bytes()
    lines = data.decode("utf-8").splitlines()
    header = json.loads(lines[0])["header"]
    rows = [json.loads(l) for l in lines[1:] if l.strip()]
    m = Manifest(
        ids=np.array([r["id"] for r in rows], dtype=object),
        shard=np.array([r["shard"] for r in rows], dtype=int),
        length=np.array([r["length"] for r in rows], dtype=int),
        success=np.array([r["success"] for r in rows], dtype=int),
        split=np.array([r["split"] for r in rows], dtype=object),
        proprio_ok=np.array([r["proprio_ok"] for r in rows], dtype=bool),
        header=header,
    )
    return m, sha256_bytes(data)


def build_manifest(npz_dir: Path, divisor: int = 3, every: int = 5) -> Manifest:
    """Read lengths and labels from every ``shard_*.npz`` (features are not loaded)."""

    files = sorted(glob.glob(str(Path(npz_dir) / "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shard_*.npz under {npz_dir}")
    ids, shards, lengths, succ, prop = [], [], [], [], []
    for f in files:
        sid = int(Path(f).stem.split("_")[1])
        with np.load(f) as npz:
            ln, sc = npz["length"], npz["success"]
            ok = npz["proprio_ok"] if "proprio_ok" in npz.files else np.ones(len(ln), bool)
        for e in range(len(ln)):
            ids.append(f"{sid:05d}/{e}")
            shards.append(sid)
            lengths.append(int(ln[e]))
            succ.append(int(sc[e]))
            prop.append(bool(ok[e]))
    shard = np.array(shards, int)
    split = np.where(is_test_shard(shard, divisor, every), "test", "train").astype(object)
    # no folder path in the header: the manifest bytes (and so its SHA-256) depend only on the episodes
    header = {"kind": "real", "shards": len(files), "split_rule": f"(shard // {divisor}) % {every} == 0 is test",
              "eligibility": {"length_min": LENGTH_MIN, "length_max": LENGTH_MAX}}
    return Manifest(np.array(ids, dtype=object), shard, np.array(lengths, int), np.array(succ, int), split, np.array(prop, bool), header)


# ---------------------------------------------------------------- synthetic stand-in

def audited_length_table(audit_jsonl: Path) -> dict[int, np.ndarray]:
    """Length counts by outcome from the owner's audit sample (``results/episodes/kuka.jsonl``).

    Returns {0: counts over lengths 3..24, 1: counts}. Laplace smoothing (+0.25) keeps every
    eligible length possible, so the synthetic pool is not artificially infeasible."""

    rows = [json.loads(l) for l in Path(audit_jsonl).read_text().splitlines() if l.strip()]
    table = {y: np.full(LENGTH_MAX - LENGTH_MIN + 1, 0.25) for y in (0, 1)}
    for r in rows:
        L = int(r["n_steps"])
        if LENGTH_MIN <= L <= LENGTH_MAX:
            table[int(r["success"])][L - LENGTH_MIN] += 1
    return table


def synthetic_manifest(audit_jsonl: Path, *, train_successes: int = 3989, train_failures: int = 6900,
                       test_successes: int = 996, test_failures: int = 1804, seed: int = 20261023) -> Manifest:
    """A metadata-only stand-in with the real pool sizes and the audited length shape.

    Pool sizes: the owner's log shows 3,989 train successes and 2,800 test episodes with 996
    successes; the 3,000 train failures in that log were a memory cap, and the README reports
    13.7k extracted episodes, so about 6,900 train failures exist. The failure count is an
    assumption and is stated in the header."""

    rng = np.random.default_rng(seed)
    table = audited_length_table(audit_jsonl)
    lengths = np.arange(LENGTH_MIN, LENGTH_MAX + 1)
    ids, shard, length, success, split = [], [], [], [], []

    def draw(n: int, y: int, is_test: bool, shard_start: int):
        p = table[y] / table[y].sum()
        L = rng.choice(lengths, size=n, p=p)
        # distribute over synthetic shards that obey the split rule
        cand = [s for s in range(shard_start, shard_start + 3000) if bool(is_test_shard(s)) == is_test]
        sh = rng.choice(cand[:400], size=n)
        for k in range(n):
            ids.append(None)  # filled below with a per-shard index
            shard.append(int(sh[k]))
            length.append(int(L[k]))
            success.append(y)
            split.append("test" if is_test else "train")

    draw(train_successes, 1, False, 0)
    draw(train_failures, 0, False, 0)
    draw(test_successes, 1, True, 0)
    draw(test_failures, 0, True, 0)
    shard_arr = np.array(shard, int)
    order = np.lexsort((np.arange(len(shard_arr)), shard_arr))
    counters: dict[int, int] = {}
    id_arr = np.empty(len(shard_arr), dtype=object)
    for k in order:
        s = int(shard_arr[k])
        id_arr[k] = f"{s:05d}/{counters.get(s, 0)}"
        counters[s] = counters.get(s, 0) + 1
    header = {"kind": "SYNTHETIC", "warning": "metadata stand-in for dry runs only; never train or freeze on it",
              "assumed_counts": {"train_successes": train_successes, "train_failures": train_failures,
                                                               "test_successes": test_successes, "test_failures": test_failures},
              "split_rule": "(shard // 3) % 5 == 0 is test", "eligibility": {"length_min": LENGTH_MIN, "length_max": LENGTH_MAX}, "seed": seed}
    m = Manifest(id_arr, shard_arr, np.array(length, int), np.array(success, int), np.array(split, dtype=object),
                 np.ones(len(shard_arr), bool), header)
    # canonical order: by shard then index
    o = np.lexsort((np.array([int(i.split("/")[1]) for i in m.ids]), m.shard))
    return Manifest(m.ids[o], m.shard[o], m.length[o], m.success[o], m.split[o], m.proprio_ok[o], header)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the real manifest from data/kuka/shard_*.npz")
    b.add_argument("--npz-dir", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("synthetic", help="build the SYNTHETIC stand-in from the audit sample")
    s.add_argument("--audit-jsonl", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--train-failures", type=int, default=6900)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        m = build_manifest(Path(args.npz_dir))
    else:
        m = synthetic_manifest(Path(args.audit_jsonl), train_failures=args.train_failures)
    digest = save_manifest(m, Path(args.out))
    source = args.npz_dir if args.cmd == "build" else args.audit_jsonl
    write_json(Path(args.out).with_suffix(".summary.json"), {"sha256": digest, **m.summary(), "header": m.header, "source": str(source)})
    print(json.dumps({"sha256": digest, **m.summary()}, indent=1))


if __name__ == "__main__":
    main()
