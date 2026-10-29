"""Frozen evaluation support and overlap weights (Protocol §2, "Length standardization").

Support: exact-length bins l = 3..24. A length is retained when the evaluation set has at least
``min_eval_per_outcome_per_length`` episodes of each outcome and the training pool has both outcomes
at that length. Support is fixed from labels and lengths only, before any model score exists.

Weights: with n_{y,l} retained evaluation episodes of outcome y at length l, n_y = sum_l n_{y,l},
r_{y,l} = n_{y,l} / n_y and the overlap target q_l = min(r_{0,l}, r_{1,l}) / sum_k min(r_{0,k}, r_{1,k}),
episode i gets a_i = pi_{Y_i} q_{L_i} / n_{Y_i, L_i}, where pi_1 = pi is the prevalence among all
eligible evaluation episodes and pi_0 = 1 - pi. Then sum_i a_i = 1, both outcomes share the length
distribution q, and the weighted truth is exactly pi. These are evaluation weights only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import load_config, sha256_json, write_json
from .design import LENGTHS
from .manifest import Manifest, load_manifest


def evaluation_support(m: Manifest, config: dict) -> dict:
    ev = config["evaluation"]
    te, tr = m.test_mask(), m.train_mask()
    pi = float(m.success[te].mean())
    counts = {y: np.array([int(((m.length == l) & te & (m.success == y)).sum()) for l in LENGTHS]) for y in (0, 1)}
    train_counts = {y: np.array([int(((m.length == l) & tr & (m.success == y)).sum()) for l in LENGTHS]) for y in (0, 1)}
    keep = np.ones(len(LENGTHS), bool)
    reasons: dict[int, list[str]] = {}
    for li, l in enumerate(LENGTHS):
        for y in (0, 1):
            if counts[y][li] < ev["min_eval_per_outcome_per_length"]:
                keep[li] = False
                reasons.setdefault(l, []).append(f"eval outcome {y}: {counts[y][li]} < {ev['min_eval_per_outcome_per_length']}")
            if train_counts[y][li] < ev["min_train_per_outcome_per_length"]:
                keep[li] = False
                reasons.setdefault(l, []).append(f"train outcome {y}: {train_counts[y][li]} < {ev['min_train_per_outcome_per_length']}")
    n_y = {y: int(counts[y][keep].sum()) for y in (0, 1)}
    if min(n_y.values()) == 0:
        raise ValueError("no supported lengths for one outcome")
    r = {y: counts[y] / n_y[y] for y in (0, 1)}
    overlap = np.where(keep, np.minimum(r[0], r[1]), 0.0)
    q = overlap / overlap.sum()
    pi_y = {1: pi, 0: 1 - pi}
    weights: dict[str, float] = {}
    for i in np.where(te)[0]:
        li = int(m.length[i]) - LENGTHS[0]
        if not keep[li]:
            continue
        y = int(m.success[i])
        weights[str(m.ids[i])] = float(pi_y[y] * q[li] / counts[y][li])
    a = np.array(list(weights.values()))
    y_w = np.array([m.success[m.index_of()[i]] for i in weights])
    ess = float(1.0 / (a ** 2).sum())
    ess_by = {y: float(a[y_w == y].sum() ** 2 / (a[y_w == y] ** 2).sum()) for y in (0, 1)}
    out = {
        "prevalence_pi": pi,
        "eligible_test_episodes": int(te.sum()),
        "retained_test_episodes": int(len(weights)),
        "excluded_test_episodes": int(te.sum() - len(weights)),
        "support_lengths": [l for l, k in zip(LENGTHS, keep) if k],
        "excluded_lengths": {str(l): v for l, v in reasons.items()},
        "eval_counts_by_length": {"success": counts[1].tolist(), "failure": counts[0].tolist(), "lengths": LENGTHS},
        "train_counts_by_length": {"success": train_counts[1].tolist(), "failure": train_counts[0].tolist()},
        "n_y_retained": {"success": n_y[1], "failure": n_y[0]},
        "q_by_length": dict(zip(map(str, LENGTHS), q.tolist())),
        "weight_sum": float(a.sum()),
        "weighted_truth": float((a * y_w).sum()),
        "ess": ess,
        "ess_by_outcome": {"success": ess_by[1], "failure": ess_by[0]},
        "weights": weights,
    }
    out["sha256"] = sha256_json({k: v for k, v in out.items() if k != "sha256"})
    assert abs(out["weight_sum"] - 1) < 1e-9 and abs(out["weighted_truth"] - pi) < 1e-9
    return out


def support_table(s: dict) -> str:
    rows = ["| Length | Eval S | Eval F | Train S | Train F | Retained | q_l |", "|---|---|---|---|---|---|---|"]
    for li, l in enumerate(LENGTHS):
        kept = l in s["support_lengths"]
        rows.append(f"| {l} | {s['eval_counts_by_length']['success'][li]} | {s['eval_counts_by_length']['failure'][li]} | "
                    f"{s['train_counts_by_length']['success'][li]} | {s['train_counts_by_length']['failure'][li]} | "
                    f"{'yes' if kept else 'no'} | {s['q_by_length'][str(l)]:.4f} |")
    rows.append(f"\nπ = {s['prevalence_pi']:.6f}; retained {s['retained_test_episodes']} of {s['eligible_test_episodes']} evaluation episodes; "
                f"ESS = {s['ess']:.0f} (successes {s['ess_by_outcome']['success']:.0f}, failures {s['ess_by_outcome']['failure']:.0f}); "
                f"Σa = {s['weight_sum']:.6f}; weighted truth = {s['weighted_truth']:.6f}.")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Freeze the evaluation support and weights from the manifest.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    m, digest = load_manifest(Path(args.manifest))
    s = evaluation_support(m, load_config(args.config))
    s["manifest_sha256"] = digest
    s["manifest_kind"] = m.header.get("kind")
    write_json(Path(args.out), s)
    Path(args.out).with_suffix(".md").write_text(support_table(s) + "\n")
    print(json.dumps({k: v for k, v in s.items() if k not in ("weights", "q_by_length", "eval_counts_by_length", "train_counts_by_length")}, indent=1))
    print(support_table(s))


if __name__ == "__main__":
    main()
