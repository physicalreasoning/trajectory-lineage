"""Seeded samplers for designs O, A, B and V, and the feasibility certificate.

All samplers work on the metadata manifest only (no features), so the whole selection for all 110
runs is built and certified in seconds, before any training, and can be frozen by hash.

Pairing. For a seed, each outcome pool (train successes, train failures) is permuted once with
``numpy.random.default_rng(seed)``. Every arm of that seed starts from prefixes of these two
permutations, so arms within a seed share episodes as far as their constraints allow, which is the
pairing structure the paired t analysis relies on. Different seeds use different permutations.

Design O  original episode counts (prefixes), as the owner's ``make_arm``.
Design A  exactly T distinct transitions at episode failure fraction phi. The total episode count
          N is a multiple of 20 so phi*N is an integer for every phi in {0, .10, .25, .50, .65}.
          For each candidate N the prefix totals are computed; the N whose total is closest to T is
          repaired to exactly T by same-outcome swaps (a selected episode is exchanged for an
          unselected one whose transition count moves the total toward T; each swap strictly
          reduces the gap, so the loop terminates). Swaps never change phi.
Design B  for lengths l = 3..24, 20*k_l episodes: 18 S + 2 F at phi = .10 and 7 S + 13 F at
          phi = .65. Per-length caps k_l <= min(S_l // 18, F_l // 13) make both arms feasible. The
          vector k is the integer least-squares solution closest to the pooled availability shape
          with sum_l k_l (l - 1) = T / 20 (dynamic programme, exact). Then, within each arm, the
          normalized outcome-conditional length distributions are identical by construction
          (k_l / sum k for both outcomes), and the two arms have the same total histogram 20*k_l.
          If T = 31,600 is infeasible, the budget falls back by the registered rule and Design A
          comparators at that budget are added for both B fractions.
Design V  shared base = first 1,500 successes of the seed; add distinct transitions of further
          successes (V+S) or of failures (V+F) until the arm holds exactly 31,600 transitions (Design
          A's budget), each by prefix + repair. V arms are therefore transition-matched to A.

The certificate (``certify``) re-derives every constraint from the manifest instead of trusting the
sampler's own bookkeeping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import load_config, sha256_json, write_json
from .manifest import LENGTH_MAX, LENGTH_MIN, Manifest, load_manifest

LENGTHS = list(range(LENGTH_MIN, LENGTH_MAX + 1))


class Infeasible(Exception):
    """A registered constraint cannot be met with the available pool."""


@dataclass
class Run:
    run_id: str
    design: str
    arm: str
    phi: float | None
    seed: int
    target_transitions: int | None
    n_success: int
    n_failure: int
    n_episodes: int
    transitions: int
    hist_success: list[int]  # counts over LENGTHS
    hist_failure: list[int]
    ids_sha256: str
    swaps: int = 0
    notes: dict = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- pools and pairing

def train_pools(m: Manifest) -> tuple[np.ndarray, np.ndarray]:
    """Manifest row indices of eligible train successes and failures, in canonical order."""

    tr = m.train_mask()
    return np.where(tr & (m.success == 1))[0], np.where(tr & (m.success == 0))[0]


def seeded_permutations(S: np.ndarray, F: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return S[rng.permutation(len(S))], F[rng.permutation(len(F))]


def _hist(m: Manifest, rows: list[int] | np.ndarray, y: int) -> list[int]:
    rows = np.asarray(rows, int)
    sel = rows[m.success[rows] == y]
    c = Counter(m.length[sel].tolist())
    return [int(c.get(l, 0)) for l in LENGTHS]


def _make_run(m: Manifest, design: str, arm: str, phi: float | None, seed: int, rows: list[int], target: int | None,
              swaps: int = 0, notes: dict | None = None) -> Run:
    rows_arr = np.asarray(rows, int)
    ids = sorted(m.ids[rows_arr].tolist())
    n_s = int(m.success[rows_arr].sum())
    return Run(
        run_id=f"{design}_{arm}_s{seed}", design=design, arm=arm, phi=phi, seed=seed, target_transitions=target,
        n_success=n_s, n_failure=int(len(rows_arr) - n_s), n_episodes=int(len(rows_arr)),
        transitions=int(m.transitions[rows_arr].sum()), hist_success=_hist(m, rows_arr, 1), hist_failure=_hist(m, rows_arr, 0),
        ids_sha256=sha256_json(ids), swaps=swaps, notes=notes or {}, ids=ids,
    )


# ---------------------------------------------------------------- repair

def _best_swap(classes: list[tuple[list[int], list[int]]], trans: np.ndarray, delta: int):
    """The single same-class swap that brings ``delta`` closest to zero and strictly reduces |delta|.

    Ties are broken deterministically: smallest residual, then class order, then smaller incoming
    value, then smaller outgoing value. Returns (class, pos_out, pos_in, change) or None."""

    best = None
    for k, (sel, unsel) in enumerate(classes):
        if not sel or not unsel:
            continue
        vs, vu = np.unique(trans[sel]), np.unique(trans[unsel])
        change = vu[:, None] - vs[None, :]
        resid = np.abs(delta - change)
        ok = resid < abs(delta)
        if not ok.any():
            continue
        i, j = np.unravel_index(int(np.argmin(np.where(ok, resid, np.inf))), resid.shape)
        cand = (int(resid[i, j]), k, int(vu[i]), int(vs[j]))
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    _, k, v_in, v_out = best
    sel, unsel = classes[k]
    pos_out = max(p for p, e in enumerate(sel) if trans[e] == v_out)      # most recently added
    pos_in = min(p for p, e in enumerate(unsel) if trans[e] == v_in)      # next in permutation order
    return k, pos_out, pos_in, v_in - v_out


def repair_to_target(classes: list[tuple[list[int], list[int]]], trans: np.ndarray, target: int,
                     max_swaps: int = 500) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Swap within classes until the selected transitions sum to ``target`` exactly."""

    classes = [(list(s), list(u)) for s, u in classes]
    total = int(sum(int(trans[s].sum()) for s, _ in classes if s))
    swaps = 0
    while total != target:
        if swaps >= max_swaps:
            raise Infeasible(f"repair exceeded {max_swaps} swaps (gap {target - total})")
        sw = _best_swap(classes, trans, target - total)
        if sw is None:
            raise Infeasible(f"no swap reduces the gap of {target - total} transitions")
        k, pos_out, pos_in, change = sw
        sel, unsel = classes[k]
        e_out, e_in = sel.pop(pos_out), unsel.pop(pos_in)
        sel.append(e_in)
        unsel.append(e_out)
        total += change
        swaps += 1
    return classes, swaps


def _prefix_closest(order: np.ndarray, trans: np.ndarray, target: int) -> int:
    """Prefix length whose transition total is closest to ``target`` (ties: shorter)."""

    c = np.concatenate([[0], np.cumsum(trans[order])])
    return int(np.argmin(np.abs(c - target)))


# ---------------------------------------------------------------- designs

def design_O(m: Manifest, perm_S, perm_F, seed: int, arms: dict) -> list[Run]:
    runs = []
    for arm, c in arms.items():
        n_s, n_f = int(c["n_success"]), int(c["n_failure"])
        if n_s > len(perm_S) or n_f > len(perm_F):
            raise Infeasible(f"O {arm}: need {n_s} S / {n_f} F, have {len(perm_S)} / {len(perm_F)}")
        rows = np.concatenate([perm_S[:n_s], perm_F[:n_f]])
        runs.append(_make_run(m, "O", arm, n_f / (n_s + n_f), seed, rows, None))
    return runs


def design_A_arm(m: Manifest, perm_S, perm_F, seed: int, phi: float, T: int, multiple: int,
                 max_candidates: int, design: str = "A") -> Run:
    trans = m.transitions
    cS = np.concatenate([[0], np.cumsum(trans[perm_S])])
    cF = np.concatenate([[0], np.cumsum(trans[perm_F])])
    cands = []
    for N in range(multiple, len(perm_S) + len(perm_F) + 1, multiple):
        n_f_real = phi * N
        if abs(n_f_real - round(n_f_real)) > 1e-9:
            continue
        n_f = int(round(n_f_real))
        n_s = N - n_f
        if n_s > len(perm_S) or n_f > len(perm_F):
            break
        total = int(cS[n_s] + cF[n_f])
        cands.append((abs(T - total), N, n_s, n_f, total))
    if not cands:
        raise Infeasible(f"A phi={phi}: no episode total is a multiple of {multiple} within the pool")
    cands.sort()
    errors = []
    for gap, N, n_s, n_f, total in cands[:max_candidates]:
        classes = [(list(perm_S[:n_s]), list(perm_S[n_s:])), (list(perm_F[:n_f]), list(perm_F[n_f:]))]
        try:
            classes, swaps = repair_to_target(classes, trans, T)
        except Infeasible as e:
            errors.append(f"N={N}: {e}")
            continue
        rows = classes[0][0] + classes[1][0]
        return _make_run(m, design, f"phi{phi:.2f}", phi, seed, rows, T, swaps,
                         {"episode_total_candidates_tried": len(errors) + 1, "prefix_gap_before_repair": int(gap)})
    raise Infeasible(f"A phi={phi}: all candidates failed: {errors}")


def design_B_shape(avail_S: np.ndarray, avail_F: np.ndarray, phis: list[float], unit: int, T: int) -> dict:
    """Common per-length unit counts k_l for Design B, or the infeasibility report.

    Returns a dict with ``feasible``, ``k`` (list over LENGTHS), ``caps``, ``max_feasible_transitions``,
    and the real-valued target ``k_star`` used by the least-squares DP."""

    need_S = max(int(round((1 - p) * unit)) for p in phis)
    need_F = max(int(round(p * unit)) for p in phis)
    for p in phis:  # the unit must split exactly
        if abs(p * unit - round(p * unit)) > 1e-9:
            raise ValueError(f"unit {unit} does not split phi={p} into integers")
    caps = np.minimum(avail_S // need_S, avail_F // need_F).astype(int)
    w = np.array([l - 1 for l in LENGTHS], int)  # transitions per episode at each length
    max_T = int((caps * w).sum() * unit)
    out = {"unit": unit, "need_S_per_unit": need_S, "need_F_per_unit": need_F, "caps": caps.tolist(),
           "max_feasible_transitions": max_T, "requested_transitions": T}
    if T % unit:
        raise ValueError(f"T={T} is not a multiple of unit {unit}")
    R = T // unit
    if max_T < T:
        out.update(feasible=False, k=None, k_star=None)
        return out
    # water-filling target proportional to pooled availability under caps
    a = (avail_S + avail_F).astype(float)
    lo, hi = 0.0, 1.0
    while (np.minimum(caps, hi * a) * w).sum() < R:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if (np.minimum(caps, mid * a) * w).sum() < R:
            lo = mid
        else:
            hi = mid
    k_star = np.minimum(caps, hi * a)
    # exact integer least squares: min sum (k_l - k*_l)^2  s.t.  sum k_l w_l = R, 0 <= k_l <= cap_l
    INF = np.inf
    cost = np.full(R + 1, INF)
    cost[0] = 0.0
    choice = np.zeros((len(LENGTHS), R + 1), int)
    for li, (cap, wl, ks) in enumerate(zip(caps, w, k_star)):
        new = np.full(R + 1, INF)
        for k in range(int(cap) + 1):
            shift = k * int(wl)
            if shift > R:
                break
            cand = cost[: R + 1 - shift] + (k - ks) ** 2
            better = cand < new[shift:]
            new[shift:][better] = cand[better]
            choice[li, shift:][better] = k
        cost = new
    if not np.isfinite(cost[R]):
        out.update(feasible=False, k=None, k_star=k_star.tolist(), reason="no integer combination hits the budget exactly")
        return out
    k = np.zeros(len(LENGTHS), int)
    s = R
    for li in range(len(LENGTHS) - 1, -1, -1):
        k[li] = choice[li, s]
        s -= int(k[li] * w[li])
    assert s == 0 and int((k * w).sum()) == R and np.all(k <= caps)
    out.update(feasible=True, k=k.tolist(), k_star=k_star.tolist(), episodes_per_arm=int(k.sum() * unit),
               lengths_used=[l for l, kk in zip(LENGTHS, k) if kk > 0], objective=float(((k - k_star) ** 2).sum()))
    return out


def design_B_arm(m: Manifest, perm_S, perm_F, seed: int, phi: float, k: list[int], unit: int, T: int) -> Run:
    n_f_unit = int(round(phi * unit))
    n_s_unit = unit - n_f_unit
    rows: list[int] = []
    for l, kk in zip(LENGTHS, k):
        if kk == 0:
            continue
        s_l = perm_S[m.length[perm_S] == l][: n_s_unit * kk]
        f_l = perm_F[m.length[perm_F] == l][: n_f_unit * kk]
        if len(s_l) < n_s_unit * kk or len(f_l) < n_f_unit * kk:
            raise Infeasible(f"B phi={phi} length {l}: need {n_s_unit * kk} S / {n_f_unit * kk} F")
        rows += s_l.tolist() + f_l.tolist()
    return _make_run(m, "B", f"phi{phi:.2f}", phi, seed, rows, T, 0, {"unit_counts_k": list(map(int, k))})


def design_V(m: Manifest, perm_S, perm_F, seed: int, base_successes: int, total: int) -> list[Run]:
    """Shared base of successes; add further successes or failures until the arm holds exactly ``total`` transitions."""

    trans = m.transitions
    if base_successes > len(perm_S):
        raise Infeasible(f"V: base needs {base_successes} successes, have {len(perm_S)}")
    base = perm_S[:base_successes]
    base_transitions = int(trans[base].sum())
    added = total - base_transitions
    if added <= 0:
        raise Infeasible(f"V: base already holds {base_transitions} transitions >= total {total}")
    rest_S = perm_S[base_successes:]
    runs = []
    for arm, pool in (("V+S", rest_S), ("V+F", perm_F)):
        n0 = _prefix_closest(pool, trans, added)
        classes, swaps = repair_to_target([(list(pool[:n0]), list(pool[n0:]))], trans, added)
        add_rows = classes[0][0]
        run = _make_run(m, "V", arm, None, seed, np.concatenate([base, add_rows]), total, swaps,
                        {"base_ids_sha256": sha256_json(sorted(m.ids[base].tolist())), "base_episodes": int(len(base)),
                         "base_transitions": base_transitions, "added_episodes": int(len(add_rows)),
                         "added_transitions": int(trans[np.asarray(add_rows, int)].sum())})
        runs.append(run)
    return runs


# ---------------------------------------------------------------- whole selection

def fallback_budget(max_T: int, rule: dict) -> int:
    b = int(np.floor(rule["fraction_of_max"] * max_T / rule["round_to"]) * rule["round_to"])
    return b


def build_selection(m: Manifest, manifest_sha256: str, config: dict) -> dict:
    S, F = train_pools(m)
    avail_S = np.array([int((m.length[S] == l).sum()) for l in LENGTHS])
    avail_F = np.array([int((m.length[F] == l).sum()) for l in LENGTHS])
    dA, dB, dV, dO = config["designs"]["A"], config["designs"]["B"], config["designs"]["V"], config["designs"]["O"]

    shape = design_B_shape(avail_S, avail_F, dB["phis"], dB["unit"], dB["transitions"])
    budget_B = dB["transitions"]
    comparators_at_B = False
    if not shape["feasible"]:
        budget_B = fallback_budget(shape["max_feasible_transitions"], dB["fallback"])
        shape = design_B_shape(avail_S, avail_F, dB["phis"], dB["unit"], budget_B) if budget_B >= dB["unit"] else shape
        comparators_at_B = shape["feasible"]
    runs: list[Run] = []
    problems: list[str] = []
    for seed in config["seeds"]:
        perm_S, perm_F = seeded_permutations(S, F, seed)
        try:
            runs += design_O(m, perm_S, perm_F, seed, dO["arms"])
        except Infeasible as e:
            problems.append(str(e))
        for phi in dA["phis"]:
            try:
                runs.append(design_A_arm(m, perm_S, perm_F, seed, phi, dA["transitions"], dA["episode_multiple"], dA["max_candidate_totals"]))
            except Infeasible as e:
                problems.append(f"seed {seed}: {e}")
        if shape["feasible"]:
            for phi in dB["phis"]:
                try:
                    runs.append(design_B_arm(m, perm_S, perm_F, seed, phi, shape["k"], dB["unit"], budget_B))
                except Infeasible as e:
                    problems.append(f"seed {seed}: {e}")
            if comparators_at_B:
                for phi in dB["phis"]:
                    try:
                        runs.append(design_A_arm(m, perm_S, perm_F, seed, phi, budget_B, dA["episode_multiple"], dA["max_candidate_totals"], design="AB"))
                    except Infeasible as e:
                        problems.append(f"seed {seed}: {e}")
        else:
            problems.append(f"seed {seed}: Design B infeasible at any budget >= {dB['unit']} (max feasible {shape['max_feasible_transitions']})")
        try:
            runs += design_V(m, perm_S, perm_F, seed, dV["base_successes"], dV["total_transitions"])
        except Infeasible as e:
            problems.append(f"seed {seed}: {e}")

    selection = {
        "config_name": config["name"], "config_version": config["version"], "config_sha256": sha256_json(config),
        "manifest_sha256": manifest_sha256, "manifest_kind": m.header.get("kind"),
        "pool": {"train_successes": int(len(S)), "train_failures": int(len(F)), "lengths": LENGTHS,
                 "avail_success_by_length": avail_S.tolist(), "avail_failure_by_length": avail_F.tolist(),
                 "test_episodes": int(m.test_mask().sum())},
        "design_B": {"budget_transitions": budget_B, "reduced_from": dB["transitions"] if budget_B != dB["transitions"] else None,
                     "A_comparators_at_budget": comparators_at_B, **shape},
        "runs": [asdict(r) for r in runs],
        "n_runs": len(runs),
        "problems": problems,
    }
    selection["certificate"] = certify(m, selection, config)
    selection["selection_sha256_without_certificate"] = sha256_json({k: v for k, v in selection.items() if k != "certificate"})
    return selection


# ---------------------------------------------------------------- certificate (independent re-derivation)

def certify(m: Manifest, selection: dict, config: dict) -> dict:
    """Re-check every registered constraint from the manifest; never trust the sampler's fields."""

    idx = m.index_of()
    train = m.train_mask()
    checks: list[dict] = []
    by_seed: dict[int, dict[str, dict]] = {}

    def ok(run_id: str, name: str, passed: bool, detail: str = "") -> None:
        checks.append({"run_id": run_id, "check": name, "pass": bool(passed), "detail": detail})

    for r in selection["runs"]:
        rows = np.array([idx[i] for i in r["ids"]], int)
        ok(r["run_id"], "ids_distinct", len(set(r["ids"])) == len(r["ids"]))
        ok(r["run_id"], "ids_in_eligible_train_split", bool(train[rows].all()))
        ok(r["run_id"], "ids_hash_matches", sha256_json(sorted(r["ids"])) == r["ids_sha256"])
        n_s = int(m.success[rows].sum())
        n_f = len(rows) - n_s
        trans = int(m.transitions[rows].sum())
        ok(r["run_id"], "counts_match_record", n_s == r["n_success"] and n_f == r["n_failure"] and trans == r["transitions"])
        if r["design"] in ("A", "B", "AB"):
            ok(r["run_id"], "transitions_exact", trans == r["target_transitions"], f"{trans} vs {r['target_transitions']}")
            ok(r["run_id"], "phi_exact", abs(n_f - r["phi"] * len(rows)) < 1e-9, f"{n_f}/{len(rows)}")
        if r["design"] == "B":
            hs = np.array([int(((m.length[rows] == l) & (m.success[rows] == 1)).sum()) for l in LENGTHS])
            hf = np.array([int(((m.length[rows] == l) & (m.success[rows] == 0)).sum()) for l in LENGTHS])
            ok(r["run_id"], "B_conditional_length_distributions_equal", bool(np.all(hs * n_f == hf * n_s)),
               "hist_S * n_F == hist_F * n_S at every length")
            by_seed.setdefault(r["seed"], {})[r["arm"]] = {"total_hist": (hs + hf).tolist()}
        if r["design"] == "V":
            ok(r["run_id"], "V_total_transitions_exact", trans == r["target_transitions"], f"{trans} vs {r['target_transitions']}")
        if r["design"] == "O":
            arms = config["designs"]["O"]["arms"][r["arm"]]
            ok(r["run_id"], "O_counts", n_s == arms["n_success"] and n_f == arms["n_failure"])
    for seed, arms in by_seed.items():
        hists = [a["total_hist"] for a in arms.values()]
        ok(f"B_seed{seed}", "B_total_histograms_identical_across_arms", all(h == hists[0] for h in hists))
    # V: shared base across the two V arms of a seed
    v_runs = [r for r in selection["runs"] if r["design"] == "V"]
    for seed in sorted({r["seed"] for r in v_runs}):
        pair = [r for r in v_runs if r["seed"] == seed]
        if len(pair) == 2:
            ok(f"V_seed{seed}", "V_base_shared", pair[0]["notes"]["base_ids_sha256"] == pair[1]["notes"]["base_ids_sha256"])
            a, b = set(pair[0]["ids"]), set(pair[1]["ids"])
            ok(f"V_seed{seed}", "V_additions_disjoint", len(a & b) == pair[0]["notes"]["base_episodes"])
    expected = len(config["seeds"]) * (len(config["designs"]["O"]["arms"]) + len(config["designs"]["A"]["phis"]) + len(config["designs"]["B"]["phis"]) + 2)
    if selection["design_B"].get("A_comparators_at_budget"):
        expected += len(config["seeds"]) * len(config["designs"]["B"]["phis"])
    ok("ALL", "run_count", selection["n_runs"] == expected and not selection["problems"], f"{selection['n_runs']} runs, expected {expected}; problems: {selection['problems']}")
    seeds_seen = sorted({r["seed"] for r in selection["runs"]})
    ok("ALL", "seeds_registered", seeds_seen == sorted(config["seeds"]), str(seeds_seen))
    ok("ALL", "historical_seeds_excluded", not (set(seeds_seen) & set(config["historical_seeds_excluded"])))
    failed = [c for c in checks if not c["pass"]]
    return {"feasible": not failed, "n_checks": len(checks), "n_failed": len(failed), "failed": failed,
            "design_B_budget": selection["design_B"]["budget_transitions"],
            "design_B_reduced": selection["design_B"]["reduced_from"] is not None,
            "checks": checks}


# ---------------------------------------------------------------- reporting

def selection_table(selection: dict) -> str:
    rows = ["| Design | Arm | Seeds | Episodes (min–max) | Successes | Failures | Transitions | Swaps (max) |", "|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in selection["runs"]:
        groups.setdefault((r["design"], r["arm"]), []).append(r)
    for (d, a), rs in groups.items():
        ne = [r["n_episodes"] for r in rs]
        ns = [r["n_success"] for r in rs]
        nf = [r["n_failure"] for r in rs]
        tr = sorted({r["transitions"] for r in rs})
        trs = f"{tr[0]:,}" if len(tr) == 1 else f"{tr[0]:,}–{tr[-1]:,}"
        rows.append(f"| {d} | {a} | {len(rs)} | {min(ne):,}–{max(ne):,} | {min(ns):,}–{max(ns):,} | {min(nf):,}–{max(nf):,} | {trs} | {max(r['swaps'] for r in rs)} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build and certify the registered selection for all designs and seeds.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True, help="selection JSON (ids inline)")
    ap.add_argument("--allow-synthetic", action="store_true", help="permit a SYNTHETIC manifest (dry runs only)")
    args = ap.parse_args(argv)
    m, digest = load_manifest(Path(args.manifest))
    if m.header.get("kind") == "SYNTHETIC" and not args.allow_synthetic:
        raise SystemExit("refusing a SYNTHETIC manifest without --allow-synthetic")
    config = load_config(args.config)
    sel = build_selection(m, digest, config)
    write_json(Path(args.out), sel)
    cert = sel["certificate"]
    summary = {k: v for k, v in cert.items() if k != "checks"}
    summary["design_B"] = {k: v for k, v in sel["design_B"].items() if k not in ("k_star",)}
    write_json(Path(args.out).with_name(Path(args.out).stem + ".certificate.json"), {**summary, "checks": cert["checks"]})
    Path(args.out).with_suffix(".md").write_text(selection_table(sel) + "\n")
    print(json.dumps(summary, indent=1))
    print(selection_table(sel))


if __name__ == "__main__":
    main()
