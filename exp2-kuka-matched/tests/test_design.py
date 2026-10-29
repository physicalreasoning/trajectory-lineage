from __future__ import annotations

import numpy as np
import pytest

from registered.design import (LENGTHS, Infeasible, build_selection, certify, design_A_arm, design_B_shape, repair_to_target,
                               seeded_permutations, train_pools)
from tests.conftest import make_manifest


def test_certificate_passes_and_counts_runs(selection, config):
    cert = selection["certificate"]
    assert cert["feasible"], cert["failed"]
    expected = 110 + (20 if selection["design_B"]["A_comparators_at_budget"] else 0)
    assert selection["n_runs"] == expected
    assert not selection["problems"]
    assert cert["n_checks"] > 600


def test_design_A_exact_transitions_and_phi(selection, manifest):
    idx = manifest.index_of()
    for r in selection["runs"]:
        if r["design"] != "A":
            continue
        rows = np.array([idx[i] for i in r["ids"]])
        assert int(manifest.transitions[rows].sum()) == 31600
        n_f = int((manifest.success[rows] == 0).sum())
        assert n_f == pytest.approx(r["phi"] * len(rows), abs=1e-9)
        assert len(rows) % 20 == 0
        assert len(set(r["ids"])) == len(r["ids"])
        assert all(manifest.split[rows] == "train")


def test_design_B_equal_conditional_distributions_and_shared_histogram(selection, manifest):
    idx = manifest.index_of()
    by_seed: dict[int, list[list[int]]] = {}
    for r in selection["runs"]:
        if r["design"] != "B":
            continue
        rows = np.array([idx[i] for i in r["ids"]])
        L, Y = manifest.length[rows], manifest.success[rows]
        hs = np.array([((L == l) & (Y == 1)).sum() for l in LENGTHS])
        hf = np.array([((L == l) & (Y == 0)).sum() for l in LENGTHS])
        n_s, n_f = hs.sum(), hf.sum()
        assert np.all(hs * n_f == hf * n_s), "normalized conditional length distributions must be identical"
        assert int(manifest.transitions[rows].sum()) == selection["design_B"]["budget_transitions"]
        by_seed.setdefault(r["seed"], []).append((hs + hf).tolist())
    for seed, hists in by_seed.items():
        assert len(hists) == 2 and hists[0] == hists[1]


def test_design_V_shared_base_and_total_matched_to_A(selection, manifest):
    idx = manifest.index_of()
    for seed in {r["seed"] for r in selection["runs"]}:
        pair = {r["arm"]: r for r in selection["runs"] if r["design"] == "V" and r["seed"] == seed}
        assert pair["V+S"]["notes"]["base_ids_sha256"] == pair["V+F"]["notes"]["base_ids_sha256"]
        for r in pair.values():
            rows = np.array([idx[i] for i in r["ids"]])
            assert int(manifest.transitions[rows].sum()) == 31600  # transition-matched to Design A
        assert pair["V+F"]["n_failure"] > 0 and pair["V+S"]["n_failure"] == 0


def test_pairing_same_seed_shares_prefix(manifest):
    S, F = train_pools(manifest)
    a = seeded_permutations(S, F, 100)
    b = seeded_permutations(S, F, 100)
    c = seeded_permutations(S, F, 101)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_selection_is_deterministic(manifest, config):
    s1 = build_selection(manifest, "m", config)
    s2 = build_selection(manifest, "m", config)
    assert s1["selection_sha256_without_certificate"] == s2["selection_sha256_without_certificate"]
    assert [r["ids_sha256"] for r in s1["runs"]] == [r["ids_sha256"] for r in s2["runs"]]


def test_repair_terminates_and_hits_target():
    trans = np.array([2, 5, 9, 12, 20, 23, 7, 7, 3, 15])
    sel, unsel = [0, 1, 2], [3, 4, 5, 6, 7, 8, 9]
    classes, swaps = repair_to_target([(sel, unsel)], trans, 40)
    assert sum(trans[classes[0][0]]) == 40 and swaps >= 1
    with pytest.raises(Infeasible):
        repair_to_target([([0], [1])], trans, 100)


def test_design_B_shape_dp_is_exact_and_capped():
    avail_S = np.array([0, 0, 300, 400, 300, 200, 500, 300, 300, 150, 250, 150, 150, 70, 80, 90, 70, 200, 10, 40, 10, 5])
    avail_F = np.array([40, 10, 90, 10, 40, 230, 200, 270, 420, 380, 270, 380, 270, 260, 240, 140, 310, 100, 2800, 330, 130, 10])
    probe = design_B_shape(avail_S, avail_F, [0.10, 0.65], 20, 20)
    T = (probe["max_feasible_transitions"] // 2000) * 2000  # a feasible budget for this pool
    shape = design_B_shape(avail_S, avail_F, [0.10, 0.65], 20, T)
    assert shape["feasible"]
    k = np.array(shape["k"])
    assert int((k * (np.array(LENGTHS) - 1)).sum()) * 20 == T
    assert np.all(k <= np.array(shape["caps"]))
    assert np.all(18 * k <= avail_S) and np.all(13 * k <= avail_F)
    tiny = design_B_shape(avail_S // 10, avail_F // 10, [0.10, 0.65], 20, 31600)
    assert not tiny["feasible"] and tiny["max_feasible_transitions"] < 31600


def test_fallback_budget_adds_A_comparators(config):
    m = make_manifest(np.random.default_rng(3), train_f=3000)  # the owner's memory cap starves Design B
    sel = build_selection(m, "capped", config)
    assert sel["design_B"]["reduced_from"] == 31600
    assert sel["design_B"]["budget_transitions"] % 1000 == 0 and sel["design_B"]["budget_transitions"] < 31600
    assert sel["design_B"]["A_comparators_at_budget"]
    assert sel["n_runs"] == 130 and sel["certificate"]["feasible"], sel["certificate"]["failed"]
    ab = [r for r in sel["runs"] if r["design"] == "AB"]
    assert {r["transitions"] for r in ab} == {sel["design_B"]["budget_transitions"]}


def test_certificate_detects_tampering(selection, manifest, config):
    import copy
    bad = copy.deepcopy(selection)
    r = next(x for x in bad["runs"] if x["design"] == "A")
    r["ids"][0], r["ids"][1] = r["ids"][1], r["ids"][0]  # same set: still passes
    assert certify(manifest, bad, config)["feasible"]
    r["ids"].append(r["ids"][0])  # duplicate id
    cert = certify(manifest, bad, config)
    assert not cert["feasible"] and any(c["check"] == "ids_distinct" for c in cert["failed"])


def test_design_A_phi_zero_has_no_failures(manifest):
    S, F = train_pools(manifest)
    pS, pF = seeded_permutations(S, F, 100)
    r = design_A_arm(manifest, pS, pF, 100, 0.0, 31600, 20, 8)
    assert r.n_failure == 0 and r.transitions == 31600
