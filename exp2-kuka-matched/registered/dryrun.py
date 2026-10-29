"""Synthetic dry run and operating characteristics of the frozen Experiment 2 analyzer.

Per-episode judge scores are simulated for every run of the certified selection under declared true
contrasts, then passed through the same validation and analyzer (``analyze.analyze``) that will score
the real study. Nothing in the analyzer is bypassed, so the registration documents exactly what the
frozen code decides under each truth.

Generative model (rate scale). Episode i in run (d, s) is a predicted success with probability
    pi_i(d, s) = base_{Y_i} + offset_d + u_s + v_{d,s},
base_1 = 0.62, base_0 = 0.45 (weighted base rate 0.51, as the owner's zero-failure arm), offset_d
encodes the declared contrasts, u_s ~ N(0, 0.01^2) is a shared seed effect, and v_{d,s} is arm-by-seed
noise whose SD is calibrated so that the paired seed contrast has the requested SD (1.88 historical,
3 sensitivity, 5 stress) after the binomial component implied by the frozen weights. The judge
probability is drawn so that 1[p > 0.5] equals the simulated outcome and the mean probability carries
half the contrast, which keeps the mean-probability sensitivity distinct from the primary response.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import PACKAGE_ROOT, load_config, sha256_json, write_json
from .analyze import analyze
from .manifest import load_manifest

BASE = {1: 0.62, 0: 0.45}
SEED_SD = 0.01

# true contrasts in percentage points: Δ_O, Δ_A, Δ_B, Δ_V; optional missing run and SD override
SCENARIOS: dict[str, dict] = {
    "global_null": dict(O=0, A=0, B=0, V=0),
    "A_at_margin_3": dict(O=0, A=3, B=0, V=0),
    "A_at_minus_margin_-3": dict(O=0, A=-3, B=0, V=0),
    "A_5_planning": dict(O=10, A=5, B=5, V=5),
    "historical_10": dict(O=10, A=10, B=10, V=10),
    "transition_confound": dict(O=10, A=0, B=0, V=0),
    "length_shortcut": dict(O=10, A=10, B=0, V=10),
    "persists_after_balancing": dict(O=8, A=8, B=8, V=8),
    "reverse_-5": dict(O=0, A=-5, B=0, V=0),
    "missing_run_historical_10": dict(O=10, A=10, B=10, V=10, missing=["A_phi0.65_s104"]),
    "A_5_stress_sd5": dict(O=10, A=5, B=5, V=5, sd=5.0),
}
INTERPRETATION_KEYS = {
    "transition_budget_sensitive": "Effect is sensitive to the transition-budget intervention",
    "survives_matching": "Effect survives transition matching",
    "length_balancing_attenuates": "Training-length balancing attenuates",
    "persists_after_balancing": "Effect persists after removing outcome",
    "insufficient_evidence": "Insufficient evidence",
    "V_failures_lower": "Failure additions lower estimates",
}


def arm_offset(design: str, arm: str, truth: dict, selection: dict) -> float:
    """Rate offset (fraction) for an arm so that the .10 - .65 contrast equals the declared points."""

    if design == "V":
        return 0.0 if arm == "V+S" else -truth["V"] / 100
    key = "A" if design == "AB" else design
    phi = float(arm.replace("phi", ""))
    slope = truth[key] / 100 / 0.55
    return -slope * (phi - 0.10)


def calibrated_arm_seed_sd(target_points: float, support: dict) -> float:
    a = np.array(list(support["weights"].values()))
    binom_var_points = 2 * (100 ** 2) * 0.25 * float((a ** 2).sum())  # two arms, p(1-p) <= 0.25
    rest = max(target_points ** 2 - binom_var_points, 0.0)
    return float(np.sqrt(rest / 2) / 100)


def synthetic_runs(rng: np.random.Generator, selection: dict, support: dict, config: dict, truth: dict,
                   eval_ids: list[str], y_eval: np.ndarray, sd_points: float) -> list[dict]:
    sel_hash = selection["selection_sha256_without_certificate"]
    cfg_hash = sha256_json(config)
    sd_v = calibrated_arm_seed_sd(sd_points, support)
    u = {s: rng.normal(0, SEED_SD) for s in config["seeds"]}
    base = np.where(y_eval == 1, BASE[1], BASE[0])
    missing = set(truth.get("missing", []))
    runs = []
    for r in selection["runs"]:
        rec = {"run_id": r["run_id"], "design": r["design"], "arm": r["arm"], "seed": r["seed"], "selection_sha256": sel_hash,
               "config_sha256": cfg_hash, "ids_sha256": r["ids_sha256"], "status": "completed"}
        if r["run_id"] in missing:
            rec["status"] = "infrastructure_failure"
            runs.append(rec)
            continue
        pi = np.clip(base + arm_offset(r["design"], r["arm"], truth, selection) + u[r["seed"]] + rng.normal(0, sd_v), 0.01, 0.99)
        h = rng.random(len(pi)) < pi
        unif = rng.random(len(pi))
        p = np.where(h, 0.5 + 0.5 * unif, 0.5 * unif)
        rec.update(eval_ids=eval_ids, p=p.tolist())
        runs.append(rec)
    return runs


_CACHE: dict = {}


def _frozen(paths: dict) -> tuple:
    key = tuple(sorted(paths.items()))
    if key not in _CACHE:
        config = load_config(paths["config"])
        selection = json.loads(Path(paths["selection"]).read_text())
        support = json.loads(Path(paths["support"]).read_text())
        m, _ = load_manifest(Path(paths["manifest"]))
        te = m.test_mask()
        _CACHE[key] = (config, selection, support, m.ids[te].tolist(), m.success[te], {str(i): int(y) for i, y in zip(m.ids, m.success)})
    return _CACHE[key]


def _stable_int(name: str) -> int:
    import hashlib
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _one(args) -> dict:
    name, truth, sd_points, dataset_seed, paths = args
    config, selection, support, eval_ids, y_eval, truth_map = _frozen(paths)
    rng = np.random.default_rng([dataset_seed, _stable_int(name)])
    runs = synthetic_runs(rng, selection, support, config, truth, eval_ids, y_eval, truth.get("sd", sd_points))
    res = analyze(runs, selection, support, config, truth=truth_map, with_auroc=False)
    return {"scenario": name, "dataset": dataset_seed, "decisions": res["decisions"], "interpretation": res["interpretation"],
            "A_mean": res["quantities"]["A"]["t"]["mean"] if res["quantities"]["A"]["t"] else None,
            "A_sd": res["quantities"]["A"]["t"]["sd"] if res["quantities"]["A"]["t"] else None,
            "A_bounds": res["quantities"]["A"].get("worst_case_bounds_on_mean")}


def analytic_power(delta: float, sd: float, n: int, margin: float, alpha: float = 0.05) -> float:
    from scipy import stats
    lam = (delta - margin) * np.sqrt(n) / sd
    return float(stats.nct.sf(stats.t.ppf(1 - alpha, n - 1), n - 1, lam))


def summarize(results: list[dict], n_datasets: int, sd_points: float, config: dict) -> dict:
    out: dict[str, dict] = {}
    for name, truth in SCENARIOS.items():
        rows = [r for r in results if r["scenario"] == name]
        q_probs = {}
        for q in ["A", "O", "B", "V", "O_minus_A", "A_minus_B"]:
            labels = [r["decisions"][q] for r in rows]
            q_probs[q] = {lab: labels.count(lab) / len(rows) for lab in ("supported", "equivalent", "inconclusive")}
        interp = {k: np.mean([any(v in s for s in r["interpretation"]) for r in rows]) for k, v in INTERPRETATION_KEYS.items()}
        a_sd = [r["A_sd"] for r in rows if r["A_sd"] is not None]
        out[name] = {"truth": truth, "n_datasets": len(rows), "decision_probabilities": q_probs, "interpretation_frequency": interp,
                     "realized_paired_sd_A": {"mean": float(np.mean(a_sd)) if a_sd else None}, "mock_first_dataset": rows[0]["decisions"] if rows else None,
                     "A_worst_case_bounds_first_dataset": rows[0]["A_bounds"] if rows else None}
    # validation against the registration's claims
    n = n_datasets
    tol = lambda p: 2.5 * np.sqrt(p * (1 - p) / n)  # noqa: E731
    margin = config["decision"]["margin_points"]
    seeds = len(config["seeds"])
    pw = analytic_power(5.0, sd_points, seeds, margin)
    checks = [
        {"check": "false support of A at the margin (Δ_A = 3) ≤ 5% (+MC tolerance)", "value": out["A_at_margin_3"]["decision_probabilities"]["A"]["supported"], "bound": 0.05 + tol(0.05),
         "pass": out["A_at_margin_3"]["decision_probabilities"]["A"]["supported"] <= 0.05 + tol(0.05)},
        {"check": "false equivalence of A at the lower margin (Δ_A = −3) ≤ 5% (+MC tolerance)", "value": out["A_at_minus_margin_-3"]["decision_probabilities"]["A"]["equivalent"], "bound": 0.05 + tol(0.05),
         "pass": out["A_at_minus_margin_-3"]["decision_probabilities"]["A"]["equivalent"] <= 0.05 + tol(0.05)},
        {"check": f"power for A at Δ_A = 5, SD {sd_points} matches the analytic value {pw:.3f} (±MC tolerance)", "value": out["A_5_planning"]["decision_probabilities"]["A"]["supported"], "bound": pw,
         "pass": abs(out["A_5_planning"]["decision_probabilities"]["A"]["supported"] - pw) <= tol(max(min(pw, 0.95), 0.05)) + 0.02},
        {"check": "missing run makes A, O−A and A−B inconclusive in every dataset", "value": min(out["missing_run_historical_10"]["decision_probabilities"][q]["inconclusive"] for q in ("A", "O_minus_A", "A_minus_B")), "bound": 1.0,
         "pass": all(out["missing_run_historical_10"]["decision_probabilities"][q]["inconclusive"] == 1.0 for q in ("A", "O_minus_A", "A_minus_B"))},
        {"check": "global null: familywise false 'supported' among secondaries ≤ 5% (+MC tolerance)", "value": None, "bound": 0.05 + tol(0.05), "pass": None},
        {"check": "historical 10-point effect: all four drops supported with probability ≥ 0.99", "value": min(out["historical_10"]["decision_probabilities"][q]["supported"] for q in ("A", "O", "B", "V")), "bound": 0.99,
         "pass": all(out["historical_10"]["decision_probabilities"][q]["supported"] >= 0.99 for q in ("A", "O", "B", "V"))},
    ]
    rows = [r for r in results if r["scenario"] == "global_null"]
    fw = np.mean([any(r["decisions"][q] == "supported" for q in config["decision"]["secondary"]) for r in rows])
    checks[4]["value"] = float(fw)
    checks[4]["pass"] = bool(fw <= 0.05 + tol(0.05))
    return {"sd_points": sd_points, "n_datasets": n_datasets, "analytic_power_A5": pw, "scenarios": out, "validation": checks,
            "all_checks_pass": all(c["pass"] for c in checks)}


def table_markdown(summary: dict) -> str:
    L = [f"**Paired SD {summary['sd_points']} points, {summary['n_datasets']} synthetic datasets per truth.** Analytic power at Δ_A = 5: {summary['analytic_power_A5']:.3f}.", "",
         "| Truth (Δ_O, Δ_A, Δ_B, Δ_V in points) | A supported | A equivalent | A inconclusive | O | B | V | O−A | A−B | Interpretation rows (freq.) |", "|---|---|---|---|---|---|---|---|---|---|"]

    def top(d: dict) -> str:
        k = max(d, key=d.get)
        return f"{k} {d[k]:.2f}"

    for name, s in summary["scenarios"].items():
        t = s["truth"]
        extra = ("; missing " + ", ".join(t["missing"])) if t.get("missing") else ""
        extra += f"; SD {t['sd']}" if t.get("sd") else ""
        dp = s["decision_probabilities"]
        interp = "; ".join(f"{k} {v:.2f}" for k, v in s["interpretation_frequency"].items() if v > 0) or "—"
        L.append(f"| {name}: ({t['O']}, {t['A']}, {t['B']}, {t['V']}){extra} | {dp['A']['supported']:.2f} | {dp['A']['equivalent']:.2f} | {dp['A']['inconclusive']:.2f} | "
                 f"{top(dp['O'])} | {top(dp['B'])} | {top(dp['V'])} | {top(dp['O_minus_A'])} | {top(dp['A_minus_B'])} | {interp} |")
    L += ["", "| Validation | Value | Bound | Pass |", "|---|---|---|---|"]
    for c in summary["validation"]:
        L.append(f"| {c['check']} | {c['value']:.3f} | {c['bound']:.3f} | {'yes' if c['pass'] else 'NO'} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Synthetic dry run of the frozen Experiment 2 analyzer.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", type=int, default=200)
    ap.add_argument("--sd", type=float, action="append", default=None, help="paired SD in points (repeat); default 1.88 and 3.0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="output JSON; .md written alongside")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    paths = {"config": args.config or str(PACKAGE_ROOT / "configs" / "exp2_registered.json"), "selection": args.selection, "support": args.support, "manifest": args.manifest}
    sds = args.sd or [1.88, 3.0]
    summaries = {}
    md = ["# Experiment 2 dry run: operating characteristics of the frozen analyzer", ""]
    for sd in sds:
        jobs = [(name, truth, sd, k, paths) for name, truth in SCENARIOS.items() for k in range(args.datasets)]
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_one, jobs, chunksize=4))
        summaries[str(sd)] = summarize(results, args.datasets, sd, config)
        md.append(table_markdown(summaries[str(sd)]))
    write_json(Path(args.out), {"scenarios": SCENARIOS, "by_sd": summaries, "config_sha256": sha256_json(config)})
    Path(args.out).with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
