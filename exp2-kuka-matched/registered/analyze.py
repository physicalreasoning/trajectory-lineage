"""The registered decision contract for Experiment 2 (Protocol §2, "Analysis plan").

Inputs are frozen artefacts (selection, support, config, optional freeze manifest) and one JSON per
training run written by ``run_registered``. The analyzer:

1. validates every run against the frozen selection (run id, hashes, evaluation ids, probabilities);
2. computes weighted thresholded success R_ds = sum_i a_i 1[p_ids > 0.5] per run, plus the
   mean-probability and unweighted descriptives, Brier score and error against truth;
3. forms paired seed contrasts D_ds = 100 (R_d,.10,s - R_d,.65,s) for designs O, A, B, the V contrast
   100 (R_V+S,s - R_V+F,s), and the differences O - A and A - B (A at B's budget when comparators exist);
4. applies paired t inference: support if the one-sided 95% lower bound exceeds the margin (3 points),
   absence if the 90% CI lies inside (-3, 3) (two one-sided tests at 5%); the ten secondary p-values
   (support and equivalence for O, B, V, O - A, A - B) are Holm-adjusted at familywise 0.05;
5. makes every decision that depends on a missing or failed run inconclusive and reports worst-case
   bounds with rates in [0, 1] instead of analysing only the completed runs;
6. maps the decision pattern to the registered interpretation table.

Nothing here reads a model or a feature; it is pure arithmetic on frozen inputs, so it can be dry-run
on synthetic scores (``dryrun``) and the operating characteristics attached to the registration.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

from . import load_config, sha256_json, write_json

REQUIRED_RUN_FIELDS = ("run_id", "design", "arm", "seed", "status", "selection_sha256", "config_sha256")
COMPLETED = "completed"


class AnalysisError(Exception):
    """Frozen inputs or run files are inconsistent; no decision is made."""


# ---------------------------------------------------------------- validation

def validate_runs(runs: list[dict], selection: dict, support: dict, config: dict) -> list[str]:
    """Return a list of hard problems (empty if the run set is admissible)."""

    problems: list[str] = []
    expected = {r["run_id"]: r for r in selection["runs"]}
    seen: set[str] = set()
    sel_hash = selection["selection_sha256_without_certificate"]
    cfg_hash = sha256_json(config)
    weight_ids = set(support["weights"])
    for r in runs:
        for f in REQUIRED_RUN_FIELDS:
            if f not in r:
                problems.append(f"{r.get('run_id', '?')}: missing field {f}")
        rid = r.get("run_id")
        if rid not in expected:
            problems.append(f"{rid}: not in the frozen selection")
            continue
        if rid in seen:
            problems.append(f"{rid}: duplicate run file")
        seen.add(rid)
        if r.get("selection_sha256") != sel_hash:
            problems.append(f"{rid}: selection hash {str(r.get('selection_sha256'))[:12]} != frozen {sel_hash[:12]} (stale artefact)")
        if r.get("config_sha256") != cfg_hash:
            problems.append(f"{rid}: config hash does not match the frozen config")
        if r.get("ids_sha256") not in (None, expected[rid]["ids_sha256"]):
            problems.append(f"{rid}: training ids hash differs from the frozen selection")
        if r.get("status") == COMPLETED:
            ids, p = r.get("eval_ids"), r.get("p")
            if not ids or p is None or len(ids) != len(p):
                problems.append(f"{rid}: eval_ids and p missing or misaligned")
                continue
            if len(set(ids)) != len(ids):
                problems.append(f"{rid}: duplicate evaluation ids")
            if not weight_ids <= set(ids):
                problems.append(f"{rid}: {len(weight_ids - set(ids))} supported evaluation episodes have no score")
            arr = np.asarray(p, float)
            if not np.all(np.isfinite(arr)) or arr.min() < 0 or arr.max() > 1:
                problems.append(f"{rid}: probabilities outside [0, 1] or non-finite")
        elif r.get("status") not in ("infrastructure_failure", "algorithmic_failure"):
            problems.append(f"{rid}: unknown status {r.get('status')}")
    return problems


# ---------------------------------------------------------------- per-run rates

def run_rates(r: dict, support: dict, truth: dict[str, int], threshold: float, with_auroc: bool = True) -> dict:
    ids = r["eval_ids"]
    p = np.asarray(r["p"], float)
    w = np.array([support["weights"].get(i, 0.0) for i in ids])
    y = np.array([truth[i] for i in ids], float)
    h = (p > threshold).astype(float)
    pi = support["prevalence_pi"]
    R = float((w * h).sum())
    out = {
        "R_thr": R, "R_meanp": float((w * p).sum()),
        "R_thr_unweighted": float(h.mean()), "R_meanp_unweighted": float(p.mean()),
        "brier_w": float((w * (p - y) ** 2).sum()), "signed_error_vs_truth": R - pi, "abs_error_vs_truth": abs(R - pi),
        "p_on_failures_w": float((w * p)[y == 0].sum() / w[y == 0].sum()), "p_on_successes_w": float((w * p)[y == 1].sum() / w[y == 1].sum()),
    }
    out["auroc_unweighted"] = None
    if with_auroc:
        try:
            from sklearn.metrics import roc_auc_score
            out["auroc_unweighted"] = float(roc_auc_score(y, p))
        except Exception:  # sklearn optional
            pass
    return out


# ---------------------------------------------------------------- inference

def paired_t(d: np.ndarray, margin: float, alpha: float, eq_level: float) -> dict:
    d = np.asarray(d, float)
    n = int(len(d))
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else None, "sd": None, "se": None, "lower95_one_sided": None,
                "ci90": None, "p_support": None, "p_equivalence": None}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t_a = stats.t.ppf(1 - alpha, df)
    t_eq = stats.t.ppf(1 - (1 - eq_level) / 2, df)
    if se == 0:
        p_sup = 0.0 if mean > margin else 1.0
        p_eq = 0.0 if abs(mean) < margin else 1.0
    else:
        p_sup = float(stats.t.sf((mean - margin) / se, df))                     # H0: delta <= margin
        p1 = float(stats.t.sf((mean + margin) / se, df))                        # H0: delta <= -margin
        p2 = float(stats.t.cdf((mean - margin) / se, df))                       # H0: delta >= margin
        p_eq = max(p1, p2)
    return {"n": n, "mean": mean, "sd": sd, "se": float(se), "lower95_one_sided": float(mean - t_a * se),
            "ci90": [float(mean - t_eq * se), float(mean + t_eq * se)], "p_support": p_sup, "p_equivalence": p_eq}


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict]:
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    adjusted: dict[str, dict] = {}
    running = 0.0
    for i, (k, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[k] = {"p": p, "p_holm": running, "reject": running < alpha}
    for k, v in pvals.items():
        if v is None:
            adjusted[k] = {"p": None, "p_holm": None, "reject": False}
    return adjusted


# ---------------------------------------------------------------- contracts

def contrast_specs(selection: dict) -> dict[str, list[tuple[str, str, float]]]:
    """Quantity -> list of (design, arm, coefficient) applied to R and multiplied by 100."""

    a_for_b = "AB" if selection["design_B"].get("A_comparators_at_budget") else "A"
    specs = {
        "A": [("A", "phi0.10", 1), ("A", "phi0.65", -1)],
        "O": [("O", "phi0.10", 1), ("O", "phi0.65", -1)],
        "B": [("B", "phi0.10", 1), ("B", "phi0.65", -1)],
        "V": [("V", "V+S", 1), ("V", "V+F", -1)],
        "O_minus_A": [("O", "phi0.10", 1), ("O", "phi0.65", -1), ("A", "phi0.10", -1), ("A", "phi0.65", 1)],
        "A_minus_B": [(a_for_b, "phi0.10", 1), (a_for_b, "phi0.65", -1), ("B", "phi0.10", -1), ("B", "phi0.65", 1)],
    }
    if a_for_b == "AB":
        specs["AB"] = [("AB", "phi0.10", 1), ("AB", "phi0.65", -1)]
    return specs


def seed_contrasts(rates: dict[tuple[str, str, int], dict], spec, seeds, key="R_thr"):
    """Per-seed contrast values (NaN when a run is missing) and the missing (design, arm, seed) list."""

    vals, missing = [], []
    for s in seeds:
        v = 0.0
        ok = True
        for d, a, c in spec:
            r = rates.get((d, a, s))
            if r is None:
                ok = False
                missing.append((d, a, s))
            else:
                v += c * r[key]
        vals.append(100 * v if ok else np.nan)
    return np.array(vals), missing


def worst_case_bounds(rates, spec, seeds, key="R_thr") -> list[float]:
    """Bounds on the seed-mean contrast when missing rates range over [0, 1]."""

    lo = hi = 0.0
    n = len(seeds)
    for s in seeds:
        for d, a, c in spec:
            r = rates.get((d, a, s))
            if r is None:
                lo += min(0.0, 100 * c) / n
                hi += max(0.0, 100 * c) / n
            else:
                lo += 100 * c * r[key] / n
                hi += 100 * c * r[key] / n
    return [float(lo), float(hi)]


def interpret(dec: dict[str, str]) -> list[str]:
    """Registered outcome-to-interpretation table (Protocol §2)."""

    rows = []
    g = dec.get
    if g("O") == "supported" and g("A") == "equivalent" and g("O_minus_A") == "supported":
        rows.append("Effect is sensitive to the transition-budget intervention; changed episode counts also matter.")
    if g("A") == "supported":
        rows.append("Effect survives transition matching and evaluation standardization; training length shortcuts remain possible.")
    if g("A") == "supported" and g("B") == "equivalent" and g("A_minus_B") == "supported":
        rows.append("Training-length balancing attenuates the effect, conditional on matched budgets and reported population changes.")
    if g("B") == "supported":
        rows.append("Effect persists after removing outcome–length association in WM training; this does not isolate dynamics error.")
    if g("A") == "inconclusive" or g("B") == "inconclusive":
        rows.append("Insufficient evidence for the inconclusive design(s); no 'effect erased' claim.")
    if g("V") == "supported":
        rows.append("Failure additions lower estimates relative to success additions; benefit requires error/discrimination evidence.")
    return rows or ["No registered interpretation row applies; report descriptively."]


def analyze(runs: list[dict], selection: dict, support: dict, config: dict, judge: dict | None = None,
            truth: dict[str, int] | None = None, with_auroc: bool = True) -> dict:
    problems = validate_runs(runs, selection, support, config)
    if problems:
        raise AnalysisError("; ".join(problems[:20]) + (f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""))
    dc = config["decision"]
    margin, alpha, eq_level = dc["margin_points"], dc["alpha_support_one_sided"], dc["equivalence_ci_level"]
    seeds = list(config["seeds"])
    if truth is None:
        truth = {}
    expected_ids = {r["run_id"] for r in selection["runs"]}
    present = {r["run_id"]: r for r in runs}
    missing_runs = sorted(expected_ids - set(present))
    failed_runs = {r["run_id"]: r["status"] for r in runs if r["status"] != COMPLETED}

    rates: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        if r["status"] == COMPLETED:
            rates[(r["design"], r["arm"], r["seed"])] = run_rates(r, support, truth, config["evaluation"]["threshold"], with_auroc)

    specs = contrast_specs(selection)
    quantities: dict[str, dict] = {}
    for q, spec in specs.items():
        d, missing = seed_contrasts(rates, spec, seeds)
        d_meanp, _ = seed_contrasts(rates, spec, seeds, key="R_meanp")
        complete = ~np.isnan(d)
        entry = {"spec": spec, "per_seed": [None if np.isnan(x) else float(x) for x in d],
                 "missing_runs": sorted({f"{dd}_{a}_s{s}" for dd, a, s in missing}),
                 "complete": bool(complete.all())}
        if complete.all():
            entry["t"] = paired_t(d, margin, alpha, eq_level)
            entry["t_meanp_sensitivity"] = paired_t(d_meanp, margin, alpha, eq_level)
        else:
            entry["t"] = None
            entry["worst_case_bounds_on_mean"] = worst_case_bounds(rates, spec, seeds)
            entry["descriptive_completed_seeds_only"] = paired_t(d[complete], margin, alpha, eq_level) if complete.sum() >= 2 else None
        quantities[q] = entry

    # primary
    decisions: dict[str, str] = {}
    prim = quantities[dc["primary"]]
    if prim["t"] is None:
        decisions[dc["primary"]] = "inconclusive"
    else:
        t = prim["t"]
        if t["lower95_one_sided"] > margin:
            decisions[dc["primary"]] = "supported"
        elif t["ci90"][0] > -margin and t["ci90"][1] < margin:
            decisions[dc["primary"]] = "equivalent"
        else:
            decisions[dc["primary"]] = "inconclusive"
    # secondary with Holm over ten p-values
    pvals: dict[str, float | None] = {}
    for q in dc["secondary"]:
        t = quantities[q]["t"]
        pvals[f"{q}:support"] = None if t is None else t["p_support"]
        pvals[f"{q}:equivalence"] = None if t is None else t["p_equivalence"]
    adj = holm(pvals, dc["holm_alpha"])
    for q in dc["secondary"]:
        if quantities[q]["t"] is None:
            decisions[q] = "inconclusive"
        elif adj[f"{q}:support"]["reject"]:
            decisions[q] = "supported"
        elif adj[f"{q}:equivalence"]["reject"]:
            decisions[q] = "equivalent"
        else:
            decisions[q] = "inconclusive"
    # descriptive arm table
    arm_table: dict[str, dict] = {}
    for (d, a, s), r in rates.items():
        arm_table.setdefault(f"{d}_{a}", []).append(r)
    arm_summary = {k: {m: {"mean": float(np.mean([x[m] for x in v if x[m] is not None])), "sd": float(np.std([x[m] for x in v if x[m] is not None], ddof=1)) if len(v) > 1 else None}
                       for m in ("R_thr", "R_meanp", "R_thr_unweighted", "R_meanp_unweighted", "brier_w", "signed_error_vs_truth", "abs_error_vs_truth", "auroc_unweighted") if any(x[m] is not None for x in v)}
                   for k, v in arm_table.items()}
    judge_ref = None
    if judge is not None and truth:
        judge_ref = run_rates({"eval_ids": judge["eval_ids"], "p": judge["p_real"]}, support, truth, config["evaluation"]["threshold"])
        y = np.array([truth[i] for i in judge["eval_ids"]])
        h = np.asarray(judge["p_real"]) > config["evaluation"]["threshold"]
        judge_ref.update({"false_positive_rate": float(h[y == 0].mean()), "false_negative_rate": float((~h[y == 1]).mean()),
                          "validity_gate_auroc_min": config["judge"]["validity_auroc"],
                          "validity_gate_passed": bool(judge_ref["auroc_unweighted"] is not None and judge_ref["auroc_unweighted"] >= config["judge"]["validity_auroc"])})
    return {
        "config_sha256": sha256_json(config), "selection_sha256": selection["selection_sha256_without_certificate"], "support_sha256": support["sha256"],
        "margin_points": margin, "seeds": seeds, "n_runs_expected": len(expected_ids), "n_runs_completed": len(rates),
        "missing_runs": missing_runs, "failed_runs": failed_runs,
        "decisions": decisions, "interpretation": interpret(decisions),
        "holm": adj, "quantities": quantities, "arm_summary": arm_summary, "judge_on_real": judge_ref,
        "prevalence_pi": support["prevalence_pi"], "ess": support["ess"], "retained_eval_episodes": support["retained_test_episodes"],
    }


# ---------------------------------------------------------------- report

def report_markdown(res: dict) -> str:
    L = [f"# Experiment 2 decision report", "",
         f"Runs completed {res['n_runs_completed']} / {res['n_runs_expected']}; missing {len(res['missing_runs'])}; failed {len(res['failed_runs'])}. "
         f"Margin ±{res['margin_points']:g} points. π = {res['prevalence_pi']:.4f}; ESS = {res['ess']:.0f}; retained evaluation episodes = {res['retained_eval_episodes']}.", "",
         "## Decisions", "", "| Quantity | Role | Mean (points) | SD | One-sided 95% lower | 90% CI | p support | p equiv | Holm p (support / equiv) | Decision |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q, e in res["quantities"].items():
        role = "primary" if q == "A" else ("secondary" if q in res["decisions"] else "descriptive")
        dec = res["decisions"].get(q, "—")
        if e["t"] is None:
            wc = e.get("worst_case_bounds_on_mean")
            L.append(f"| {q} | {role} | — | — | — | — | — | — | — | **{dec}** (missing: {', '.join(e['missing_runs'])}; worst-case mean ∈ [{wc[0]:.1f}, {wc[1]:.1f}]) |")
            continue
        t = e["t"]
        hs = res["holm"].get(f"{q}:support", {}).get("p_holm")
        he = res["holm"].get(f"{q}:equivalence", {}).get("p_holm")
        holm_txt = "n/a" if hs is None else f"{hs:.4f} / {he:.4f}"
        L.append(f"| {q} | {role} | {t['mean']:.2f} | {t['sd']:.2f} | {t['lower95_one_sided']:.2f} | [{t['ci90'][0]:.2f}, {t['ci90'][1]:.2f}] | {t['p_support']:.4f} | {t['p_equivalence']:.4f} | {holm_txt} | **{dec}** |")
    L += ["", "## Interpretation (registered table)", ""] + [f"- {s}" for s in res["interpretation"]]
    L += ["", "## Arm summary (weighted thresholded success, mean over seeds)", "", "| Arm | R_thr | R_meanp | unweighted R_thr | Brier (w) | signed error vs truth | AUROC |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(res["arm_summary"].items()):
        au = v.get("auroc_unweighted", {}).get("mean")
        L.append(f"| {k} | {v['R_thr']['mean']:.4f} | {v['R_meanp']['mean']:.4f} | {v['R_thr_unweighted']['mean']:.4f} | {v['brier_w']['mean']:.4f} | {v['signed_error_vs_truth']['mean']:+.4f} | {'' if au is None else f'{au:.3f}'} |")
    if res["judge_on_real"]:
        j = res["judge_on_real"]
        L += ["", "## Judge on real held-out trajectories", "",
              f"Weighted thresholded rate {j['R_thr']:.4f} (truth {res['prevalence_pi']:.4f}); Brier {j['brier_w']:.4f}; AUROC {j['auroc_unweighted']:.3f}; "
              f"FPR {j['false_positive_rate']:.3f}; FNR {j['false_negative_rate']:.3f}; validity gate (AUROC ≥ {j['validity_gate_auroc_min']}) {'passed' if j['validity_gate_passed'] else 'FAILED'}."]
    if res["failed_runs"]:
        L += ["", "## Failed runs", ""] + [f"- {k}: {v}" for k, v in res["failed_runs"].items()]
    return "\n".join(L) + "\n"


def load_runs(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("*_s*.json"))]


def truth_from_manifest(manifest_path: Path) -> dict[str, int]:
    from .manifest import load_manifest
    m, _ = load_manifest(manifest_path)
    return {str(i): int(y) for i, y in zip(m.ids, m.success)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the registered Experiment 2 decision contract.")
    ap.add_argument("--runs", required=True, help="directory of per-run JSON files")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge", default=None, help="judge.json with eval_ids and p_real")
    ap.add_argument("--freeze", default=None, help="freeze manifest; analysis refuses to run if hashes differ or sign-offs are missing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    config = load_config(args.config)
    selection = json.loads(Path(args.selection).read_text())
    support = json.loads(Path(args.support).read_text())
    if args.freeze:
        from .freeze import verify_freeze
        verify_freeze(json.loads(Path(args.freeze).read_text()), config=config, selection=selection, support=support, require_signoff=True)
    judge = json.loads(Path(args.judge).read_text()) if args.judge else None
    res = analyze(load_runs(Path(args.runs)), selection, support, config, judge, truth_from_manifest(Path(args.manifest)))
    write_json(Path(args.out), res)
    Path(args.out).with_suffix(".md").write_text(report_markdown(res))
    print(report_markdown(res))


if __name__ == "__main__":
    main()
