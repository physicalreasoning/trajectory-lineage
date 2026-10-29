"""Joint analysis of the Experiment 3 confirmation and its registered axes, with one Holm family.

Inputs: the main confirmation (capacity design; its primary claim is unchanged and comes from
``analyze_2d``) and four axis configurations, each with its own record export:

  architecture  exposure2d_confirm_latent_dinov2, exposure2d_confirm_latent_vjepa2
  dose          exposure2d_confirm_dose16 (about 50 presentations per reused trajectory),
                exposure2d_confirm_dose256 (about 3)

Every record set is validated and analyzed with ``analyze_2d`` under its own configuration (full
record validation, complete-block rule, competence screen, state hashes). The joint step then forms
ONE secondary family and applies Holm at the registered one-sided alpha:

  from the main confirmation   identity_specific, detectable, consequential_harm, capacity_increase
  architecture                 detectable_latent_dinov2:  G > 0 under the DINOv2 recipe
                               detectable_latent_vjepa2:  G > 0 under the V-JEPA 2 recipe
  dose                         dose_increase:  G(16 sources) - G(256 sources) > 0 (Welch)

Labels follow the Holm decisions; absence labels use two-sided 90% intervals inside the negligible
bound. An axis whose record set has an unresolved failure, or too few screened blocks, contributes
no p-value and is labelled unresolved; the main primary claim is never affected by an axis. The per
configuration four-axis classifications are reported as descriptive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_2d import analyze_2d, classify_gap_2d, welch  # noqa: E402
from analyze_confirmation import FIELDS, check_identity, holm, interval, one_sided_p  # noqa: E402
from freeze import FreezeError  # noqa: E402
from freeze import validate as validate_freeze  # noqa: E402

AXES = {
    "latent_dinov2": "architecture",
    "latent_vjepa2": "architecture",
    "dose16": "dose",
    "dose256": "dose",
}


def screened_gaps(result: dict[str, Any], runs: list[dict[str, Any]], config: dict[str, Any]) -> np.ndarray | None:
    """Reuse gaps of the screened blocks, or None when the configuration's decisions are suppressed."""

    if "primary" not in result or result["primary"].get("decisions_suppressed"):
        return None
    # recompute from the validated records used by analyze_2d: screened = completed and competent
    gaps = [run["contrasts"][FIELDS["G"]] for run in runs if run["status"] == "completed" and run.get("base_competent")]
    return np.array(gaps) if len(gaps) >= 2 else None


def analyze_axes(main: dict[str, Any], axes: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    """``main``: {"result": analyze_2d output, "config": config}; ``axes``: name -> (validated runs, config)."""

    contract = main["config"]["decision_contract"]
    alpha_support = float(contract["alpha_one_sided"])
    alpha_absence = float(contract["alpha_absence_one_sided"])
    negligible = float(contract["negligible_bound"])
    per_axis: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    gaps: dict[str, np.ndarray | None] = {}
    for name, (runs, config) in axes.items():
        result = analyze_2d(runs, config, check_records=False)  # records were validated by the caller
        g = screened_gaps(result, runs, config)
        gaps[name] = g
        entry: dict[str, Any] = {
            "kind": AXES[name],
            "scheduled": result["scheduled"],
            "completed": result["completed"],
            "screened": result["screened"],
            "failures": result["failures"],
            "descriptive_classification": result.get("primary", {}).get("axes", {}).get("magnitude", "unavailable"),
            "primary_claim_descriptive": result["primary_claim"],
        }
        if g is not None:
            g95, g90 = interval(g, alpha_support), interval(g, alpha_absence)
            entry.update(G_95=g95, G_90=g90, magnitude_descriptive=classify_gap_2d(g95, g90, contract))
            if AXES[name] == "architecture":
                p_values[f"detectable_{name}"] = one_sided_p(g, 0.0)
        per_axis[name] = entry
    # dose trend
    dose = None
    if gaps.get("dose16") is not None and gaps.get("dose256") is not None:
        t95 = welch(gaps["dose16"], gaps["dose256"], alpha_support)
        t90 = welch(gaps["dose16"], gaps["dose256"], alpha_absence)
        dose = {"contrast": "G(16 sources, ~50 presentations) - G(256 sources, ~3 presentations)", "ci95": t95, "ci90": {k: t90[k] for k in ("lower", "upper")}}
        p_values["dose_increase"] = t95["p_one_sided"]
    # one Holm family: main secondaries + axes
    main_p = dict(main["result"]["primary"]["secondary_p_values"]) if "primary" in main["result"] and not main["result"]["primary"].get("decisions_suppressed") else {}
    family = {**main_p, **p_values}
    decisions = holm(family, alpha_support) if family else {}
    labels: dict[str, str] = {}
    for name in ("latent_dinov2", "latent_vjepa2"):
        entry = per_axis.get(name)
        if entry is None or "G_90" not in entry:
            labels[name] = "unresolved"
        elif decisions.get(f"detectable_{name}"):
            labels[name] = "reuse_detectable_under_recipe"
        elif entry["G_90"]["lower"] > -negligible and entry["G_90"]["upper"] < negligible:
            labels[name] = "negligible_under_recipe"
        else:
            labels[name] = "unresolved"
    if dose is None:
        labels["dose"] = "unresolved"
    elif decisions.get("dose_increase"):
        labels["dose"] = "dose_dependent"
    elif dose["ci90"]["lower"] > -negligible and dose["ci90"]["upper"] < negligible:
        labels["dose"] = "dose_independent"
    else:
        labels["dose"] = "unresolved"
    main_labels = {}
    if main_p:
        main_labels = {
            "event_identity": "event_specific" if decisions.get("identity_specific") else main["result"]["primary"]["axes"]["event_identity"],
            "capacity": "capacity_dependent" if decisions.get("capacity_increase") else main["result"].get("capacity", {}).get("label", "n/a"),
        }
    return {
        "main_primary_claim": main["result"]["primary_claim"],
        "main_scheduled": main["result"]["scheduled"],
        "joint_family_p_values": family,
        "joint_holm_rejections": decisions,
        "main_labels_under_joint_holm": main_labels,
        "axes": per_axis,
        "dose_trend": dose,
        "labels": labels,
        "note": "Primary claim from analyze_2d on the main confirmation, unchanged. Secondary labels use the joint Holm family (main secondaries + axes).",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, default=Path("experiments/exposure2d_confirmation.json"))
    parser.add_argument("--main-records", type=Path, required=True)
    parser.add_argument("--axis", action="append", default=[], metavar="NAME=CONFIG:RECORDS", help="e.g. latent_dinov2=experiments/exposure2d_confirm_latent_dinov2.json:results/latent_dinov2_raw.json")
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_freeze(json.loads(args.freeze_manifest.read_text()), args.main_config, Path(__file__).with_name("analyze_2d.py"))
    except FreezeError as error:
        raise SystemExit(f"refusing to analyze: {error}") from error
    main_config = json.loads(args.main_config.read_text())
    main_runs = check_identity(json.loads(args.main_records.read_text()), args.freeze_manifest, args.main_config)
    main_result = analyze_2d(main_runs, main_config)
    axes: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    from analyze_confirmation import validate_records

    for item in args.axis:
        name, rest = item.split("=", 1)
        config_path, records_path = rest.split(":", 1)
        if name not in AXES:
            raise SystemExit(f"unknown axis {name}; expected one of {sorted(AXES)}")
        config = json.loads(Path(config_path).read_text())
        runs = json.loads(Path(records_path).read_text())
        runs = runs["runs"] if isinstance(runs, dict) and "runs" in runs else runs
        axes[name] = (validate_records(runs, int(config["blocks"]), config), config)
    result = analyze_axes({"result": main_result, "config": main_config}, axes)
    result["main"] = main_result
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"main_primary_claim": result["main_primary_claim"], "labels": result["labels"]}, indent=2))


if __name__ == "__main__":
    main()
