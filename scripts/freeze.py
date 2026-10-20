"""Freeze-manifest validation shared by the exposure runner and the confirmatory analyzer.

A manifest binds the confirmatory configuration, a clean code commit, the analyzer, dated named
sign-offs, the agreed decision statement, and every registered pilot seed. Hashes bind bytes; the
string checks reject obvious placeholders but cannot by themselves prove that approval occurred,
so the manifest should be committed alongside the approval record (for example, the email).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

PLACEHOLDERS = {"full name", "name", "tbd", "todo", "x", "a", "approver"}
REQUIRED_ROLES = ("paper_lead", "evaluation_science")
PROVISIONAL_ROLES = ("workstream_owner",)
REPOSITORY = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPOSITORY / "experiments"
APPROVAL_PATHS = ("approvals/",)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FreezeError(Exception):
    """Raised when a manifest does not authorize unblinding or analysis."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> tuple[str, bool]:
    """Return the HEAD commit and whether tracked files are unmodified."""

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", False
    return commit, status == ""


def changed_since(commit: str) -> list[str] | None:
    """Tracked files changed between ``commit`` and HEAD, or None if not an ancestor."""

    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output = subprocess.check_output(["git", "diff", "--name-only", commit, "HEAD"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in output.splitlines() if line]


def code_matches(frozen: str, state: Callable[[], tuple[str, bool]], changes) -> bool:
    """HEAD equals the frozen commit, or differs from it only by approval records."""

    head, clean = state()
    if not clean:
        return False
    if head == frozen:
        return True
    changed = changes(frozen)
    return changed is not None and all(
        any(path.startswith(prefix) for prefix in APPROVAL_PATHS) for path in changed
    )


def pilot_registry(experiments: Path, config: dict) -> set[int]:
    """Every registered pilot seed: exposure_pilot_*.json files plus declared extra seeds."""

    seeds = {int(seed) for seed in config.get("excluded_pilot_seeds", [])}
    for path in sorted(experiments.glob("exposure_pilot_*.json")):
        pilot = json.loads(path.read_text())
        if "experiment_seed" in pilot:
            seeds.add(int(pilot["experiment_seed"]))
    return seeds


def valid_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    parts = text.split()
    return text.lower() not in PLACEHOLDERS and len(parts) >= 2 and all(len(p) >= 2 for p in parts)


def validate(
    manifest: dict,
    config_path: Path,
    analyzer_path: Path,
    *,
    state: Callable[[], tuple[str, bool]] = git_state,
    changes: Callable[[str], list[str] | None] = changed_since,
    experiments: Path = EXPERIMENTS,
) -> None:
    config = json.loads(config_path.read_text())
    if config.get("role") != "confirmatory":
        raise FreezeError("only a confirmatory configuration can be unblinded")
    if manifest.get("config_sha256") != sha256(config_path):
        raise FreezeError("manifest does not match the configuration")
    if not code_matches(str(manifest.get("code_commit")), state, changes):
        raise FreezeError(
            "code differs from the frozen commit (only files under approvals/ may change)"
        )
    if manifest.get("analyzer_sha256") != sha256(analyzer_path):
        raise FreezeError("analysis code differs from the frozen analyzer")
    mode = manifest.get("mode", "confirmatory")
    if mode == "provisional":
        # Thresholds frozen by the workstream owner before unblinding while collaborator review is
        # pending. Outputs are stamped provisional; collaborator signatures are never implied.
        roles = list(PROVISIONAL_ROLES)
        if "provisional" not in str(manifest.get("decision_statement", "")).lower():
            raise FreezeError("a provisional manifest must say so in its decision statement")
    elif mode == "confirmatory":
        roles = list(
            dict.fromkeys([*REQUIRED_ROLES, *(config.get("required_signoff_roles") or [])])
        )
    else:
        raise FreezeError("manifest mode must be confirmatory or provisional")
    signoffs = manifest.get("signoffs", {})
    for role in roles:
        if not valid_name(signoffs.get(role)):
            raise FreezeError(f"missing or placeholder sign-off for role {role!r}")
    if not DATE.match(str(manifest.get("signoff_date", ""))):
        raise FreezeError("signoff_date must be YYYY-MM-DD")
    if len(str(manifest.get("decision_statement", "")).strip()) < 40:
        raise FreezeError("manifest must record the agreed decision behind the thresholds")
    registry = pilot_registry(experiments, config)
    excluded = {int(seed) for seed in manifest.get("excluded_pilot_seeds", [])}
    if not registry <= excluded:
        raise FreezeError(f"manifest omits registered pilot seeds {sorted(registry - excluded)}")
    if int(config["experiment_seed"]) in registry | excluded:
        raise FreezeError("confirmation seed collides with a pilot seed")


def run_identity(manifest_path: Path, config_path: Path) -> dict:
    """Identifiers written into raw exports and checked again at analysis time."""

    config = json.loads(config_path.read_text())
    return {
        "config_sha256": sha256(config_path),
        "experiment_seed": int(config["experiment_seed"]),
        "manifest_sha256": sha256(manifest_path),
        "code_commit": json.loads(manifest_path.read_text()).get("code_commit"),
        "mode": json.loads(manifest_path.read_text()).get("mode", "confirmatory"),
    }
