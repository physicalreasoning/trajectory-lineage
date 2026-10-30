"""Freeze-manifest validation shared by the exposure runner and the confirmatory analyzer.

A manifest binds the confirmatory configuration(s), a clean code commit, the analyzer, dated named
sign-offs, the agreed decision statement, every registered pilot seed and, where the configuration
requires it, the environment: the lockfile hash, the pinned runtime versions with their hash, and
the hardware class of the confirmatory runs. Hashes bind bytes; the string checks reject obvious
placeholders but cannot by themselves prove that approval occurred, so the manifest should be
committed alongside the approval record (for example, the email).

One manifest may bind several configurations (``"configs": {name: sha256}``), for example the main
2D confirmation and its four axis configurations.

Command line (run on the confirmatory runtime, before the freeze):

    python scripts/freeze.py environment --hardware-class A100 --out approvals/environment.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Callable

PLACEHOLDERS = {"full name", "name", "tbd", "todo", "x", "a", "approver"}
REQUIRED_ROLES = ("paper_lead", "stats_check")
PROVISIONAL_ROLES = ("workstream_owner",)
REPOSITORY = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPOSITORY / "experiments"
APPROVAL_PATHS = ("approvals/",)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONFIRMATORY_ROLES = ("confirmatory", "confirmatory_axis")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# packages whose versions can change the numbers; pinned in the manifest and checked at run start
RUNTIME_PACKAGES = ("torch", "numpy", "scipy", "h5py", "transformers", "accelerate")


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
    for path in sorted(experiments.glob("exposure*_pilot_*.json")):
        pilot = json.loads(path.read_text())
        if "experiment_seed" in pilot:
            seeds.add(int(pilot["experiment_seed"]))
    return seeds


def runtime_versions() -> dict[str, str]:
    """Python, the registered packages and the CUDA build of torch, for the current process."""

    versions = {"python": platform.python_version()}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    try:
        import torch  # noqa: PLC0415  (optional; base environments have no torch)

        versions["cuda"] = str(torch.version.cuda or "none")
    except ImportError:
        versions["cuda"] = "none"
    return versions


def runtime_sha256(versions: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()


def detected_hardware(device: str) -> str:
    """A readable name for the device the run uses (GPU model, Apple MPS, or the CPU)."""

    if str(device).startswith("cuda"):
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0)
    if str(device).startswith("mps"):
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def hardware_matches(declared: str, detected: str) -> bool:
    """The declared class (for example "A100") appears in the detected device name."""

    return bool(declared.strip()) and declared.strip().lower() in detected.lower()


def environment_block(hardware_class: str, lockfile: Path | None = None) -> dict:
    lockfile = lockfile or REPOSITORY / "uv.lock"
    versions = runtime_versions()
    return {
        "lockfile": lockfile.name,
        "lockfile_sha256": sha256(lockfile),
        "runtime": versions,
        "runtime_sha256": runtime_sha256(versions),
        "hardware_class": hardware_class,
    }


def validate_environment(environment: object, repository: Path = REPOSITORY) -> None:
    """The manifest's environment block is complete and its lockfile matches the repository."""

    if not isinstance(environment, dict):
        raise FreezeError("this configuration requires an environment block in the manifest")
    lockfile = repository / str(environment.get("lockfile", "uv.lock"))
    if not lockfile.exists() or environment.get("lockfile_sha256") != sha256(lockfile):
        raise FreezeError("environment lockfile_sha256 does not match the repository lockfile")
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict) or not {"python", *RUNTIME_PACKAGES} <= set(runtime):
        raise FreezeError("environment runtime must list python and every registered package")
    if environment.get("runtime_sha256") != runtime_sha256(runtime):
        raise FreezeError("environment runtime_sha256 does not match its runtime versions")
    hardware = str(environment.get("hardware_class", "")).strip()
    if len(hardware) < 2 or hardware.lower() in PLACEHOLDERS:
        raise FreezeError("environment must name the hardware class of the confirmatory runs")


def check_runtime(manifest: dict, device: str, deviation: str | None = None) -> dict:
    """Compare the live runtime and device with the frozen environment at run start.

    A different hardware class or runtime refuses the run unless ``deviation`` gives a reason; the
    returned record (stored with the outputs) logs what was compared and any deviation."""

    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        return {"checked": False, "reason": "manifest has no environment block"}
    live = runtime_versions()
    detected = detected_hardware(device)
    differences = {
        k: [environment["runtime"].get(k), v]
        for k, v in live.items()
        if environment["runtime"].get(k) != v
    }
    hardware_ok = hardware_matches(str(environment["hardware_class"]), detected)
    record = {
        "checked": True,
        "declared_hardware_class": environment["hardware_class"],
        "detected_hardware": detected,
        "hardware_matches": hardware_ok,
        "runtime_differences": differences,
        "runtime_sha256_live": runtime_sha256(live),
        "deviation": deviation,
    }
    if (differences or not hardware_ok) and not (deviation and len(deviation.strip()) >= 10):
        problems = []
        if not hardware_ok:
            problems.append(
                f"hardware {detected!r} is not the frozen class {environment['hardware_class']!r}"
            )
        if differences:
            problems.append(f"runtime differs from the freeze: {differences}")
        raise FreezeError(
            "; ".join(problems)
            + " (a rerun uses the frozen environment; otherwise pass --deviation with a reason,"
            " which is logged)"
        )
    return record


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
    repository: Path = REPOSITORY,
) -> None:
    config = json.loads(config_path.read_text())
    if config.get("role") not in CONFIRMATORY_ROLES:
        raise FreezeError("only a confirmatory configuration can be unblinded")
    bound = manifest.get("configs")
    if isinstance(bound, dict):
        if bound.get(config_path.stem) != sha256(config_path):
            raise FreezeError(f"manifest does not bind this configuration ({config_path.stem})")
    elif manifest.get("config_sha256") != sha256(config_path):
        raise FreezeError("manifest does not match the configuration")
    if not code_matches(str(manifest.get("code_commit")), state, changes):
        raise FreezeError(
            "code differs from the frozen commit (only files under approvals/ may change)"
        )
    if manifest.get("analyzer_sha256") != sha256(analyzer_path):
        raise FreezeError("analysis code differs from the frozen analyzer")
    mode = manifest.get("mode", "confirmatory")
    if mode == "provisional" and not config.get("allow_provisional_freeze", True):
        raise FreezeError(
            "this configuration requires confirmatory approvals (no provisional freeze)"
        )
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
    if mode == "confirmatory" and config.get("require_environment"):
        validate_environment(manifest.get("environment"), repository)


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze helpers")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("environment", help="write the environment block for the freeze manifest")
    env.add_argument("--hardware-class", required=True, help='for example "A100"')
    env.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    block = environment_block(args.hardware_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")
    print(json.dumps(block, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
