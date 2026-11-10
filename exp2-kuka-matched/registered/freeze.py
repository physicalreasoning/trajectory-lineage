"""Freeze manifest for Experiment 2.

The manifest binds, by SHA-256: the registered config, the episode manifest, the certified selection
(all 110 run id lists), the evaluation support and weights, and the analyzer code. It records dated,
named sign-offs, the owner's pipeline (git commit and the SHA-256 of ``experiment/wm.py``; the
preregistration names commit ed416e15), and the environment: the lockfile hash of the owner's
project, the pinned runtime versions with their hash, and the hardware class of the registered runs. ``verify_freeze`` is called
by the driver before training (code, selection and environment block must match) and by the analyzer
before any decision (sign-offs must also be present); ``check_runtime`` compares the live runtime and
device with the frozen ones when training starts. Hashes bind bytes;
the sign-off strings reject placeholders but cannot prove approval, so the manifest should be stored
next to the approval record (for example, the OSF registration and the owner's email).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

from . import PACKAGE_ROOT, load_config, sha256_file, sha256_json, write_json

CODE_FILES = ("registered/__init__.py", "registered/manifest.py", "registered/design.py", "registered/weights.py",
              "registered/analyze.py", "registered/freeze.py", "registered/run_registered.py")
REQUIRED_ROLES = ("paper_lead", "stats_check")  # internal signers; the experiment owner may add a third line
PLACEHOLDERS = {"", "name", "full name", "tbd", "todo", "x", "owner", "lead"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
OWNER_REPOSITORY = "https://github.com/subhashjprasad/robot-world-model-optimism-research"
OWNER_MODEL_FILE = "experiment/wm.py"
RUNTIME_PACKAGES = ("torch", "numpy", "scipy", "scikit-learn", "pandas", "transformers")


class FreezeError(Exception):
    pass


def code_hashes(root: Path = PACKAGE_ROOT) -> dict[str, str]:
    return {f: sha256_file(root / f) for f in CODE_FILES if (root / f).exists()}


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    try:
        import torch  # noqa: PLC0415  (optional)

        versions["cuda"] = str(torch.version.cuda or "none")
    except ImportError:
        versions["cuda"] = "none"
    return versions


def runtime_sha256(versions: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()


def detected_hardware(device: str) -> str:
    if str(device).startswith("cuda"):
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0)
    if str(device).startswith("mps"):
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def environment_block(hardware_class: str, lockfile: Path) -> dict:
    """Lockfile hash of the owner's project, pinned runtime versions and their hash, hardware class."""

    versions = runtime_versions()
    return {"lockfile": str(lockfile), "lockfile_sha256": sha256_file(Path(lockfile)), "runtime": versions,
            "runtime_sha256": runtime_sha256(versions), "hardware_class": hardware_class}


def owner_commit(owner_repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(owner_repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(f"cannot read the git commit of the owner's repository at {owner_repo}") from error


def owner_pipeline_block(owner_repo: Path) -> dict:
    """The owner's pipeline as it is run: git commit and the SHA-256 of the model file."""

    return {"repository": OWNER_REPOSITORY, "commit": owner_commit(owner_repo),
            "wm_py_sha256": sha256_file(Path(owner_repo) / OWNER_MODEL_FILE)}


def validate_owner_pipeline(pipeline: object, owner_repo: Path | None = None) -> None:
    """The block must name a full commit and a file hash; with ``owner_repo``, both must match it."""

    if not isinstance(pipeline, dict) or not HEX40.match(str(pipeline.get("commit", ""))) \
            or not HEX64.match(str(pipeline.get("wm_py_sha256", ""))):
        raise FreezeError("the freeze manifest must record the owner's pipeline commit and the wm.py hash")
    if owner_repo is None:
        return
    if sha256_file(Path(owner_repo) / OWNER_MODEL_FILE) != pipeline["wm_py_sha256"]:
        raise FreezeError(f"the owner's {OWNER_MODEL_FILE} differs from the frozen file (edited or another commit)")
    head = owner_commit(owner_repo)
    if head != pipeline["commit"]:
        raise FreezeError(f"the owner's repository is at {head[:12]}, not at the frozen commit {pipeline['commit'][:12]}")


def load_environment(path: Path) -> dict:
    """An environment block from a file: a bare block, or a freeze manifest that holds one."""

    data = json.loads(Path(path).read_text())
    return data["environment"] if isinstance(data, dict) and "environment" in data else data


def validate_environment(environment: object, lockfile: Path | None = None) -> None:
    if not isinstance(environment, dict):
        raise FreezeError("the freeze manifest has no environment block")
    if not HEX64.match(str(environment.get("lockfile_sha256", ""))):
        raise FreezeError("environment lockfile_sha256 is missing")
    if lockfile is not None and environment["lockfile_sha256"] != sha256_file(Path(lockfile)):
        raise FreezeError("the owner's lockfile differs from the frozen one")
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict) or not {"python", *RUNTIME_PACKAGES} <= set(runtime):
        raise FreezeError("environment runtime must list python and every registered package")
    if environment.get("runtime_sha256") != runtime_sha256(runtime):
        raise FreezeError("environment runtime_sha256 does not match its runtime versions")
    hardware = str(environment.get("hardware_class", "")).strip()
    if len(hardware) < 2 or hardware.lower() in PLACEHOLDERS:
        raise FreezeError("environment must name the hardware class of the registered runs")


def check_runtime(freeze: dict, device: str, deviation: str | None = None) -> dict:
    """Refuse training on another hardware class or runtime unless a logged deviation is given."""

    environment = freeze["environment"]
    live = runtime_versions()
    detected = detected_hardware(device)
    differences = {k: [environment["runtime"].get(k), v] for k, v in live.items() if environment["runtime"].get(k) != v}
    hardware_ok = environment["hardware_class"].strip().lower() in detected.lower()
    record = {"declared_hardware_class": environment["hardware_class"], "detected_hardware": detected, "hardware_matches": hardware_ok,
              "runtime_differences": differences, "runtime_sha256_live": runtime_sha256(live), "deviation": deviation}
    if (differences or not hardware_ok) and not (deviation and len(deviation.strip()) >= 10):
        raise FreezeError(f"environment differs from the freeze: hardware {detected!r} vs {environment['hardware_class']!r}, runtime {differences}; "
                          "rerun on the frozen environment or pass --deviation with a reason (logged)")
    return record


def build_freeze(config: dict, manifest_path: Path, selection: dict, support: dict, approvals: list[dict], registration: str | None,
                 environment: dict | None = None, owner_pipeline: dict | None = None) -> dict:
    if selection.get("manifest_kind") == "SYNTHETIC":
        raise FreezeError("refusing to freeze a selection built on a SYNTHETIC manifest")
    if not selection["certificate"]["feasible"]:
        raise FreezeError("refusing to freeze: certificate reports failed checks")
    if support["manifest_sha256"] != selection["manifest_sha256"]:
        raise FreezeError("support and selection were built from different manifests")
    return {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_name": config["name"], "config_version": config["version"], "config_sha256": sha256_json(config),
        "manifest_sha256": selection["manifest_sha256"], "manifest_path": str(manifest_path),
        "selection_sha256": selection["selection_sha256_without_certificate"],
        "support_sha256": support["sha256"],
        "code_sha256": code_hashes(),
        "design_B_budget": selection["design_B"]["budget_transitions"], "n_runs": selection["n_runs"],
        "approvals": approvals, "registration": registration, "environment": environment, "owner_pipeline": owner_pipeline,
        "status": "signed" if {a.get("role") for a in approvals} >= set(REQUIRED_ROLES) else "DRAFT - unsigned",
        "statement": "Training runs start only after this manifest is signed and the registration is submitted. No hypothesis, margin, weight, support, or selection changes afterwards.",
    }


def verify_freeze(freeze: dict, *, config: dict, selection: dict | None = None, support: dict | None = None,
                  require_signoff: bool, check_code: bool = True, lockfile: Path | None = None,
                  owner_repo: Path | None = None) -> None:
    if sha256_json(config) != freeze["config_sha256"]:
        raise FreezeError("config differs from the frozen config")
    if selection is not None and selection["selection_sha256_without_certificate"] != freeze["selection_sha256"]:
        raise FreezeError("selection differs from the frozen selection")
    if support is not None and support["sha256"] != freeze["support_sha256"]:
        raise FreezeError("support/weights differ from the frozen support")
    if check_code:
        now = code_hashes()
        changed = [f for f, h in freeze["code_sha256"].items() if now.get(f) != h]
        if changed:
            raise FreezeError(f"code changed since freeze: {changed}")
    validate_environment(freeze.get("environment"), lockfile)
    validate_owner_pipeline(freeze.get("owner_pipeline"), owner_repo)
    if require_signoff:
        roles = {a.get("role") for a in freeze.get("approvals", [])}
        missing = [r for r in REQUIRED_ROLES if r not in roles]
        if missing:
            raise FreezeError(f"missing sign-off roles: {missing}")
        for a in freeze["approvals"]:
            if str(a.get("name", "")).strip().lower() in PLACEHOLDERS or not DATE.match(str(a.get("date", ""))):
                raise FreezeError(f"placeholder or undated approval: {a}")


def sign(freeze: dict, role: str, name: str, date: str) -> dict:
    """Add one internal sign-off to a draft; it becomes signed once every required role has signed."""

    if role not in REQUIRED_ROLES:
        raise FreezeError(f"role must be one of {REQUIRED_ROLES}")
    if str(name).strip().lower() in PLACEHOLDERS or len(str(name).split()) < 2:
        raise FreezeError("sign with a full name")
    if not DATE.match(date):
        raise FreezeError("date must be YYYY-MM-DD")
    approvals = [a for a in freeze.get("approvals", []) if a.get("role") != role] + [{"role": role, "name": name, "date": date}]
    signed = {**freeze, "approvals": approvals}
    signed["status"] = "signed" if {a["role"] for a in approvals} >= set(REQUIRED_ROLES) else "DRAFT - unsigned"
    return signed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Create or verify the Experiment 2 freeze manifest.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--config", default=None)
    c.add_argument("--manifest", required=True)
    c.add_argument("--selection", required=True)
    c.add_argument("--support", required=True)
    c.add_argument("--approve", action="append", default=[], help='"role:Full Name:YYYY-MM-DD" (repeat)')
    c.add_argument("--registration", default=None, help="OSF registration URL or DOI")
    c.add_argument("--owner-repo", required=True, help="robot-world-model-optimism-research checked out at the registered commit")
    c.add_argument("--hardware-class", default=None, help='hardware of the registered runs, for example "A100" (not with --environment)')
    c.add_argument("--environment", default=None,
                   help="reuse an environment block recorded on the registered hardware (a block, or a freeze manifest that has one)")
    c.add_argument("--lockfile", default=None, help="uv.lock of the owner's project (default: <owner-repo>/uv.lock)")
    c.add_argument("--out", required=True)
    g = sub.add_parser("sign", help="add one internal sign-off to a freeze manifest")
    g.add_argument("--freeze", required=True)
    g.add_argument("--role", required=True, choices=REQUIRED_ROLES)
    g.add_argument("--name", required=True)
    g.add_argument("--date", required=True, help="YYYY-MM-DD")
    v = sub.add_parser("verify")
    v.add_argument("--freeze", required=True)
    v.add_argument("--config", default=None)
    v.add_argument("--selection", default=None)
    v.add_argument("--support", default=None)
    v.add_argument("--require-signoff", action="store_true")
    v.add_argument("--lockfile", default=None, help="check the owner's uv.lock against the frozen hash")
    v.add_argument("--owner-repo", default=None, help="check the owner's commit and wm.py against the frozen ones")
    args = ap.parse_args(argv)
    config = load_config(getattr(args, "config", None))
    if args.cmd == "create":
        approvals = []
        for a in args.approve:
            role, name, date = a.split(":", 2)
            approvals.append({"role": role, "name": name, "date": date})
        lockfile = Path(args.lockfile) if args.lockfile else Path(args.owner_repo) / "uv.lock"
        if (args.environment is None) == (args.hardware_class is None):
            raise SystemExit("give exactly one of --hardware-class (record this machine) or --environment (reuse a recorded block)")
        if args.environment:
            environment = load_environment(Path(args.environment))
            validate_environment(environment, lockfile)
        else:
            environment = environment_block(args.hardware_class, lockfile)
        fz = build_freeze(config, Path(args.manifest), json.loads(Path(args.selection).read_text()), json.loads(Path(args.support).read_text()), approvals, args.registration,
                          environment, owner_pipeline_block(Path(args.owner_repo)))
        write_json(Path(args.out), fz)
        print(json.dumps({k: v for k, v in fz.items() if k != "code_sha256"}, indent=1))
    elif args.cmd == "sign":
        signed = sign(json.loads(Path(args.freeze).read_text()), args.role, args.name, args.date)
        write_json(Path(args.freeze), signed)
        print(f"{args.freeze}: {signed['status']}; signed by {sorted(a['role'] for a in signed['approvals'])}")
    else:
        fz = json.loads(Path(args.freeze).read_text())
        verify_freeze(fz, config=config, selection=json.loads(Path(args.selection).read_text()) if args.selection else None,
                      support=json.loads(Path(args.support).read_text()) if args.support else None, require_signoff=args.require_signoff,
                      lockfile=Path(args.lockfile) if args.lockfile else None, owner_repo=Path(args.owner_repo) if args.owner_repo else None)
        print("freeze manifest verified")


if __name__ == "__main__":
    main()
