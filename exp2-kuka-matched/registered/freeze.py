"""Freeze manifest for Experiment 2.

The manifest binds, by SHA-256: the registered config, the episode manifest, the certified selection
(all 110 run id lists), the evaluation support and weights, and the analyzer code. It records dated,
named sign-offs. ``verify_freeze`` is called by the driver before training (code and selection must
match) and by the analyzer before any decision (sign-offs must also be present). Hashes bind bytes;
the sign-off strings reject placeholders but cannot prove approval, so the manifest should be stored
next to the approval record (for example, the OSF registration and the owner's email).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from . import PACKAGE_ROOT, load_config, sha256_file, sha256_json, write_json

CODE_FILES = ("registered/__init__.py", "registered/manifest.py", "registered/design.py", "registered/weights.py",
              "registered/analyze.py", "registered/freeze.py", "registered/run_registered.py")
REQUIRED_ROLES = ("paper_lead", "stats_check")  # internal signers; the experiment owner may add a third line
PLACEHOLDERS = {"", "name", "full name", "tbd", "todo", "x", "owner", "lead"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FreezeError(Exception):
    pass


def code_hashes(root: Path = PACKAGE_ROOT) -> dict[str, str]:
    return {f: sha256_file(root / f) for f in CODE_FILES if (root / f).exists()}


def build_freeze(config: dict, manifest_path: Path, selection: dict, support: dict, approvals: list[dict], registration: str | None) -> dict:
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
        "approvals": approvals, "registration": registration,
        "statement": "Training runs start only after this manifest is signed and the registration is submitted. No hypothesis, margin, weight, support, or selection changes afterwards.",
    }


def verify_freeze(freeze: dict, *, config: dict, selection: dict | None = None, support: dict | None = None,
                  require_signoff: bool, check_code: bool = True) -> None:
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
    if require_signoff:
        roles = {a.get("role") for a in freeze.get("approvals", [])}
        missing = [r for r in REQUIRED_ROLES if r not in roles]
        if missing:
            raise FreezeError(f"missing sign-off roles: {missing}")
        for a in freeze["approvals"]:
            if str(a.get("name", "")).strip().lower() in PLACEHOLDERS or not DATE.match(str(a.get("date", ""))):
                raise FreezeError(f"placeholder or undated approval: {a}")


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
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--freeze", required=True)
    v.add_argument("--config", default=None)
    v.add_argument("--selection", default=None)
    v.add_argument("--support", default=None)
    v.add_argument("--require-signoff", action="store_true")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    if args.cmd == "create":
        approvals = []
        for a in args.approve:
            role, name, date = a.split(":", 2)
            approvals.append({"role": role, "name": name, "date": date})
        fz = build_freeze(config, Path(args.manifest), json.loads(Path(args.selection).read_text()), json.loads(Path(args.support).read_text()), approvals, args.registration)
        write_json(Path(args.out), fz)
        print(json.dumps({k: v for k, v in fz.items() if k != "code_sha256"}, indent=1))
    else:
        fz = json.loads(Path(args.freeze).read_text())
        verify_freeze(fz, config=config, selection=json.loads(Path(args.selection).read_text()) if args.selection else None,
                      support=json.loads(Path(args.support).read_text()) if args.support else None, require_signoff=args.require_signoff)
        print("freeze manifest verified")


if __name__ == "__main__":
    main()
