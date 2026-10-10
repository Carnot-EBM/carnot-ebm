"""REQ-VERIFY-8381: small adapters reuse bounded children and checked publication."""

from __future__ import annotations

import os
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import logit_policy_certificate_8381 as e
from carnot.reporting import spline_table_execution_8352 as base
from carnot.reporting import threshold_guard_8362 as historical
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.verify.dyadic_logit_v1 import progress

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8381-private"
ORIGINAL_CONTROLS = base.controls


def manifest(private: Path) -> list[Json]:
    """Freeze coverage and historical authority consumers before numerical measurement."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(commands, "m", e):
        plan = list(commands.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].append("tests/python/test_local_update_isolation_8306.py")
    plan.append(
        dict(
            name="threshold_consumers",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                "from pathlib import Path; from carnot.reporting.logit_policy_execution_8381 import historical_consumers; raise SystemExit(historical_consumers(Path("
                + repr(str(private / "historical"))
                + ")))",
            ],
            expected=0,
            deadline=300,
            scope="owned",
        )
    )
    return plan


def historical_consumers(private: Path) -> int:
    """Original assertions run against preserved authority instead of mutable active task bytes."""
    import pytest

    private.mkdir(parents=True, exist_ok=True)
    for name in ["results", "python", "scripts"]:
        (private / name).symlink_to(e.ROOT / name, target_is_directory=True)
    (private / "openspec/change-proposals").mkdir(parents=True)
    prior = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    work = json.loads(Path(prior["work_reference"]["path"]).read_bytes())
    for name in ["research-roadmap.yaml", "research-roadmap-vNEXT.md"]:
        ref = next(r for r in work["inputs"]["refs"] if Path(r["source_path"]).name == name)
        dest = private / (name if name.endswith("yaml") else "openspec/change-proposals/" + name)
        dest.write_bytes(Path(ref["path"]).read_bytes())
    for name in [
        "research-roadmap-v720-preserved-20261009.md",
        "v717-local-learning-protocol.json",
    ]:
        (private / "openspec/change-proposals" / name).symlink_to(
            e.ROOT / "openspec/change-proposals" / name
        )
    original = historical.authenticate
    with patch.object(
        historical,
        "authenticate",
        lambda root, raw: original(private if root == e.ROOT else root, raw),
    ):
        return int(
            pytest.main(
                [
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/test_threshold_guard_8362.py",
                    "--basetemp=" + str(private / "tests"),
                ]
            )
        )


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh children reject repaired primitive bytes and a truly absent replay input."""
    receipts: list[Json] = list(ORIGINAL_CONTROLS(value, raw))
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if work.get("primitive_refs"):
        attack = raw / "primitive_tamper"
        primitives = json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
        primitives["rows"][0]["action"] = "tampered"
        atomic_json(attack / "primitives.json", primitives)
        changed = deepcopy(work)
        changed["primitive_refs"] = [e.reference(attack / "primitives.json")]
        atomic_json(attack / "measurement.json", changed)
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (e.NAME + ".json")
        candidate = attack / "candidate.json"
        atomic_json(candidate, e.build(changed, value["validation_receipts"], attack, output))
        receipts.append(
            child(
                "cold_primitive_tamper",
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(candidate),
                ],
                raw / "cold",
                expected=1,
                deadline=120,
                heartbeat=20,
            )
        )
    receipts.append(
        child(
            "cold_missing_input",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(raw / "absent.json"),
            ],
            raw / "cold",
            expected=1,
            deadline=30,
            heartbeat=20,
        )
    )
    receipts.append(
        child(
            "deliberate_error",
            [str(e.ROOT / ".venv/bin/python"), "-u", "-c", "raise SystemExit(7)"],
            raw / "cold",
            expected=7,
            deadline=30,
            heartbeat=20,
        )
    )
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Disk scratch, memory and the task alarm bound the reused numerical pipeline."""
    args = list(sys.argv[1:] if argv is None else argv)
    progress("start_exp8381")
    if "--date" in args:
        i = args.index("--date") + 1
        if i == len(args) or args[i] != "20261010":
            raise SystemExit("date must be 20261010")
        args[i] = "20261009"
    SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
    SCRATCH.chmod(0o700)
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if SCRATCH.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    resources = dict(
        private_scratch=str(SCRATCH),
        scratch_mode=oct(SCRATCH.stat().st_mode & 0o777),
        filesystem=mount[mount.index("-") + 1],
        available_memory_bytes=memory,
        task_cap_s=4800,
        child_poll_max_s=30,
        current_llm_calls=0,
    )
    original_authenticate = e.authenticate

    def authenticate(root: Path, raw: Path) -> Json:
        e.require(
            SCRATCH, "disk_backed_scratch", True, resources["filesystem"] not in ["tmpfs", "ramfs"]
        )
        e.require(SCRATCH, "available_memory_at_least_1GB", True, memory >= 1_000_000_000)
        return dict(original_authenticate(root, raw), resources=resources)

    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    with (
        patch.object(base, "e", e),
        patch.object(base, "manifest", manifest),
        patch.object(base, "progress", progress),
        patch.object(base, "controls", controls),
        patch.object(e, "authenticate", authenticate),
        patch("tempfile.tempdir", str(SCRATCH)),
        patch.dict(os.environ, {"TMPDIR": str(SCRATCH), "PYTHONUNBUFFERED": "1"}),
    ):
        try:
            return int(base.main(args))
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous)
