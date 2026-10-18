"""REQ-VERIFY-8362: use the established bounded child and publication pipeline."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import spline_table_execution_8352 as base
from carnot.reporting import threshold_guard_8362 as e
from carnot.reporting import v717_contract_runner as commands
from carnot.verify.threshold_guard_8362 import progress
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ORIGINAL_CONTROLS = base.controls
SCRATCH = Path.home() / ".cache/carnot-exp8362-private"


def manifest(private: Path) -> list[Json]:
    """Freeze scoped owned coverage, original kernels and private E2E-018 consumers."""
    with patch.object(commands, "m", e):
        plan = list(commands.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].append("tests/python/test_local_update_isolation_8306.py")
    plan.append(
        dict(
            name="private_E2E018_guard",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                e.TEST + "::test_private_e2e018",
                "--basetemp=" + str(private / "e2e018"),
            ],
            expected=0,
            deadline=180,
            scope="owned",
        )
    )
    return plan


def controls(value: Json, raw: Path) -> list[Json]:
    """A repaired primitive hash still needs independent numerical recomputation."""
    with patch.object(base, "e", e):
        receipts: list[Json] = list(ORIGINAL_CONTROLS(value, raw))
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if work.get("primitive_refs"):
        attack = raw / "primitive_tamper"
        primitives = json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
        primitives["rows"][0]["x"][0] += 0.125
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
    return receipts


def main(argv: list[str] | None = None) -> int:
    """A task cap and disk scratch bound the reused unbuffered runner."""
    args = list(sys.argv[1:] if argv is None else argv)
    progress("start")
    if "--date" in args:
        index = args.index("--date") + 1
        if index == len(args) or args[index] != "20261010":
            raise SystemExit("date must be 20261010")
        args[index] = "20261009"
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with (
            patch("tempfile.tempdir", str(SCRATCH)),
            patch.dict(os.environ, {"TMPDIR": str(SCRATCH), "PYTHONUNBUFFERED": "1"}),
            patch.object(base, "e", e),
            patch.object(base, "manifest", manifest),
            patch.object(base, "progress", progress),
            patch.object(base, "controls", controls),
        ):
            return int(base.main(args))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
