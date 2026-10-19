"""REQ-VERIFY-8374: reuse bounded children and unchanged terminal validators."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as base
from carnot.reporting import v718_replay_runner as qualified
from carnot.reporting import v722_contract_methods as e

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache" / "carnot-exp8374-private"


def manifest(private: Path) -> list[Json]:
    """Freeze scoped coverage and real private consumer checks before aggregation."""
    with patch.object(base, "m", e):
        plan = list(base.manifest(private))
    plan[0]["deadline"] = 900
    plan.extend(
        [
            dict(
                name="private_E2E018_V722",
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    e.TEST + "::test_private_e2e018",
                    "--basetemp=" + str(private / "e2e018-v722"),
                ],
                expected=0,
                deadline=180,
                scope="owned",
            ),
            dict(
                name="private_disk_scratch",
                argv=["/usr/bin/df", "-T", str(private)],
                expected=0,
                deadline=10,
                scope="owned",
            ),
            dict(
                name="repository_health_once",
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--basetemp=" + str(private / "global-suite"),
                ],
                expected=0,
                deadline=900,
                scope="global",
            ),
        ]
    )
    for spec in plan:
        spec["task_cap_s"] = 4800
    return plan


def main(argv: list[str] | None = None) -> int:
    """Keep private disk scratch and a hard cap while reusing qualified execution."""
    args = list(sys.argv[1:] if argv is None else argv)
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    if "--date" in args:
        index = args.index("--date") + 1
        if index == len(args) or args[index] != "20261010":
            raise SystemExit("date must be 20261010")
        args[index] = "20261008"
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with (
            patch("tempfile.tempdir", str(SCRATCH)),
            patch.dict(
                os.environ,
                {"TMPDIR": str(SCRATCH), "PYTHONUNBUFFERED": "1", "JAX_PLATFORMS": "cpu"},
            ),
            patch.object(qualified, "e", e),
            patch.object(qualified, "manifest", manifest),
            patch.object(base, "m", e),
        ):
            return int(qualified.main(args))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
