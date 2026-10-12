"""REQ-VERIFY-8443: reuse bounded publication checks with current dated operands."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v725_contract_runner as shipped
from carnot.reporting import v727_contract_methods as e
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.v709_execution import child as original_child

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Reuse serial owned coverage and E2E-018, adding the frozen audit CLI regression."""
    with patch.object(shipped, "e", e):
        plan = list(_MANIFEST(private))
    plan.insert(
        2,
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "frozen-replay"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    owned = next(p for p in plan if p["name"] == "owned_tests")
    owned["deadline"] = 1200
    owned["argv"].insert(-1, e.EXTRA_TEST)
    for p in plan:
        if p["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            p["argv"].append(e.EXTRA_TEST)
    return plan


def bounded_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Original assertions stay fixed while full historical replay gets a bounded clock."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 240
    return dict(original_child(name, argv, logs, **kwargs))


def main(argv: list[str] | None = None) -> int:
    """The thin dated entry owns its deadline while imported consumers stay unchanged."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--history-request", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(argv)
    if args.history_request:
        if not args.history_output:
            parser.error("history output required")
        return int(history.worker(args.history_request, args.history_output))
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8443-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped, "e", e),
                patch.object(shipped, "manifest", manifest),
                patch.object(shipped, "child", bounded_child),
                patch.object(shipped.shipped, "child", bounded_child),
            ):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
