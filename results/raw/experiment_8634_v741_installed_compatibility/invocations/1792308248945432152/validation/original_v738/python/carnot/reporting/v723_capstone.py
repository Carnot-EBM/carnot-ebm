"""REQ-VERIFY-8401: use bounded shipped validation and checked atomic publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v721_capstone as base
from carnot.reporting import v723_capstone_evidence as e
from carnot.reporting import v717_contract_runner as validation

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze new statement coverage and private consumers; reuse authenticated global health."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(validation, "m", e):
        plan = validation.manifest(private)
    plan[0]["deadline"] = 900
    for name, tests in [
        ("full_contract_mutations", ["tests/python/test_v723_contract_methods_8388.py"]),
        ("private_E2E021", ["tests/python/test_restricted_decision_audit_8210.py"]),
        (
            "qualified_branch_consumers",
            [
                "tests/python/test_v721_capstone_frozen_aliases_8373.py",
                "tests/python/test_v721_capstone_worker_memory_8373.py",
                "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
            ],
        ),
    ]:
        plan.append(
            dict(
                name=name,
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *tests,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=600,
                scope="owned",
            )
        )
    return list(plan)


def terminal_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """A complete cold replay includes all branch children and must fit its actual workload."""
    if name == "terminal_cold":
        kwargs["deadline"] = 180
    return dict(base.child(name, argv, logs, **kwargs))


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Existing execution retains failures, exact streams and unchanged primary validators."""
    with (
        patch.multiple(base, e=e, manifest=manifest),
        patch.object(base.terminal, "child", terminal_child),
    ):
        return int(base.run(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """Real CLI controls stay private while the task owns one bounded invocation."""
    e.progress("start_no_model_load_current_LLM_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        return e.worker_process(args.worker_request, args.worker_output)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8401-owned-", dir="/var/tmp") as directory:
            return run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                failed=args.failed_child,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
