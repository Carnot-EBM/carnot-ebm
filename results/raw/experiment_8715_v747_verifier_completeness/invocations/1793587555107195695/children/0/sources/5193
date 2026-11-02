"""REQ-VERIFY-8589: reuse bounded private validation and exact-byte publication.

Only task identity and the scoped test list change. Historical test authority,
publication validators and reader selection retain their original behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_runner_8454 as shipped
from carnot.reporting import nine_input_board_runner_8538 as validation
from carnot.reporting import resident_hardware_boundary_8589 as e
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]
SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]


def manifest(private: Path) -> list[Json]:
    """Freeze original nodes before disjoint children; verbose results prove their exact union."""
    with patch.multiple(validation, e=e, SUITES=SUITES):
        plan = validation.manifest(private)
    for row in plan:
        if "collected_node_ids" in row:
            row["argv"] = ["-vv" if token == "-q" else token for token in row["argv"]]
    return list(plan)


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, owned_failure: bool = False
) -> int:
    """A private failure executes a real nonzero child; no fabricated receipt grants readiness."""

    def bounded(plan: list[Json], logs: Path) -> list[Json]:
        if owned_failure:
            plan[0]["argv"] = [sys.executable, "-u", "-c", "raise SystemExit(7)"]
        return list(execute(plan, logs))

    with patch.multiple(shipped, e=e, manifest=manifest, execute=bounded):
        return int(shipped.run(root, output, private, control=control))


def main(argv: list[str] | None = None) -> int:
    """Accept the registered date and keep private controls outside reader-visible results."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift", int(passed), 0)
        return int(not passed)
    if args.private_control and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private control output requires disk scratch outside the checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8589-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                control=args.private_control,
                owned_failure=args.owned_failure,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
