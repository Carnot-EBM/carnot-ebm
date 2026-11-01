"""REQ-VERIFY-8693: retain bounded validation receipts and publish checked bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8693 as e
from carnot.reporting import hardware_evidence_runner_8651 as shared
from carnot.reporting import hardware_evidence_validation_8665 as validation

scan, terminal = shared.scan, shared.terminal


def validate(raw: Path) -> list[e.Json]:
    """Owned test/CLI coverage stays separate from unchanged consumer qualification."""
    with (
        patch.object(validation, "e", e),
        patch.object(validation, "SUITES", [e.TEST, *validation.SUITES[1:]]),
    ):
        return validation.validate(raw)


def run(
    root: Path, output: Path, private: Path, date: str, control: bool, owned_failure: bool
) -> int:
    """Existing supervision and atomic publication keep failed children visible."""
    with (
        patch.object(shared, "e", e),
        patch.object(shared, "validate", validate),
        patch.object(shared, "terminal", terminal),
    ):
        return shared.run(root, output, private, date, control, owned_failure)


def main(argv: list[str] | None = None) -> int:
    """Private controls cannot write inside the checkout, even through a symlink."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if len(args.date) != 8:
            return 2
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if (args.private_control or args.owned_failure) and (
        args.output.absolute().is_relative_to(e.ROOT)
        or args.output.resolve().is_relative_to(e.ROOT)
    ):
        parser.error("private output requires scratch outside checkout")
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8693-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                args.date,
                args.private_control,
                args.owned_failure,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
