"""REQ-VERIFY-8600: reuse supervised validation and exact-byte publication consumers."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_hardware_boundary_8600 as e
from carnot.reporting import resident_hardware_runner_8589 as previous

Json = dict[str, Any]
SUITES = [e.TEST, *previous.SUITES[1:]]
_manifest = previous.manifest


def manifest(private: Path) -> list[Json]:
    """Keep original suite authority and collect exact node IDs before disjoint children."""
    with patch.multiple(previous, e=e, SUITES=SUITES):
        return list(_manifest(private))


def main(argv: list[str] | None = None) -> int:
    """Dates bind invocation identity; private controls cannot publish into the checkout."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(args.date) != 8:
            return 2
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private control output requires disk scratch outside the checkout")
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    measured = e.measure
    try:
        with TemporaryDirectory(prefix="exp8600-private-", dir="/var/tmp") as directory:
            with (
                patch.multiple(previous, e=e, manifest=manifest),
                patch.object(
                    e, "measure", lambda root, raw, private: measured(root, raw, private, args.date)
                ),
            ):
                return int(
                    previous.run(
                        args.root,
                        args.output.absolute(),
                        Path(directory),
                        control=args.private_control,
                        owned_failure=args.owned_failure,
                    )
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
