#!/usr/bin/env python3
"""REQ-VERIFY-8385: use private disk scratch and bounded operation-map execution."""

import argparse
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import board_operation_evidence_8385 as e  # noqa: E402
from carnot.reporting import board_operation_runner_8385 as r  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []
SCRATCH = Path.home() / ".cache/carnot-exp8385-private"


def main(argv: list[str] | None = None) -> int:
    """Keep children and coverage on private disk without loading a current model."""
    e.progress("start_no_model_load_zero_current_LLM_calls")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.private_e2e and args.output.resolve().is_relative_to(ROOT):
        parser.error("private E2E output requires scratch outside the checkout")
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with TemporaryDirectory(prefix="invocation-", dir=SCRATCH) as scratch:
            return r.run(args.root, args.output.absolute(), Path(scratch), control=args.private_e2e)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


if __name__ == "__main__":
    raise SystemExit(main())
