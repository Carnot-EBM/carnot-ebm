#!/usr/bin/env python3
"""Publish thirteen-task authority and methods (REQ-REPORT-7940-V689)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from carnot.reporting.v689_contract_methods import cold_replay  # noqa: E402
from carnot.reporting.v689_contract_validation import execute, publish  # noqa: E402

MODEL_SPECS: list[str] = []


def main() -> int:
    """Expose private input paths so historical evidence stays outside fixture writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260930", choices=["20260930"])
    parser.add_argument(
        "--design", type=Path, default=ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    )
    parser.add_argument("--staged", type=Path, default=ROOT / "research-roadmap-next.yaml")
    parser.add_argument("--active", type=Path, default=ROOT / "research-roadmap.yaml")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/experiment_7892_v685_source_boundary.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_7940_v689_contract_methods.json"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "results/raw/experiment_7940_v689_contract_methods/rows.json",
    )
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    args = parser.parse_args()
    print("[exp7940] start completed_units=0 elapsed_s=0", flush=True)
    if args.terminal_recheck:
        if args.output.resolve().is_relative_to(ROOT / "results"):
            parser.error("terminal fixture outputs must be private")
        args.raw = args.output.parent / "raw" / args.output.stem / "rows.json"
        with tempfile.TemporaryDirectory(prefix="exp7940-terminal-") as directory:
            publish(
                ROOT,
                json.loads(args.terminal_recheck.read_text()),
                args.output,
                args.raw,
                Path(directory),
            )
        return 0
    if args.cold_replay:
        passed = cold_replay(args.cold_replay, args.raw)
        print("cold_replay_passed" if passed else "cold_replay_mismatch", flush=True)
        return 0 if passed else 1
    if args.fixture_e2e and any(
        path.resolve().is_relative_to(ROOT / "results") for path in (args.output, args.raw)
    ):
        parser.error("fixture outputs must be private and outside results")
    with tempfile.TemporaryDirectory(prefix="exp7940-") as directory:
        execute(ROOT, args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
