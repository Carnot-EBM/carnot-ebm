#!/usr/bin/env python3
"""Reduce new receipt identities without models. Spec: REQ-REPORT-7962."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from carnot.reporting import arc_supervisor_v690_delta as task
from carnot.reporting.current_work_receipt import atomic_json


def main(argv: list[str] | None = None, scope: Any = None) -> int:
    """Keep fixture publication separate from live evidence and current authority."""
    scope = scope or task
    print(f"[exp{scope.EXPERIMENT_ID}] phase=start completed_units=0 elapsed_s=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=[scope.RUN_DATE], default=scope.RUN_DATE)
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--producer", type=Path, action="append")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    replay = args.cold_replay or args.terminal_recheck
    if replay:
        errors = scope.replay(json.loads(replay.read_text()))
        report = {"cold_replay_errors": errors}
        if args.output:
            atomic_json(args.output, report)
        print(json.dumps(report), flush=True)
        return int(bool(errors))
    with tempfile.TemporaryDirectory(
        prefix=f"carnot-exp{scope.EXPERIMENT_ID}-", dir="/tmp"
    ) as scratch:
        return reduce_or_execute(args, parser, scope, Path(scratch))


def reduce_or_execute(args: Any, parser: argparse.ArgumentParser, scope: Any, private: Path) -> int:
    """Keep mutable child fixtures inside the owned temporary directory."""
    if args.reduce_ledger:
        if not args.producer or args.output is None:
            parser.error("--reduce-ledger requires --producer and --output")
        value = getattr(scope, "scan", scope.previous.scan)(
            args.reduce_ledger,
            args.producer,
            dict(prior={}, inventory={}),
            private,
            current_date=args.date,
        )
        blocked = bool(value["scan_failures"])
        value.update(
            fixture_claim_scope="circular_positive",
            honest_verdict="complete_blocked_private_input"
            if blocked
            else "complete_circular_positive_private_reducer",
            verdict_class="blocked" if blocked else "circular_positive",
            gate_check_summary=value["scan_failures"],
        )
        atomic_json(args.output, value)
        print(json.dumps({"honest_verdict": value["honest_verdict"]}), flush=True)
        return int(blocked)
    return int(scope.execute(args.output or scope.OUTPUT, private))


if __name__ == "__main__":
    sys.exit(main())
