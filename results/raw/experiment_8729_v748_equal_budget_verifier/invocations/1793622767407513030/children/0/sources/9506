#!/usr/bin/env python3
"""Dispatch current Exp7825 qualification and publish its owned receipt."""

from __future__ import annotations

print("[exp7825] start elapsed_s=0 completed_units=0", flush=True)

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from carnot import experiment_7825_v680_training_runtime as exp
from carnot.reporting.current_work_receipt import atomic_json
from scripts.experiments import experiment_7811_v679_training_runtime as dispatcher

execute = dispatcher.execute


def run(date: str, private: Path | None = None) -> dict[str, Any]:
    """Use the qualified dispatcher with current explicit paths and executor."""
    private = private or Path(tempfile.mkdtemp(prefix="exp7825-attempt-", dir="/tmp"))
    return dispatcher.run(date, private, task=exp, executor=execute)


def main(argv: list[str] | None = None) -> int:
    """Select the owned run, fixture E2E or fresh cold candidate reduction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--mini-e2e", action="store_true")
    parser.add_argument("--private-root", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(exp.cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    if args.mini_e2e:
        if args.private_root is None:
            parser.error("--mini-e2e requires --private-root")
        measured = exp.measure(args.date, args.private_root / "measurement")
        if measured["verdict_class"] == "blocked":
            print(json.dumps({"valid": False, "reason": "external_precondition"}), flush=True)
            return 1
        candidate = args.private_root / "candidate.json"
        atomic_json(candidate, measured)
        replay = exp.cold_reduce(candidate)
        valid = bool(
            replay["valid"]
            and measured["online_fixture"]["valid"]
            and all(row["reload_decision_equal"] for row in measured["fixture_training_rows"])
        )
        print(json.dumps({"valid": valid, "row_count": replay["row_count"]}), flush=True)
        return 0 if valid else 1
    result = run(args.date, args.private_root)
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "honest_verdict": result["honest_verdict"],
                "output": str(exp.OUTPUT),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
