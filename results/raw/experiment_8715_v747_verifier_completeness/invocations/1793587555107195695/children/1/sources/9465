#!/usr/bin/env python3
"""Run Exp7797 fixture qualification and supervised validation."""

from __future__ import annotations

print("[exp7797] start elapsed_s=0 completed_units=0", flush=True)

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_7797_v678_training_runtime as exp
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

PRIVATE = Path("/tmp/exp7797-private")


def _one_command(spec: dict[str, Any], *, child: bool = False) -> dict[str, Any]:
    """Supervise one frozen child and retain its durable exact log bytes."""
    timeout = 3600.0 if spec["name"] == "full_python_suite" else 900.0
    result = run_commands(
        exp.ROOT,
        [CommandSpec(spec["name"], tuple(spec["argv"]), "frozen_exp7797", timeout)],
        log_dir=exp.RAW / "validation_logs",
        extra_env={
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_CORE": "sysmon",
            "CARNOT_VALIDATION_CHILD": "1" if child else "0",
        },
        heartbeat_s=30.0,
    )[0]
    if spec["name"].startswith("coverage_shard_"):
        data = Path(
            next(arg.split("=", 1)[1] for arg in spec["argv"] if arg.startswith("--data-file="))
        )
        result["data_path"] = str(data)
        result["data_hash"] = sha256_file(data) if data.is_file() else None
    if spec["name"] == "coverage_report":
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result["output_tail"])
        result["coverage_percent"] = int(match.group(1)) if match else None
    return result


def _apply_gate(
    record: dict[str, Any],
    scope: dict[str, Any],
    receipts: list[dict[str, Any]],
    required: set[str],
) -> None:
    """Carry both validation and fixture failures into a single terminal class."""
    gate = exp.reduce_validation(scope, receipts, required)
    record["gate_check_summary"] = [*record["gate_check_summary"], *gate["gate_check_summary"]]
    ready = not record["gate_check_summary"] and record["acceptance_gate_results"]["validity"]
    record["training_runtime_ready_score"] = int(ready)
    record["online_runtime_ready_score"] = int(ready)
    record["acceptance_gate_results"]["readiness"] = int(ready)
    record["acceptance_gate_results"]["validity"] = bool(ready)
    record["verdict_class"] = "circular_positive" if ready else "disqualified"
    record["honest_verdict"] = (
        "complete_circular_positive_training_and_online_runtime"
        if ready
        else "complete_disqualified_required_validation"
    )


def _reader_flag(receipt: dict[str, Any]) -> bool:
    """Read the terminal adversarial boolean from the child JSON report."""
    if receipt["exit_code"] != 0:
        return True
    try:
        report = json.loads(Path(receipt["log_path"]).read_text())
        return bool(report["flagged_count"])
    except (OSError, ValueError, KeyError):
        return True


def run(date: str) -> dict[str, Any]:
    """Measure, validate the exact candidate, and publish one atomic verdict."""
    started = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    exp.progress(started, "measure", "before_benchmark", 0)
    record = exp.measure(date)
    exp.progress(started, "measure", "after_benchmark", len(record["rows"]))
    scope = json.loads(exp.SCOPE.read_text())
    candidate = PRIVATE / "candidate.json"
    if record["verdict_class"] == "blocked":
        atomic_json(candidate, record)
        specs = [
            item
            for item in scope["commands"]
            if item["name"] in {"adversarial_verify", "strict_row_consistency"}
        ]
        readers = [_one_command(spec) for spec in specs]
        record["validation_receipts"] = {
            "frozen_affected_scope": scope,
            "terminal_readers": readers,
        }
        record["flagged_adversarial"] = _reader_flag(readers[0])
        record["duration_s"] = time.monotonic() - started
        atomic_json(exp.OUTPUT, record)
        exp.progress(started, "publish", "blocked_atomic_write", len(record["rows"]))
        return record
    specs = [
        item
        for item in scope["commands"]
        if item["name"] not in {"adversarial_verify", "strict_row_consistency", "cold_replay"}
    ]
    receipts = []
    for index, spec in enumerate(specs):
        exp.progress(started, "validation", f"before_subprocess_{spec['name']}", index)
        receipts.append(_one_command(spec, child=spec["name"] == "task_e2e"))
        exp.progress(started, "validation", f"after_subprocess_{spec['name']}", index + 1)
    record["coverage_shard_rows"] = [
        row for row in receipts if row["name"].startswith("coverage_shard_")
    ]
    record["validation_receipts"]["commands"] = receipts
    required = {item["name"] for item in specs}
    _apply_gate(record, scope, receipts, required)
    atomic_json(candidate, record)
    tail_specs = [
        item
        for item in scope["commands"]
        if item["name"] in {"cold_replay", "adversarial_verify", "strict_row_consistency"}
    ]
    tail = []
    for index, spec in enumerate(tail_specs):
        exp.progress(started, "readers", f"before_subprocess_{spec['name']}", index)
        tail.append(_one_command(spec))
        exp.progress(started, "readers", f"after_subprocess_{spec['name']}", index + 1)
    record["validation_receipts"]["cold_replay"] = tail[0]
    record["validation_receipts"]["terminal_readers"] = tail[1:]
    record["flagged_adversarial"] = _reader_flag(tail[1])
    for row in tail:
        if row["exit_code"] or row.get("timed_out"):
            record["gate_check_summary"].append(
                exp.failed_operand(
                    "exp7797",
                    row["log_path"],
                    row["name"] + ".exit_code",
                    0,
                    row["exit_code"],
                    row["log_sha256"],
                )
            )
    if record["flagged_adversarial"]:
        record["gate_check_summary"].append(
            exp.failed_operand(
                "exp7797",
                tail[1]["log_path"],
                "flagged_adversarial",
                False,
                True,
                tail[1]["log_sha256"],
            )
        )
    if record["gate_check_summary"]:
        record["verdict_class"] = "disqualified"
        record["honest_verdict"] = "complete_disqualified_required_validation"
        record["training_runtime_ready_score"] = record["online_runtime_ready_score"] = 0
        record["acceptance_gate_results"]["validity"] = False
        record["acceptance_gate_results"]["readiness"] = 0
    record["duration_s"] = time.monotonic() - started
    record["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": record["duration_s"]
            - sum(span["duration_s"] for span in record["phase_spans"]),
            "completed_units": len(receipts) + len(tail),
        }
    )
    atomic_json(exp.OUTPUT, record)
    exp.progress(started, "publish", "atomic_write", len(record["rows"]))
    return record


def main(argv: list[str] | None = None) -> int:
    """Run the real task or reopen candidate rows in a fresh process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(exp.cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    if os.environ.get("CARNOT_VALIDATION_CHILD") == "1":
        child_raw = PRIVATE / "e2e_raw"
        measured = exp.measure(args.date, child_raw)
        if measured["verdict_class"] == "blocked":
            print(json.dumps({"valid": False, "reason": "external_precondition"}), flush=True)
            return 1
        candidate = PRIVATE / "e2e_candidate.json"
        atomic_json(candidate, measured)
        reduced = exp.cold_reduce(candidate)
        valid = bool(
            reduced["valid"]
            and measured["online_fixture"]["valid"]
            and all(row["reload_decision_equal"] for row in measured["fixture_training_rows"])
        )
        print(json.dumps({"valid": valid, "row_count": reduced["row_count"]}), flush=True)
        return 0 if valid else 1
    result = run(args.date)
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
