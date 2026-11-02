#!/usr/bin/env python3
"""Run the frozen Exp7811 dispatcher and publish its terminal receipt."""

from __future__ import annotations

print("[exp7811] start elapsed_s=0 completed_units=0", flush=True)

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable

from carnot import experiment_7811_v679_training_runtime as exp
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def materialize(scope: dict[str, Any], private: Path) -> dict[str, Any]:
    """Bind the frozen argv templates to one new private attempt root."""
    commands = []
    for item in scope["commands"]:
        commands.append(
            {**item, "argv": [arg.replace("{private}", str(private)) for arg in item["argv"]]}
        )
    return {"requirement": scope["requirement"], "private_root": str(private), "commands": commands}


def execute(command: dict[str, Any], durable: Path, private: Path) -> dict[str, Any]:
    """Supervise only this child, close its log, then seal exact bytes."""
    started = time.monotonic()
    name = command["name"]
    log = private / name / "child.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": f"{exp.ROOT / 'python'}:{exp.ROOT}",
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_CORE": "sysmon",
            "PYTHONUNBUFFERED": "1",
        }
    )
    exp.progress(started, name, "before_subprocess", 0)
    with log.open("wb") as stream:
        child = subprocess.Popen(
            command["argv"], cwd=exp.ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        deadline = started + command["timeout_s"]
        timed_out = False
        while True:
            try:
                exit_code = child.wait(timeout=max(0.01, min(30, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                exp.progress(started, name, "subprocess_outstanding", 0)
                if time.monotonic() >= deadline:
                    timed_out = True
                    child.terminate()
                    try:
                        exit_code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        exit_code = child.wait()
                    break
    sealed = exp.seal_log(log, durable, name)
    row = {
        "name": name,
        "argv": command["argv"],
        "classification": command["classification"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        **sealed,
    }
    if name.startswith("coverage_shard_"):
        data = Path(
            next(arg.split("=", 1)[1] for arg in command["argv"] if arg.startswith("--data-file="))
        )
        row["data_path"] = str(data)
        row["data_sha256"] = sha256_file(data) if data.is_file() else None
    if name == "coverage_report":
        match = re.search(rb"TOTAL\s+\d+\s+\d+\s+(\d+)%", exp.read_sealed_log(row))
        row["coverage_percent"] = int(match.group(1)) if match else None
    exp.progress(started, name, "after_subprocess", 1)
    return row


def dispatch(
    manifest: dict[str, Any],
    private: Path,
    durable: Path,
    executor: Callable[[dict[str, Any], Path], dict[str, Any]],
    before_command: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Send every declared command, including the terminal tail, in order."""
    receipts = []
    private.mkdir(parents=True, exist_ok=True)
    for command in manifest["commands"]:
        if before_command is not None:
            before_command(command, receipts)
        for arg in command["argv"]:
            if arg.startswith("--basetemp="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
            if arg.startswith("--data-file="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        (private / command["name"]).mkdir(parents=True, exist_ok=True)
        receipts.append(executor(command, durable))
    return receipts


def _apply_gate(record: dict[str, Any], gate: dict[str, Any]) -> None:
    """Carry failed owned checks into one terminal readiness decision."""
    record["gate_check_summary"] = [*record["gate_check_summary"], *gate["gate_check_summary"]]
    ready = int(not record["gate_check_summary"] and record["acceptance_gate_results"]["validity"])
    record["training_runtime_ready_score"] = ready
    record["online_runtime_ready_score"] = ready
    record["acceptance_gate_results"]["readiness"] = ready
    record["acceptance_gate_results"]["validity"] = bool(ready)
    record["verdict_class"] = "circular_positive" if ready else "disqualified"
    record["honest_verdict"] = (
        "complete_circular_positive_training_and_online_runtime"
        if ready
        else "complete_disqualified_required_validation"
    )


def _adversarial_flag(receipt: dict[str, Any]) -> bool:
    """Use the exact terminal reader output, never a presumed clean flag."""
    if receipt["exit_code"] != 0 or receipt["timed_out"]:
        return True
    try:
        return bool(json.loads(exp.read_sealed_log(receipt))["flagged_count"])
    except (ValueError, KeyError):
        return True


def run(
    date: str,
    private: Path | None = None,
    task: Any = None,
    executor: Callable[[dict[str, Any], Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure one attempt, dispatch its frozen commands, then publish atomically."""
    started = time.monotonic()
    task = task or exp
    executor = executor or execute
    private = private or Path(tempfile.mkdtemp(prefix="exp7811-attempt-", dir="/tmp"))
    private.mkdir(parents=True, exist_ok=True)
    durable = task.RAW / "attempts" / private.name
    durable.mkdir(parents=True, exist_ok=False)
    scope = json.loads(task.SCOPE.read_text())
    manifest = materialize(scope, private)
    manifest_path = durable / "validation_command_manifest.json"
    atomic_json(manifest_path, manifest)
    task.progress(started, "measure", "before_benchmark", 0)
    record = task.measure(date, durable / "measurement")
    task.progress(started, "measure", "after_benchmark", len(record["rows"]))
    record["validation_command_manifest_path"] = str(manifest_path)
    record["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    record["validation_receipts"]["frozen_affected_scope"] = scope
    record["historical_receipt_byte_audit"] = task.historical_receipt_byte_audit(task.ROOT)
    if record["verdict_class"] != "blocked":
        inputs = record.setdefault("reproducibility_inputs", {})
        inputs["code_closure"] = {
            path: sha256_file(task.ROOT / path)
            for path in (*scope["changed_modules_and_cli"], *scope["dependency_modules"])
        }
        inputs["scope"] = sha256_file(task.SCOPE)
        inputs["command_manifest"] = record["validation_command_manifest_sha256"]
        record["reproducibility_checksum"] = canonical_hash(inputs)
    if record["verdict_class"] == "blocked":
        record["duration_s"] = time.monotonic() - started
        atomic_json(task.OUTPUT, record)
        task.progress(started, "publish", "blocked_atomic_write", len(record["rows"]))
        return record
    measurement_failures = list(record["gate_check_summary"])
    candidate = private / "candidate.json"

    def before_command(command: dict[str, Any], prior_receipts: list[dict[str, Any]]) -> None:
        if command["name"] == "cold_replay":
            prefix = {**manifest, "commands": manifest["commands"][: len(prior_receipts)]}
            provisional = task.reduce_validation(scope, prefix, prior_receipts)
            _apply_gate(record, provisional)
            record["validation_receipts"]["commands"] = list(prior_receipts)
            atomic_json(candidate, record)
            task.progress(started, "candidate", "sealed_before_cold_replay", len(prior_receipts))

    receipts = dispatch(
        manifest,
        private,
        durable,
        lambda command, folder: executor(command, folder, private),
        before_command,
    )
    record["validation_receipts"]["commands"] = receipts
    record["coverage_shard_rows"] = [
        row for row in receipts if row["name"].startswith("coverage_shard_")
    ]
    record["observed_child_commands"] = [
        {key: row[key] for key in ("name", "argv", "classification")} for row in receipts
    ]
    record["repository_health"] = next(
        row for row in receipts if row["name"] == "repository_health"
    )
    record["validation_receipts"]["candidate_path"] = str(candidate)
    record["validation_receipts"]["candidate_sha256"] = sha256_file(candidate)
    record["flagged_adversarial"] = _adversarial_flag(
        next(row for row in receipts if row["name"] == "adversarial_verify")
    )
    gate = task.reduce_validation(scope, manifest, receipts)
    record["gate_check_summary"] = measurement_failures
    _apply_gate(record, gate)
    if record["flagged_adversarial"]:
        record["gate_check_summary"].append(
            task.failed_operand(
                "exp7811",
                str(candidate),
                "flagged_adversarial",
                False,
                True,
                sha256_file(candidate),
            )
        )
        record["training_runtime_ready_score"] = record["online_runtime_ready_score"] = 0
        record["acceptance_gate_results"].update({"validity": False, "readiness": 0})
        record["verdict_class"] = "disqualified"
        record["honest_verdict"] = "complete_disqualified_required_validation"
    record["duration_s"] = time.monotonic() - started
    record["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": record["duration_s"]
            - sum(span["duration_s"] for span in record["phase_spans"]),
            "completed_units": len(receipts),
        }
    )
    atomic_json(task.OUTPUT, record)
    task.progress(started, "publish", "atomic_write", len(record["rows"]))
    return record


def main(argv: list[str] | None = None) -> int:
    """Dispatch the task, a private miniature or one cold candidate reader."""
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
