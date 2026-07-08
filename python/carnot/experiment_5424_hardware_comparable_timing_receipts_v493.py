#!/usr/bin/env python3
"""Exp5424: comparable CPU and board timing receipts without speedup claims.

Spec refs: REQ-HW-5424, SCENARIO-HW-5424.

This experiment selects the deterministic p-bit/QUBO workload prepared by
Exp5420, repeats the exact workload on CPU, and repeats the same workload on
PolarFire only when authenticated SSH is reachable. The receipt is intentionally
about comparability: matching workload hashes and result hashes make the timing
records usable evidence. They do not authorize a hardware speedup headline.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5424_hardware_comparable_timing_receipts_v493.json"
)

EXPERIMENT = 5424
EXPERIMENT_ID = "exp5424-hardware-comparable-timing-receipts-v493"
MILESTONE = "2026.07.493"
RUN_DATE = "20260708"
RANDOM_SEED = 5424
WORKLOAD_SEED = exp5420.RANDOM_SEED
SCHEMA = "carnot.experiment_5424.hardware_comparable_timing_receipts.v493"
SPEC_REFS = ("REQ-HW-5424", "SCENARIO-HW-5424")
INFERENCE_SUBSTRATE = "hardware_timing_with_cpu_reference"
REPEAT_TARGET = 3
TERMINAL_PREFIXES = ("complete:", "blocked:")

SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 30.0
LOCAL_TIMEOUT_S = 10.0

KV260_SSH_COMMAND = exp5420.KV260_SSH_COMMAND
POLARFIRE_STATUS_COMMAND = exp5420.POLARFIRE_STATUS_COMMAND
GATEMATE_DETECT_COMMAND = exp5420.GATEMATE_DETECT_COMMAND
HOST_STORAGE_MARKERS = exp5420.HOST_STORAGE_MARKERS
FORBIDDEN_COMMAND_TERMS = exp5420.FORBIDDEN_COMMAND_TERMS

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "hardware task must fail fast",
    "workload_hash": "same-workload comparison",
    "cpu_repeat_count": "CPU timing reliability",
    "board_repeat_count": "board timing reliability",
    "cpu_result_hash": "correctness comparison",
    "board_result_hash": "correctness comparison",
    "same_workload_hash_match": "no apples-to-oranges timing",
    "same_result_hash_match": "no invalid speedup",
    "polarfire_reachable": "board availability",
    "kv260_ssh_checked": "SSH-only discipline",
    "gatemate_diagnostic_checked": "physical/JTAG honesty",
    "timing_receipts": "reproducible evidence",
    "measurement_access_complete": "physical evidence boundary",
    "comparable_timing_receipts_ready": "capstone evidence",
    "hardware_speedup_claim": "no unsupported speedup",
    "inference_substrate": "explicit substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def sha256_text(text: str) -> str:
    """Return a stable SHA-256 digest for transcripts and commands."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping after deterministic serialization."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its own checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render commands using the same shell quoting discipline as Exp5420."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and convert missing tools/timeouts into receipts."""

    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=int(result.returncode),
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )
    except FileNotFoundError as exc:
        return CommandProbe(
            command=tuple(command),
            exit_code=127,
            stderr=str(exc),
            duration_s=round(time.perf_counter() - started, 6),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(
            command=tuple(command),
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )


def select_workload(root: str | Path = REPO_ROOT) -> JsonDict:
    """Select the deterministic Exp5420 workload used for comparable timing."""

    return exp5420.select_workload(root)


def polarfire_workload_command(workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the PolarFire command for the selected same-workload run."""

    return exp5420.polarfire_workload_command(workload)


def timing_distribution(values: Sequence[float]) -> JsonDict:
    """Summarize repeat timings without deriving a speedup ratio."""

    timings = [float(value) for value in values]
    if not timings:
        return {"count": 0}
    return {
        "count": len(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "mean_s": round(sum(timings) / len(timings), 9),
        "median_s": round(float(statistics.median(timings)), 9),
    }


def cpu_environment() -> JsonDict:
    """Capture CPU timing context so later readers can bound reproducibility."""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def cpu_timing_receipt(
    workload: Mapping[str, Any],
    *,
    repeat_count: int = REPEAT_TARGET,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Repeat the exact workload on CPU and record result hashes per repeat."""

    repeats: list[JsonDict] = []
    timings: list[float] = []
    for repeat_index in range(repeat_count):
        started = clock()
        exact = exp5420.exact_enumerate_workload(workload)
        elapsed = round(max(clock() - started, 0.0), 6)
        result_hash = exp5420.exact_result_hash(exact)
        timings.append(elapsed)
        repeats.append(
            {
                "repeat_index": repeat_index + 1,
                "seed": WORKLOAD_SEED,
                "workload_hash": workload["workload_hash"],
                "result_hash": result_hash,
                "wall_time_s": elapsed,
                "result_matches_selected": result_hash == workload["exact_result_hash"],
            }
        )
    return {
        "kind": "cpu_timing",
        "substrate": "cpu_exact_enumeration",
        "seed": WORKLOAD_SEED,
        "artifact_seed": RANDOM_SEED,
        "workload_hash": workload["workload_hash"],
        "result_hash": workload["exact_result_hash"],
        "repeat_count": repeat_count,
        "repeat_timings_s": timings,
        "timing_distribution": timing_distribution(timings),
        "environment": cpu_environment(),
        "repeats": repeats,
    }


def command_receipt(
    probe: CommandProbe,
    *,
    kind: str,
    timeout_s: float,
    outcome: str,
) -> JsonDict:
    """Build a compact command receipt while avoiding full raw board logs."""

    return exp5420.command_receipt(probe, kind=kind, timeout_s=timeout_s, outcome=outcome)


def _board_result_hash(valid_attempts: Sequence[Mapping[str, Any]]) -> str:
    hashes = [
        str(attempt["result_hash"])
        for attempt in valid_attempts
        if isinstance(attempt.get("result_hash"), str)
    ]
    return hashes[0] if hashes and len(set(hashes)) == 1 else ""


def collect_hardware_receipts(
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    command_runner: CommandRunner,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    """Collect SSH/JTAG measurement-access receipts and optional board timings."""

    command_receipts: list[JsonDict] = []

    kv_probe = command_runner(KV260_SSH_COMMAND, SSH_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            kv_probe,
            kind="kv260_ssh_only_reachability",
            timeout_s=SSH_TIMEOUT_S,
            outcome="reachable" if kv_probe.exit_code == 0 else "blocked",
        )
    )

    status_probe = command_runner(POLARFIRE_STATUS_COMMAND, SSH_TIMEOUT_S)
    polarfire_reachable = status_probe.exit_code == 0
    command_receipts.append(
        command_receipt(
            status_probe,
            kind="polarfire_ssh_reachability",
            timeout_s=SSH_TIMEOUT_S,
            outcome="reachable" if polarfire_reachable else "blocked",
        )
    )

    board_attempts: list[JsonDict] = []
    board_command = polarfire_workload_command(workload)
    if polarfire_reachable:
        for repeat_index in range(repeat_count):
            probe = command_runner(board_command, SSH_TIMEOUT_S)
            receipt, parse_error = exp5420.parse_board_workload_stdout(probe.stdout, workload)
            result_hash = (
                receipt.get("exact_result_hash") if isinstance(receipt, Mapping) else None
            )
            wall_time = receipt.get("wall_time_s") if isinstance(receipt, Mapping) else None
            valid = probe.exit_code == 0 and receipt is not None and parse_error is None
            board_attempts.append(
                {
                    "repeat_index": repeat_index + 1,
                    "valid": valid,
                    "parse_error": parse_error,
                    "workload_hash": receipt.get("workload_hash")
                    if isinstance(receipt, Mapping)
                    else None,
                    "result_hash": result_hash,
                    "wall_time_s": wall_time,
                    "receipt": receipt,
                }
            )
            command_receipts.append(
                command_receipt(
                    probe,
                    kind=f"polarfire_same_workload_timing_repeat_{repeat_index + 1}",
                    timeout_s=SSH_TIMEOUT_S,
                    outcome="valid_repeat" if valid else "invalid_repeat",
                )
            )

    gate_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    gate_detected = "gatemate" in gate_probe.combined_output.lower()
    command_receipts.append(
        command_receipt(
            gate_probe,
            kind="gatemate_non_destructive_dirtyjtag_detect",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome="detected" if gate_detected else "blocked",
        )
    )

    valid_attempts = [attempt for attempt in board_attempts if attempt["valid"] is True]
    board_timings = [
        float(attempt["wall_time_s"])
        for attempt in valid_attempts
        if isinstance(attempt.get("wall_time_s"), int | float)
    ]
    board_receipt = {
        "kind": "polarfire_board_timing",
        "substrate": "polarfire_board_local_python",
        "reachable": polarfire_reachable,
        "seed": WORKLOAD_SEED,
        "workload_hash": workload["workload_hash"] if polarfire_reachable else "",
        "result_hash": _board_result_hash(valid_attempts),
        "repeat_target": repeat_count,
        "repeat_count": len(valid_attempts),
        "invalid_repeat_count": len(board_attempts) - len(valid_attempts),
        "repeat_timings_s": board_timings,
        "timing_distribution": timing_distribution(board_timings),
        "attempts": board_attempts,
    }
    summary = {
        "kv260_ssh_checked": True,
        "kv260_ssh_reachable": kv_probe.exit_code == 0,
        "polarfire_reachable": polarfire_reachable,
        "gatemate_diagnostic_checked": True,
        "gatemate_reachable": gate_detected,
        "board_repeat_count": len(valid_attempts),
        "board_result_hash": board_receipt["result_hash"],
        "blocked_hardware_precondition": None
        if polarfire_reachable
        else {
            "resource": "polarfire_ssh",
            "command": command_to_string(POLARFIRE_STATUS_COMMAND),
            "exit_code": int(status_probe.exit_code),
            "stderr_excerpt": status_probe.stderr.strip()[:240],
        },
    }
    return summary, command_receipts, board_receipt


def default_tests_run() -> list[JsonDict]:
    """Keep CLI artifacts valid before external test results are attached."""

    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    return [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]


def readiness_blockers(
    *,
    polarfire_reachable: bool,
    cpu_repeat_count: int,
    board_repeat_count: int,
    same_workload_hash_match: bool,
    same_result_hash_match: bool,
) -> list[str]:
    """Explain why timing receipts are not yet comparable."""

    blockers: list[str] = []
    if not polarfire_reachable:
        blockers.append("polarfire_unreachable")
    if cpu_repeat_count < REPEAT_TARGET:
        blockers.append("cpu_repeat_count_below_threshold")
    if polarfire_reachable and board_repeat_count < REPEAT_TARGET:
        blockers.append("board_repeat_count_below_threshold")
    if polarfire_reachable and not same_workload_hash_match:
        blockers.append("same_workload_hash_mismatch")
    if polarfire_reachable and not same_result_hash_match:
        blockers.append("same_result_hash_mismatch")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """State terminal evidence status while explicitly refusing speedup."""

    if ready:
        return (
            "complete: comparable CPU and PolarFire timing receipts are "
            "hash-matched; hardware_speedup_claim=false"
        )
    joined = ",".join(blockers) if blockers else "timing_receipts_not_comparable"
    return f"blocked: {joined}; hardware_speedup_claim=false"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp5424 artifact from repeated CPU and optional board timings."""

    started = clock()
    workload = select_workload(root)
    cpu_receipt = cpu_timing_receipt(workload, repeat_count=REPEAT_TARGET, clock=clock)
    hardware_summary, command_receipts, board_receipt = collect_hardware_receipts(
        workload=workload,
        repeat_count=int(cpu_receipt["repeat_count"]),
        command_runner=command_runner,
    )

    cpu_repeat_count = int(cpu_receipt["repeat_count"])
    board_repeat_count = int(hardware_summary["board_repeat_count"])
    cpu_result_hash = str(cpu_receipt["result_hash"])
    board_result_hash = str(hardware_summary["board_result_hash"])
    polarfire_reachable = bool(hardware_summary["polarfire_reachable"])
    same_workload_hash_match = (
        polarfire_reachable
        and board_repeat_count >= REPEAT_TARGET
        and all(
            attempt.get("workload_hash") == workload["workload_hash"]
            for attempt in board_receipt["attempts"]
            if attempt.get("valid") is True
        )
        and board_receipt["invalid_repeat_count"] == 0
    )
    same_result_hash_match = (
        polarfire_reachable
        and board_repeat_count >= REPEAT_TARGET
        and board_result_hash == cpu_result_hash
        and board_receipt["invalid_repeat_count"] == 0
    )
    blockers = readiness_blockers(
        polarfire_reachable=polarfire_reachable,
        cpu_repeat_count=cpu_repeat_count,
        board_repeat_count=board_repeat_count,
        same_workload_hash_match=same_workload_hash_match,
        same_result_hash_match=same_result_hash_match,
    )
    ready = (
        cpu_repeat_count >= REPEAT_TARGET
        and board_repeat_count >= REPEAT_TARGET
        and same_workload_hash_match
        and same_result_hash_match
        and not blockers
    )
    timing_comparison = {
        "comparison_performed": ready,
        "same_workload_hash_match": same_workload_hash_match,
        "same_result_hash_match": same_result_hash_match,
        "cpu_timing_distribution": cpu_receipt["timing_distribution"],
        "board_timing_distribution": board_receipt["timing_distribution"],
        "speedup_reported": False,
        "speedup_ratio": None,
        "comparison_boundary": (
            "Timing distributions are comparable evidence only; no speedup ratio is claimed."
        ),
    }

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "workload_seed": WORKLOAD_SEED,
        "duration_s": round(max(clock() - started, 0.0), 6),
        "commit": commit,
        "preconditions_checked": True,
        "selected_workload": workload,
        "workload_hash": workload["workload_hash"],
        "cpu_repeat_count": cpu_repeat_count,
        "board_repeat_count": board_repeat_count,
        "cpu_result_hash": cpu_result_hash,
        "board_result_hash": board_result_hash,
        "same_workload_hash_match": same_workload_hash_match,
        "same_result_hash_match": same_result_hash_match,
        "polarfire_reachable": polarfire_reachable,
        "kv260_ssh_checked": bool(hardware_summary["kv260_ssh_checked"]),
        "gatemate_diagnostic_checked": bool(hardware_summary["gatemate_diagnostic_checked"]),
        "timing_receipts": [cpu_receipt, board_receipt],
        "command_receipts": command_receipts,
        "measurement_access": {
            "cpu_timing_complete": cpu_repeat_count >= REPEAT_TARGET,
            "board_timing_complete": board_repeat_count >= REPEAT_TARGET,
            "blocked_hardware_precondition": hardware_summary["blocked_hardware_precondition"],
            "missing_physical_records_recoverable_after_fact": False,
        },
        "measurement_access_complete": ready,
        "comparable_timing_receipts_ready": ready,
        "timing_comparison": timing_comparison,
        "blocked_hardware_precondition": hardware_summary["blocked_hardware_precondition"],
        "hardware_speedup_claim": False,
        "claim_refusal": (
            "No hardware speedup is claimed; this artifact only records comparable "
            "timing distributions when workload and result hashes match."
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "readiness_blockers": blockers,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalize_tests(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _artifact_mentions_host_storage(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return any(marker in encoded for marker in HOST_STORAGE_MARKERS)


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for index, item in enumerate(tests_run):
        _require(isinstance(item, Mapping), f"tests_run[{index}]")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run command")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run outcome")


def _validate_command_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("command_receipts")
    _require(isinstance(receipts, list) and receipts, "command_receipts")
    kv260_count = 0
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipt")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command_receipt command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command hash")
        _require(not _command_is_destructive(command), "destructive command")
        if receipt.get("kind") == "kv260_ssh_only_reachability":
            kv260_count += 1
            _require(
                command == command_to_string(KV260_SSH_COMMAND),
                "KV260 command must be exact SSH-only reachability precondition",
            )
    _require(kv260_count == 1, "exactly one KV260 SSH receipt required")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on mismatched hashes, unsafe probes, or speedup claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("preconditions_checked") is True, "preconditions_checked")
    _require(isinstance(artifact.get("workload_hash"), str), "workload_hash")
    _require(isinstance(artifact.get("cpu_repeat_count"), int), "cpu_repeat_count")
    _require(isinstance(artifact.get("board_repeat_count"), int), "board_repeat_count")
    _require(isinstance(artifact.get("cpu_result_hash"), str), "cpu_result_hash")
    _require(isinstance(artifact.get("board_result_hash"), str), "board_result_hash")
    _require(isinstance(artifact.get("same_workload_hash_match"), bool), "same_workload_hash_match")
    _require(isinstance(artifact.get("same_result_hash_match"), bool), "same_result_hash_match")
    _require(isinstance(artifact.get("polarfire_reachable"), bool), "polarfire_reachable")
    _require(isinstance(artifact.get("kv260_ssh_checked"), bool), "kv260_ssh_checked")
    _require(
        isinstance(artifact.get("gatemate_diagnostic_checked"), bool),
        "gatemate_diagnostic_checked",
    )
    _require(isinstance(artifact.get("timing_receipts"), list), "timing_receipts")
    _require(isinstance(artifact.get("measurement_access_complete"), bool), "measurement_access")
    _require(
        isinstance(artifact.get("comparable_timing_receipts_ready"), bool),
        "comparable_timing_receipts_ready",
    )
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict speedup boundary")
    _require(not _artifact_mentions_host_storage(artifact), "host block-device evidence present")
    _validate_tests_run(artifact.get("tests_run"))
    _validate_command_receipts(artifact)

    _require(len(artifact["workload_hash"]) == 64, "workload_hash")
    _require(artifact["cpu_repeat_count"] >= REPEAT_TARGET, "cpu_repeat_count")
    _require(len(artifact["cpu_result_hash"]) == 64, "cpu_result_hash")
    if artifact.get("same_workload_hash_match") is True:
        _require(artifact["board_repeat_count"] >= REPEAT_TARGET, "board_repeat_count")
    if artifact.get("same_result_hash_match") is True:
        _require(artifact["board_result_hash"] == artifact["cpu_result_hash"], "result hash match")
    ready = artifact.get("comparable_timing_receipts_ready") is True
    _require(artifact.get("measurement_access_complete") is ready, "measurement access boundary")
    comparison = artifact.get("timing_comparison")
    _require(isinstance(comparison, Mapping), "timing_comparison")
    _require(comparison.get("speedup_reported") is False, "speedup_reported")
    _require(comparison.get("speedup_ratio") is None, "speedup_ratio")
    _require(comparison.get("comparison_performed") is ready, "comparison gate")
    if ready:
        _require(artifact.get("same_workload_hash_match") is True, "same_workload_hash_match")
        _require(artifact.get("same_result_hash_match") is True, "same_result_hash_match")
        _require(artifact.get("board_repeat_count") >= REPEAT_TARGET, "board_repeat_count")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
    if artifact.get("polarfire_reachable") is False:
        blocker = artifact.get("blocked_hardware_precondition")
        _require(isinstance(blocker, Mapping), "blocked_hardware_precondition")
        _require(blocker.get("command") == command_to_string(POLARFIRE_STATUS_COMMAND), "blocked command")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def write_output(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write stable JSON after validating the Exp5424 artifact contract."""

    validate_artifact(artifact)
    path = Path(repo_root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    workload_root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Run Exp5424 and write the requested deliverable JSON."""

    artifact = build_artifact(
        root=workload_root,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit,
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--tests-command", default="")
    parser.add_argument("--tests-outcome", default="passed")
    args = parser.parse_args(argv)
    tests_run = None
    if args.tests_command:
        tests_run = [{"command": args.tests_command, "outcome": args.tests_outcome}]
    path = run_experiment(
        repo_root=REPO_ROOT,
        workload_root=REPO_ROOT,
        run_date=args.date,
        commit=args.commit,
        tests_run=tests_run,
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(f"artifact: {path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main(sys.argv[1:]))
