#!/usr/bin/env python3
"""Exp5434: gated p-bit/active-constraint timing variance receipts.

Spec refs: REQ-HW-5434, SCENARIO-HW-5434.

This experiment uses Exp5433's active-constraint diversity gate as the upstream
permission check, selects one deterministic active-constraint workload, and
measures repeated CPU timing against optional PolarFire board timing. The output
is a variance receipt, not an acceleration headline: CPU and board numbers are
compared only when workload and result hashes match, and every valid artifact
keeps ``hardware_speedup_claim=false``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420
from carnot import experiment_5433_active_constraint_diversity_lns_v494 as exp5433


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5434_pbit_polarfire_timing_variance_v494.json"
)
UPSTREAM_DIVERSITY_RELATIVE_PATH = exp5433.RESULT_RELATIVE_PATH

EXPERIMENT = 5434
EXPERIMENT_ID = "exp5434-pbit-polarfire-timing-variance-v494"
MILESTONE = "2026.07.494"
RUN_DATE = "20260708"
RANDOM_SEED = 5434
SCHEMA = "carnot.experiment_5434.pbit_polarfire_timing_variance.v494"
SPEC_REFS = ("REQ-HW-5434", "SCENARIO-HW-5434")
INFERENCE_SUBSTRATE = "hardware_timing_with_cpu_reference"
REPEAT_TARGET = 10
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
    "gated_upstream_ready": "structured gate provenance",
    "workload_hash": "same-workload comparison",
    "cpu_repeat_count": "timing reliability",
    "board_repeat_count": "board reliability",
    "cpu_result_hash": "correctness comparison",
    "board_result_hash": "correctness comparison",
    "same_workload_hash_match": "no apples-to-oranges timing",
    "same_result_hash_match": "no invalid timing comparison",
    "cpu_timing_variance": "timing distribution",
    "board_timing_variance": "timing distribution",
    "polarfire_reachable": "board availability",
    "kv260_ssh_checked": "SSH-only discipline",
    "gatemate_diagnostic_checked": "physical/JTAG honesty",
    "measurement_access_complete": "physical evidence boundary",
    "hardware_speedup_claim": "no unsupported speedup",
    "timing_variance_receipts_ready": "capstone evidence",
    "inference_substrate": "explicit substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON deterministically so hashes track content, not formatting."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for command strings and compact receipts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping after deterministic serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while ignoring its own checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render a command with the same shell quoting discipline as Exp5420."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and preserve missing-tool failures as receipts."""

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


def load_upstream_gate(root: str | Path = REPO_ROOT) -> JsonDict:
    """Read the Exp5433 gate that authorizes active-constraint timing work."""

    path = Path(root) / UPSTREAM_DIVERSITY_RELATIVE_PATH
    record: JsonDict = {
        "artifact_path": str(UPSTREAM_DIVERSITY_RELATIVE_PATH),
        "gate_field": "active_constraint_diversity_ready",
        "gate_value": False,
        "source_status": "missing",
        "source_experiment_id": "",
    }
    if not path.exists():  # pragma: no cover - repository fixture is required for this task.
        return record
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive artifact handling.
        record["source_status"] = "unreadable"
        return record
    record["gate_value"] = bool(source.get("active_constraint_diversity_ready"))
    record["source_status"] = str(source.get("status", "unknown"))
    record["source_experiment_id"] = str(source.get("experiment_id", ""))
    record["readiness_blockers"] = list(source.get("readiness_blockers", []))
    return record


def _load_upstream_diversity_artifact(root: str | Path = REPO_ROOT) -> JsonDict:
    path = Path(root) / UPSTREAM_DIVERSITY_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_from_constraint(constraint: str) -> list[str]:
    before, after = constraint.split("->", 1)
    return [before, after]


def precedence_energy(workload: Mapping[str, Any], sequence: Sequence[str]) -> int:
    """Score one p-bit precedence assignment with a fixed violation penalty."""

    actions = list(workload["actions"])
    precedence = [tuple(edge) for edge in workload["precedence"]]
    if len(sequence) != len(actions) or set(sequence) != set(actions):
        return 10 * len(precedence) + len(actions)
    positions = {action: index for index, action in enumerate(sequence)}
    return 10 * sum(int(positions[before] > positions[after]) for before, after in precedence)


def exact_enumerate_workload(workload: Mapping[str, Any]) -> JsonDict:
    """Enumerate every assignment ordering so the selected workload is exact."""

    best_sequence: tuple[str, ...] | None = None
    best_energy: int | None = None
    valid_count = 0
    enumerated = 0
    for permutation in itertools.permutations(tuple(workload["actions"])):
        enumerated += 1
        energy = precedence_energy(workload, permutation)
        if energy == 0:
            valid_count += 1
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_sequence = tuple(permutation)
    if best_sequence is None or best_energy is None:  # pragma: no cover - permutations emits for empty input.
        raise ValueError("workload has no permutations")
    return {
        "exact_min_energy": int(best_energy),
        "exact_best_sequence": list(best_sequence),
        "enumerated_permutation_count": enumerated,
        "exact_valid_permutation_count": valid_count,
    }


def result_hash(exact_result: Mapping[str, Any]) -> str:
    """Hash exact solver output independently from timing receipts."""

    return sha256_json(dict(exact_result))


def select_workload(root: str | Path = REPO_ROOT) -> JsonDict:
    """Select one deterministic Exp5433 p-bit/active-constraint workload."""

    upstream = _load_upstream_diversity_artifact(root)
    _require(
        upstream.get("active_constraint_diversity_ready") is True,
        "active_constraint_diversity_ready",
    )
    candidates = [
        row
        for row in upstream["row_records"]
        if row.get("hint_mode") == "lns_guided_hint"
        and row.get("final_valid") is True
        and row.get("exact_min_energy", 0) == 0
    ]
    if not candidates:
        candidates = [
            row
            for row in upstream["row_records"]
            if row.get("hint_mode") == "lns_guided_hint" and row.get("final_valid") is True
        ]
    _require(bool(candidates), "no valid Exp5433 lns_guided_hint row")
    row = sorted(candidates, key=lambda item: str(item["fixture_id"]))[0]
    active_constraint_ids = list(row["known_active_constraints"])
    workload: JsonDict = {
        "fixture_id": row["fixture_id"],
        "source_artifact": str(UPSTREAM_DIVERSITY_RELATIVE_PATH),
        "source_experiment_id": upstream["experiment_id"],
        "source_row_hint_mode": row["hint_mode"],
        "source_diversity_descriptor_checksum": upstream["diversity_descriptor_checksum"],
        "actions": list(row["expected_sequence"]),
        "precedence": [_edge_from_constraint(edge) for edge in active_constraint_ids],
        "active_constraint_ids": active_constraint_ids,
        "conflict_front": list(row["known_conflict_front"]),
        "lns_subproblem": list(row["known_lns_subproblem"]),
        "active_tail": list(row["known_active_tail"]),
        "frozen_variables": list(row["known_frozen_variables"]),
        "pbit_encoding": {
            "energy_model": "precedence_penalty_qubo",
            "spin_count": len(row["expected_sequence"]),
            "violation_penalty": 10,
        },
        "seed": RANDOM_SEED,
    }
    exact = exact_enumerate_workload(workload)
    workload["exact_result"] = exact
    workload["result_hash"] = result_hash(exact)
    workload["exact_solver_validity"] = (
        exact["exact_min_energy"] == 0
        and exact["exact_best_sequence"] == list(row["final_sequence"])
        and row.get("solver_authoritative") is True
    )
    workload["workload_hash"] = sha256_json(
        {
            "fixture_id": workload["fixture_id"],
            "source_artifact": workload["source_artifact"],
            "source_diversity_descriptor_checksum": workload[
                "source_diversity_descriptor_checksum"
            ],
            "actions": workload["actions"],
            "precedence": workload["precedence"],
            "active_constraint_ids": workload["active_constraint_ids"],
            "active_tail": workload["active_tail"],
            "frozen_variables": workload["frozen_variables"],
            "exact_result": workload["exact_result"],
            "seed": RANDOM_SEED,
        }
    )
    _require(workload["exact_solver_validity"] is True, "exact_solver_validity")
    return workload


def timing_distribution(values: Sequence[float]) -> JsonDict:
    """Summarize repeat timings with mean, median, p95, and variance."""

    timings = [float(value) for value in values]
    if not timings:
        return {
            "count": 0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "p95_s": 0.0,
            "variance_s2": 0.0,
        }
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    variance = statistics.pvariance(timings) if len(timings) > 1 else 0.0
    return {
        "count": len(timings),
        "min_s": round(min(timings), 9),
        "max_s": round(max(timings), 9),
        "mean_s": round(sum(timings) / len(timings), 9),
        "median_s": round(float(statistics.median(timings)), 9),
        "p95_s": round(float(ordered[p95_index]), 9),
        "variance_s2": round(float(variance), 12),
    }


def cpu_environment() -> JsonDict:
    """Capture CPU context so repeat timings are interpretable later."""

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
    """Repeat the exact active-constraint workload on CPU."""

    repeats: list[JsonDict] = []
    timings: list[float] = []
    for repeat_index in range(repeat_count):
        started = clock()
        exact = exact_enumerate_workload(workload)
        elapsed = round(max(clock() - started, 0.0), 9)
        repeat_result_hash = result_hash(exact)
        timings.append(elapsed)
        repeats.append(
            {
                "repeat_index": repeat_index + 1,
                "seed": RANDOM_SEED,
                "workload_hash": workload["workload_hash"],
                "result_hash": repeat_result_hash,
                "wall_time_s": elapsed,
                "result_matches_selected": repeat_result_hash == workload["result_hash"],
                "exact_solver_validity": exact["exact_min_energy"] == 0,
            }
        )
    distribution = timing_distribution(timings)
    return {
        "kind": "cpu_timing",
        "substrate": "cpu_exact_active_constraint_enumeration",
        "seed": RANDOM_SEED,
        "workload_hash": workload["workload_hash"],
        "result_hash": workload["result_hash"],
        "repeat_count": repeat_count,
        "repeat_timings_s": timings,
        "timing_distribution": distribution,
        "environment": cpu_environment(),
        "repeats": repeats,
    }


def _remote_workload_source(workload: Mapping[str, Any]) -> str:
    payload = {
        "actions": workload["actions"],
        "precedence": workload["precedence"],
        "active_constraint_ids": workload["active_constraint_ids"],
        "seed": RANDOM_SEED,
        "workload_hash": workload["workload_hash"],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return "\n".join(
        [
            "import hashlib,itertools,json,time",
            f"workload=json.loads({payload_json!r})",
            "started=time.perf_counter()",
            "def energy(seq):",
            "    positions={action:index for index,action in enumerate(seq)}",
            "    return 10*sum(int(positions[before]>positions[after]) for before,after in workload['precedence'])",
            "best_seq=None; best_energy=None; valid_count=0; enumerated=0",
            "for perm in itertools.permutations(workload['actions']):",
            "    enumerated+=1",
            "    e=energy(perm)",
            "    valid_count+=int(e==0)",
            "    if best_energy is None or e<best_energy:",
            "        best_energy=e; best_seq=list(perm)",
            "exact={'exact_min_energy':int(best_energy),'exact_best_sequence':best_seq,'enumerated_permutation_count':enumerated,'exact_valid_permutation_count':valid_count}",
            "encoded=json.dumps(exact,sort_keys=True,separators=(',',':'))",
            "receipt={'board_local':True,'workload_hash':workload['workload_hash'],'seed':workload['seed'],'active_constraint_ids':workload['active_constraint_ids'],'exact_min_energy':exact['exact_min_energy'],'exact_best_sequence':exact['exact_best_sequence'],'result_hash':hashlib.sha256(encoded.encode()).hexdigest(),'wall_time_s':round(time.perf_counter()-started,9)}",
            "print(json.dumps(receipt,sort_keys=True))",
        ]
    )


def polarfire_workload_command(workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the PolarFire SSH command for the selected exact workload."""

    remote = "python3 - <<'PY'\n" + _remote_workload_source(workload) + "\nPY"
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "polarfire",
        remote,
    )


def parse_board_workload_stdout(
    stdout: str,
    workload: Mapping[str, Any],
) -> tuple[JsonDict | None, str | None]:
    """Parse and validate one PolarFire same-workload timing receipt."""

    parsed: Any | None = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            break
    if not isinstance(parsed, Mapping):
        return None, "workload stdout is not valid JSON"

    receipt = dict(parsed)
    errors: list[str] = []
    if receipt.get("workload_hash") != workload["workload_hash"]:
        errors.append("workload_hash mismatch")
    if receipt.get("seed") != RANDOM_SEED:
        errors.append("seed mismatch")
    if receipt.get("active_constraint_ids") != workload["active_constraint_ids"]:
        errors.append("active_constraint_ids mismatch")
    exact = workload["exact_result"]
    if receipt.get("exact_min_energy") != exact["exact_min_energy"]:
        errors.append("exact_min_energy mismatch")
    if receipt.get("exact_best_sequence") != exact["exact_best_sequence"]:
        errors.append("exact_best_sequence mismatch")
    if receipt.get("result_hash") != workload["result_hash"]:
        errors.append("result_hash mismatch")
    if receipt.get("board_local") is not True:
        errors.append("board_local missing")
    if not isinstance(receipt.get("wall_time_s"), int | float) or receipt["wall_time_s"] < 0:
        errors.append("wall_time_s invalid")
    return receipt, "; ".join(errors) if errors else None


def command_receipt(
    probe: CommandProbe,
    *,
    kind: str,
    timeout_s: float,
    outcome: str,
) -> JsonDict:
    """Build a compact command receipt without storing full hardware logs."""

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
    """Collect allowed SSH/JTAG receipts and optional PolarFire timing."""

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
            receipt, parse_error = parse_board_workload_stdout(probe.stdout, workload)
            result = receipt.get("result_hash") if isinstance(receipt, Mapping) else None
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
                    "result_hash": result,
                    "wall_time_s": wall_time,
                    "receipt": receipt,
                }
            )
            command_receipts.append(
                command_receipt(
                    probe,
                    kind=f"polarfire_active_constraint_timing_repeat_{repeat_index + 1}",
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
        "seed": RANDOM_SEED,
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
    gated_upstream_ready: bool,
    polarfire_reachable: bool,
    cpu_repeat_count: int,
    board_repeat_count: int,
    same_workload_hash_match: bool,
    same_result_hash_match: bool,
) -> list[str]:
    """Explain why timing variance receipts are not complete."""

    blockers: list[str] = []
    if not gated_upstream_ready:
        blockers.append("active_constraint_diversity_not_ready")
    if cpu_repeat_count < REPEAT_TARGET:
        blockers.append("cpu_repeat_count_below_threshold")
    if not polarfire_reachable:
        blockers.append("polarfire_unreachable")
    if polarfire_reachable and board_repeat_count < REPEAT_TARGET:
        blockers.append("board_repeat_count_below_threshold")
    if polarfire_reachable and not same_workload_hash_match:
        blockers.append("same_workload_hash_mismatch")
    if polarfire_reachable and not same_result_hash_match:
        blockers.append("same_result_hash_mismatch")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """State terminal evidence status while refusing speedup claims."""

    if ready:
        return (
            "complete: gated CPU and PolarFire timing variance receipts are "
            "hash-matched; hardware_speedup_claim=false"
        )
    joined = ",".join(blockers) if blockers else "timing_variance_receipts_not_comparable"
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
    """Build the Exp5434 artifact from repeated CPU and optional board timings."""

    started = clock()
    upstream_gate = load_upstream_gate(root)
    gated_upstream_ready = bool(upstream_gate["gate_value"])
    _require(gated_upstream_ready, "active_constraint_diversity_ready")
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
    invalid_board_repeats = int(board_receipt["invalid_repeat_count"])
    same_workload_hash_match = (
        polarfire_reachable
        and board_repeat_count >= REPEAT_TARGET
        and invalid_board_repeats == 0
        and all(
            attempt.get("workload_hash") == workload["workload_hash"]
            for attempt in board_receipt["attempts"]
            if attempt.get("valid") is True
        )
    )
    same_result_hash_match = (
        polarfire_reachable
        and board_repeat_count >= REPEAT_TARGET
        and invalid_board_repeats == 0
        and board_result_hash == cpu_result_hash
    )
    blockers = readiness_blockers(
        gated_upstream_ready=gated_upstream_ready,
        polarfire_reachable=polarfire_reachable,
        cpu_repeat_count=cpu_repeat_count,
        board_repeat_count=board_repeat_count,
        same_workload_hash_match=same_workload_hash_match,
        same_result_hash_match=same_result_hash_match,
    )
    ready = (
        gated_upstream_ready
        and cpu_repeat_count >= REPEAT_TARGET
        and board_repeat_count >= REPEAT_TARGET
        and same_workload_hash_match
        and same_result_hash_match
        and not blockers
    )
    cpu_distribution = cpu_receipt["timing_distribution"]
    board_distribution = board_receipt["timing_distribution"]
    cpu_mean = float(cpu_distribution["mean_s"])
    board_mean = float(board_distribution["mean_s"])
    board_cpu_ratio = round(board_mean / cpu_mean, 9) if ready and cpu_mean > 0.0 else None
    timing_comparison = {
        "comparison_performed": ready,
        "same_workload_hash_match": same_workload_hash_match,
        "same_result_hash_match": same_result_hash_match,
        "cpu_timing_distribution": cpu_distribution,
        "board_timing_distribution": board_distribution,
        "board_cpu_ratio": board_cpu_ratio,
        "hardware_speedup_claim": False,
        "comparison_boundary": (
            "The ratio is a matched timing fact only; it is not a hardware speedup claim."
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
        "duration_s": round(max(clock() - started, 0.0), 6),
        "commit": commit,
        "preconditions_checked": True,
        "upstream_gate": upstream_gate,
        "gated_upstream_ready": gated_upstream_ready,
        "selected_workload": workload,
        "workload_hash": workload["workload_hash"],
        "cpu_repeat_count": cpu_repeat_count,
        "board_repeat_count": board_repeat_count,
        "cpu_result_hash": cpu_result_hash,
        "board_result_hash": board_result_hash,
        "same_workload_hash_match": same_workload_hash_match,
        "same_result_hash_match": same_result_hash_match,
        "cpu_timing_variance": float(cpu_distribution["variance_s2"]),
        "board_timing_variance": float(board_distribution["variance_s2"]),
        "polarfire_reachable": polarfire_reachable,
        "kv260_ssh_checked": bool(hardware_summary["kv260_ssh_checked"]),
        "gatemate_diagnostic_checked": bool(hardware_summary["gatemate_diagnostic_checked"]),
        "timing_receipts": [cpu_receipt, board_receipt],
        "command_receipts": command_receipts,
        "measurement_access": {
            "cpu_timing_complete": cpu_repeat_count >= REPEAT_TARGET,
            "board_timing_complete": board_repeat_count >= REPEAT_TARGET,
            "blocked_hardware_precondition": hardware_summary["blocked_hardware_precondition"],
            "comparison_allowed": ready,
            "missing_physical_records_recoverable_after_fact": False,
        },
        "measurement_access_complete": ready,
        "timing_variance_receipts_ready": ready,
        "timing_comparison": timing_comparison,
        "blocked_hardware_precondition": hardware_summary["blocked_hardware_precondition"],
        "hardware_speedup_claim": False,
        "claim_refusal": (
            "No hardware speedup is claimed; slower board timing is a valid "
            "outcome and this artifact only records matched variance receipts."
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
    _require(artifact.get("gated_upstream_ready") is True, "gated_upstream_ready")
    _require(isinstance(artifact.get("workload_hash"), str), "workload_hash")
    _require(isinstance(artifact.get("cpu_repeat_count"), int), "cpu_repeat_count")
    _require(isinstance(artifact.get("board_repeat_count"), int), "board_repeat_count")
    _require(isinstance(artifact.get("cpu_result_hash"), str), "cpu_result_hash")
    _require(isinstance(artifact.get("board_result_hash"), str), "board_result_hash")
    _require(isinstance(artifact.get("same_workload_hash_match"), bool), "same_workload_hash_match")
    _require(isinstance(artifact.get("same_result_hash_match"), bool), "same_result_hash_match")
    _require(isinstance(artifact.get("cpu_timing_variance"), int | float), "cpu_timing_variance")
    _require(isinstance(artifact.get("board_timing_variance"), int | float), "board_timing_variance")
    _require(isinstance(artifact.get("polarfire_reachable"), bool), "polarfire_reachable")
    _require(isinstance(artifact.get("kv260_ssh_checked"), bool), "kv260_ssh_checked")
    _require(
        isinstance(artifact.get("gatemate_diagnostic_checked"), bool),
        "gatemate_diagnostic_checked",
    )
    _require(isinstance(artifact.get("measurement_access_complete"), bool), "measurement_access")
    _require(
        isinstance(artifact.get("timing_variance_receipts_ready"), bool),
        "timing_variance_receipts_ready",
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
    receipts = artifact.get("timing_receipts")
    _require(isinstance(receipts, list) and len(receipts) == 2, "timing_receipts")
    cpu_receipt = receipts[0]
    board_receipt = receipts[1]
    _require(isinstance(cpu_receipt, Mapping), "cpu_receipt")
    _require(isinstance(board_receipt, Mapping), "board_receipt")
    _require(
        cpu_receipt["timing_distribution"]["variance_s2"] == artifact["cpu_timing_variance"],
        "cpu_timing_variance",
    )
    _require(
        board_receipt["timing_distribution"]["variance_s2"] == artifact["board_timing_variance"],
        "board_timing_variance",
    )
    if artifact.get("same_workload_hash_match") is True:
        _require(artifact["board_repeat_count"] >= REPEAT_TARGET, "board_repeat_count")
    if artifact.get("same_result_hash_match") is True:
        _require(artifact["board_result_hash"] == artifact["cpu_result_hash"], "result hash match")
    ready = artifact.get("timing_variance_receipts_ready") is True
    _require(artifact.get("measurement_access_complete") is ready, "measurement access boundary")
    comparison = artifact.get("timing_comparison")
    _require(isinstance(comparison, Mapping), "timing_comparison")
    _require(comparison.get("hardware_speedup_claim") is False, "timing comparison speedup")
    _require(comparison.get("comparison_performed") is ready, "comparison gate")
    if ready:
        _require(artifact.get("same_workload_hash_match") is True, "same_workload_hash_match")
        _require(artifact.get("same_result_hash_match") is True, "same_result_hash_match")
        _require(artifact.get("board_repeat_count") >= REPEAT_TARGET, "board_repeat_count")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(isinstance(comparison.get("board_cpu_ratio"), int | float), "board_cpu_ratio")
    if artifact.get("polarfire_reachable") is False:
        blocker = artifact.get("blocked_hardware_precondition")
        _require(isinstance(blocker, Mapping), "blocked_hardware_precondition")
        _require(blocker.get("command") == command_to_string(POLARFIRE_STATUS_COMMAND), "blocked command")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def write_output(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write stable JSON after validating the Exp5434 artifact contract."""

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
    """Run Exp5434 and write the requested deliverable JSON."""

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
