#!/usr/bin/env python3
"""Exp5478: Exp5477 CPU and reachable-board hardware receipts.

Spec refs: REQ-VERIFY-5478, SCENARIO-VERIFY-5478.

This runner is a continuity receipt, not an acceleration benchmark. It reloads
the Exp5477 workload hashes, recomputes local CPU reference hashes, probes only
SSH-reachable workload boards, and records board-local hash receipts without
turning timing into a speedup claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420
from carnot import experiment_5477_pdit_lns_boundary_exchange_v497 as exp5477


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5478_hardware_receipts_v497.json")
UPSTREAM_EXP5477_RELATIVE_PATH = exp5477.RESULT_RELATIVE_PATH

EXPERIMENT = 5478
EXPERIMENT_ID = "exp5478-hardware-receipts-v497"
MILESTONE = "2026.07.497"
RUN_DATE = "20260709"
RANDOM_SEED = 5478
SCHEMA = "carnot.experiment_5478.hardware_receipts.v497"
SPEC_REFS = ("REQ-VERIFY-5478", "SCENARIO-VERIFY-5478")
INFERENCE_SUBSTRATE = "local_cpu_and_reachable_board_receipts"
REPEAT_TARGET = 3
EXPECTED_WORKLOAD_COUNT = exp5477.EXPECTED_FIXTURE_COUNT
TERMINAL_PREFIXES = ("complete:", "blocked:")

SSH_TIMEOUT_S = 5.0
LOCAL_TIMEOUT_S = 10.0
KV260_SSH_COMMAND = exp5420.KV260_SSH_COMMAND
POLARFIRE_STATUS_COMMAND = exp5420.POLARFIRE_STATUS_COMMAND
WORKLOAD_BOARDS = ("kv260", "polarfire")
GATEMATE_DIAGNOSTIC_REASON = "diagnostic_only_no_exp5477_workload_receipt"
FORBIDDEN_COMMAND_TERMS = ("rm -rf", "mkfs", "dd ", "--write", "program", "flash")

FIELD_PRINCIPLES: dict[str, str] = {
    "upstream_workload_hashes": "Exp5477 workload identity",
    "cpu_reference_hashes": "local CPU correctness receipts",
    "reachable_boards": "boards that accepted safe workload receipts",
    "unreachable_boards": "boards blocked or diagnostic-only",
    "board_receipts": "board-local command and hash receipts",
    "repeat_count": "repeatability support",
    "timing_summary": "receipt timing distribution",
    "result_hash_match_rate": "matched hashes before claims",
    "hardware_receipts_ready": "receipt readiness",
    "hardware_speedup_claim": "receipt-bound acceleration claim",
    "inference_substrate": "explicit local CPU and board substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so hashes are host-independent."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash text receipts, command strings, and compact JSON payloads."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def aggregate_result_hash(result_hashes: Sequence[str]) -> str:
    """Collapse per-workload result hashes into one repeat-level receipt hash."""

    return sha256_json({"result_hashes": list(result_hashes)})


def command_to_string(command: Sequence[str]) -> str:
    """Render commands the same way the existing hardware receipt helpers do."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and turn expected board failures into receipts."""

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


def load_exp5477_workloads(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp5477 hashes and rebuild their canonical CPU reference payloads."""

    upstream = json.loads(
        (Path(root) / UPSTREAM_EXP5477_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    fixtures = exp5477.build_boundary_fixtures()
    fixture_payloads = []
    for fixture in fixtures:
        payload = exp5477.fixture_workload_payload(fixture)
        payload["workload_hash"] = exp5477.workload_hash(fixture)
        fixture_payloads.append(payload)
    workload_hashes = [str(payload["workload_hash"]) for payload in fixture_payloads]
    cpu_reference_hashes = [
        reference_result_hash(payload) for payload in fixture_payloads
    ]
    return {
        "source_artifact": str(UPSTREAM_EXP5477_RELATIVE_PATH),
        "source_experiment_id": str(upstream.get("experiment_id", "")),
        "source_reproducibility_checksum": str(
            upstream.get("reproducibility_checksum", "")
        ),
        "upstream_workload_hashes": list(upstream.get("workload_hashes", [])),
        "workload_hashes": workload_hashes,
        "upstream_hash_match": workload_hashes == list(upstream.get("workload_hashes", [])),
        "fixture_payloads": fixture_payloads,
        "cpu_reference_hashes": cpu_reference_hashes,
    }


def result_payload_from_workload(workload: Mapping[str, Any]) -> JsonDict:
    """Compute the exact reference result for one serialized Exp5477 workload."""

    family = str(workload["fixture_family"])
    if family == "sat":
        label, solution, objective_value = _solve_sat_payload(workload)
    elif family == "maxcut":
        label, solution, objective_value = _solve_maxcut_payload(workload)
    else:
        label, solution, objective_value = _solve_assignment_payload(workload)
    return {
        "fixture_id": str(workload["fixture_id"]),
        "fixture_family": family,
        "workload_hash": str(workload["workload_hash"]),
        "exact_label": label,
        "exact_solution": list(solution),
        "exact_objective_value": objective_value,
    }


def reference_result_hash(workload: Mapping[str, Any]) -> str:
    """Hash one exact result independently from timing receipts."""

    return sha256_json(result_payload_from_workload(workload))


def _solve_sat_payload(workload: Mapping[str, Any]) -> tuple[str, list[bool], int]:
    preferred = list(workload["preferred_solution"])
    best = min(
        (
            candidate
            for candidate in itertools.product((False, True), repeat=len(workload["variables"]))
            if all(_clause_satisfied(candidate, clause) for clause in workload["clauses"])
        ),
        key=lambda candidate: sum(
            int(bool(candidate[index]) != bool(preferred[index]))
            for index in range(len(candidate))
        ),
    )
    objective = sum(
        int(bool(best[index]) != bool(preferred[index])) for index in range(len(best))
    )
    return "sat", [bool(value) for value in best], objective


def _solve_maxcut_payload(workload: Mapping[str, Any]) -> tuple[str, list[int], int]:
    best = max(
        itertools.product((0, 1), repeat=len(workload["variables"])),
        key=lambda candidate: (
            _cut_weight(workload, candidate),
            tuple(-int(value) for value in candidate),
        ),
    )
    weight = _cut_weight(workload, best)
    return f"maxcut_weight={weight}", [int(value) for value in best], -weight


def _solve_assignment_payload(workload: Mapping[str, Any]) -> tuple[str, list[str], int]:
    best = min(
        itertools.permutations(tuple(workload["assignment_domain"])),
        key=lambda candidate: (_assignment_cost(workload, candidate), tuple(candidate)),
    )
    cost = _assignment_cost(workload, best)
    return f"assignment_cost={cost}", [str(value) for value in best], cost


def _clause_satisfied(candidate: Sequence[Any], clause: Sequence[int]) -> bool:
    return any(bool(candidate[abs(int(literal)) - 1]) == (int(literal) > 0) for literal in clause)


def _cut_weight(workload: Mapping[str, Any], candidate: Sequence[Any]) -> int:
    assignment = {
        str(name): int(candidate[index]) for index, name in enumerate(workload["variables"])
    }
    return sum(
        int(weight)
        for left, right, weight in workload["maxcut_edges"]
        if assignment[str(left)] != assignment[str(right)]
    )


def _assignment_cost(workload: Mapping[str, Any], candidate: Sequence[Any]) -> int:
    assignment = {
        str(worker): str(candidate[index])
        for index, worker in enumerate(workload["variables"])
    }
    cost_lookup = {
        (str(worker), str(job)): int(cost)
        for worker, job, cost in workload["assignment_costs"]
    }
    total = sum(
        cost_lookup[(str(worker), assignment[str(worker)])]
        for worker in workload["variables"]
    )
    for left_worker, left_job, right_worker, right_job, cost in workload["pairwise_costs"]:
        if (
            assignment[str(left_worker)] == str(left_job)
            and assignment[str(right_worker)] == str(right_job)
        ):
            total += int(cost)
    return total


def timing_distribution(values: Sequence[float]) -> JsonDict:
    """Summarize repeat timings with count, extrema, mean, median, and variance."""

    timings = [float(value) for value in values]
    if not timings:
        return {
            "count": 0,
            "min_s": 0.0,
            "max_s": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "variance_s2": 0.0,
        }
    variance = statistics.pvariance(timings) if len(timings) > 1 else 0.0
    return {
        "count": len(timings),
        "min_s": round(min(timings), 9),
        "max_s": round(max(timings), 9),
        "mean_s": round(sum(timings) / len(timings), 9),
        "median_s": round(float(statistics.median(timings)), 9),
        "variance_s2": round(float(variance), 12),
    }


def cpu_reference_receipt(
    workload: Mapping[str, Any],
    *,
    repeat_count: int = REPEAT_TARGET,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Repeat local Exp5477 exact references and record timing variance."""

    timings: list[float] = []
    repeats: list[JsonDict] = []
    for repeat_index in range(repeat_count):
        started = clock()
        result_hashes = [
            reference_result_hash(payload) for payload in workload["fixture_payloads"]
        ]
        elapsed = round(max(clock() - started, 0.0), 9)
        timings.append(elapsed)
        repeats.append(
            {
                "repeat_index": repeat_index + 1,
                "workload_hashes": list(workload["workload_hashes"]),
                "result_hashes": result_hashes,
                "aggregate_result_hash": aggregate_result_hash(result_hashes),
                "wall_time_s": elapsed,
                "matches_expected": result_hashes == workload["cpu_reference_hashes"],
            }
        )
    return {
        "kind": "cpu_reference",
        "substrate": "local_cpu_exp5477_exact_reference",
        "workload_hashes": list(workload["workload_hashes"]),
        "result_hashes": list(workload["cpu_reference_hashes"]),
        "aggregate_result_hash": aggregate_result_hash(workload["cpu_reference_hashes"]),
        "repeat_count": repeat_count,
        "repeat_timings_s": timings,
        "timing_distribution": timing_distribution(timings),
        "stable_result_hashes": all(repeat["matches_expected"] is True for repeat in repeats),
        "repeats": repeats,
    }


def _remote_workload_source(board: str, workload: Mapping[str, Any]) -> str:
    payload = {
        "board_identity": board,
        "fixture_payloads": workload["fixture_payloads"],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return "\n".join(
        [
            "import hashlib,itertools,json,time",
            f"payload=json.loads({payload_json!r})",
            "started=time.perf_counter()",
            "def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True)",
            "def h(obj): return hashlib.sha256(canon(obj).encode()).hexdigest()",
            "def clause_ok(c,clause): return any(bool(c[abs(int(lit))-1])==(int(lit)>0) for lit in clause)",
            "def cut_weight(w,c):",
            "    assign={str(name):int(c[i]) for i,name in enumerate(w['variables'])}",
            "    return sum(int(weight) for left,right,weight in w['maxcut_edges'] if assign[str(left)]!=assign[str(right)])",
            "def assignment_cost(w,c):",
            "    assign={str(worker):str(c[i]) for i,worker in enumerate(w['variables'])}",
            "    lookup={(str(worker),str(job)):int(cost) for worker,job,cost in w['assignment_costs']}",
            "    total=sum(lookup[(str(worker),assign[str(worker)])] for worker in w['variables'])",
            "    for lw,lj,rw,rj,cost in w['pairwise_costs']:",
            "        if assign[str(lw)]==str(lj) and assign[str(rw)]==str(rj): total+=int(cost)",
            "    return total",
            "def result(w):",
            "    fam=str(w['fixture_family'])",
            "    if fam=='sat':",
            "        pref=list(w['preferred_solution'])",
            "        best=min((c for c in itertools.product((False,True), repeat=len(w['variables'])) if all(clause_ok(c,cl) for cl in w['clauses'])), key=lambda c: sum(int(bool(c[i])!=bool(pref[i])) for i in range(len(c))))",
            "        sol=[bool(v) for v in best]; obj=sum(int(bool(best[i])!=bool(pref[i])) for i in range(len(best))); label='sat'",
            "    elif fam=='maxcut':",
            "        best=max(itertools.product((0,1), repeat=len(w['variables'])), key=lambda c: (cut_weight(w,c), tuple(-int(v) for v in c)))",
            "        weight=cut_weight(w,best); sol=[int(v) for v in best]; obj=-weight; label='maxcut_weight='+str(weight)",
            "    else:",
            "        best=min(itertools.permutations(tuple(w['assignment_domain'])), key=lambda c: (assignment_cost(w,c), tuple(c)))",
            "        cost=assignment_cost(w,best); sol=[str(v) for v in best]; obj=cost; label='assignment_cost='+str(cost)",
            "    return {'fixture_id':str(w['fixture_id']),'fixture_family':fam,'workload_hash':str(w['workload_hash']),'exact_label':label,'exact_solution':sol,'exact_objective_value':obj}",
            "workload_hashes=[str(w['workload_hash']) for w in payload['fixture_payloads']]",
            "result_hashes=[h(result(w)) for w in payload['fixture_payloads']]",
            "receipt={'aggregate_result_hash':h({'result_hashes':result_hashes}),'board_identity':payload['board_identity'],'board_local':True,'result_hashes':result_hashes,'wall_time_s':round(time.perf_counter()-started,9),'workload_hashes':workload_hashes}",
            "print(json.dumps(receipt,sort_keys=True))",
        ]
    )


def board_workload_command(board: str, workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the SSH command for one safe board-local Exp5477 replay."""

    target = {"kv260": "kria", "polarfire": "polarfire"}[board]
    remote = "python3 - <<'PY'\n" + _remote_workload_source(board, workload) + "\nPY"
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        target,
        remote,
    )


def parse_board_workload_stdout(
    stdout: str,
    workload: Mapping[str, Any],
    board: str,
) -> tuple[JsonDict | None, str | None]:
    """Parse and validate one board-local Exp5477 receipt."""

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
    if receipt.get("board_identity") != board:
        errors.append("board_identity mismatch")
    if receipt.get("workload_hashes") != workload["workload_hashes"]:
        errors.append("workload_hashes mismatch")
    if receipt.get("result_hashes") != workload["cpu_reference_hashes"]:
        errors.append("result_hashes mismatch")
    if receipt.get("aggregate_result_hash") != aggregate_result_hash(
        workload["cpu_reference_hashes"]
    ):
        errors.append("aggregate_result_hash mismatch")
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
    """Build the same compact command receipt shape as existing hardware tasks."""

    return exp5420.command_receipt(probe, kind=kind, timeout_s=timeout_s, outcome=outcome)


def _collect_one_board(
    board: str,
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    status_probe: CommandProbe,
    command_runner: CommandRunner,
) -> tuple[JsonDict | None, list[JsonDict], JsonDict]:
    reachable = status_probe.exit_code == 0
    if not reachable:
        return None, [], _unreachable_entry(board, status_probe)
    command = board_workload_command(board, workload)
    attempts: list[JsonDict] = []
    command_receipts: list[JsonDict] = []
    stdout_combined = ""
    stderr_combined = ""
    for repeat_index in range(repeat_count):
        probe = command_runner(command, SSH_TIMEOUT_S)
        stdout_combined += probe.stdout
        stderr_combined += probe.stderr
        receipt, parse_error = parse_board_workload_stdout(probe.stdout, workload, board)
        valid = probe.exit_code == 0 and receipt is not None and parse_error is None
        matched = valid and receipt.get("result_hashes") == workload["cpu_reference_hashes"]
        attempts.append(
            {
                "repeat_index": repeat_index + 1,
                "valid": valid,
                "matched": matched,
                "parse_error": parse_error,
                "receipt": receipt,
                "wall_time_s": receipt.get("wall_time_s") if isinstance(receipt, Mapping) else None,
            }
        )
        command_receipts.append(
            command_receipt(
                probe,
                kind=f"{board}_exp5477_workload_repeat_{repeat_index + 1}",
                timeout_s=SSH_TIMEOUT_S,
                outcome="valid_repeat" if valid else "invalid_repeat",
            )
        )
    timings = [
        float(attempt["wall_time_s"])
        for attempt in attempts
        if attempt["valid"] is True and isinstance(attempt.get("wall_time_s"), int | float)
    ]
    matched_count = sum(int(attempt["matched"] is True) for attempt in attempts)
    first_receipt = next(
        (attempt["receipt"] for attempt in attempts if isinstance(attempt["receipt"], Mapping)),
        {},
    )
    command_text = command_to_string(command)
    board_receipt = {
        "board_identity": board,
        "command": command_text,
        "command_sha256": sha256_text(command_text),
        "workload_hashes": list(first_receipt.get("workload_hashes", [])),
        "result_hashes": list(first_receipt.get("result_hashes", [])),
        "aggregate_result_hash": str(first_receipt.get("aggregate_result_hash", "")),
        "repeat_count": len(attempts),
        "matched_repeat_count": matched_count,
        "invalid_repeat_count": len(attempts) - sum(int(attempt["valid"] is True) for attempt in attempts),
        "timing_distribution": timing_distribution(timings),
        "stdout_sha256": sha256_text(stdout_combined),
        "stderr_sha256": sha256_text(stderr_combined),
        "stdout_combined": stdout_combined,
        "stderr_combined": stderr_combined,
        "attempts": attempts,
    }
    return board_receipt, command_receipts, {
        "board_identity": board,
        "reachable": True,
        "workload_execution_attempted": True,
        "blocked_reason": None,
    }


def _unreachable_entry(board: str, status_probe: CommandProbe | None = None) -> JsonDict:
    reason = f"blocked_{board}_ssh"
    return {
        "board_identity": board,
        "reachable": False,
        "workload_execution_attempted": False,
        "blocked_reason": reason,
        "status_exit_code": None if status_probe is None else int(status_probe.exit_code),
    }


def gate_mate_diagnostic_entry() -> JsonDict:
    """Record GateMate as diagnostic-only, not an Exp5477 workload substrate."""

    return {
        "board_identity": "gatemate",
        "reachable": False,
        "workload_execution_attempted": False,
        "diagnostic_only": True,
        "blocked_reason": GATEMATE_DIAGNOSTIC_REASON,
    }


def collect_board_receipts(
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    command_runner: CommandRunner,
) -> JsonDict:
    """Collect KV260 and PolarFire SSH-safe workload receipts when reachable."""

    command_receipts: list[JsonDict] = []
    board_receipts: list[JsonDict] = []
    reachable_boards: list[str] = []
    unreachable_boards: list[JsonDict] = []

    probes = (
        ("kv260", KV260_SSH_COMMAND, "kv260_ssh_reachability"),
        ("polarfire", POLARFIRE_STATUS_COMMAND, "polarfire_ssh_reachability"),
    )
    for board, status_command, kind in probes:
        status_probe = command_runner(status_command, SSH_TIMEOUT_S)
        command_receipts.append(
            command_receipt(
                status_probe,
                kind=kind,
                timeout_s=SSH_TIMEOUT_S,
                outcome="reachable" if status_probe.exit_code == 0 else "blocked",
            )
        )
        board_receipt, repeat_command_receipts, status = _collect_one_board(
            board,
            workload=workload,
            repeat_count=repeat_count,
            status_probe=status_probe,
            command_runner=command_runner,
        )
        command_receipts.extend(repeat_command_receipts)
        if board_receipt is None:
            unreachable_boards.append(status)
        else:
            reachable_boards.append(board)
            board_receipts.append(board_receipt)
    unreachable_boards.append(gate_mate_diagnostic_entry())
    return {
        "reachable_boards": reachable_boards,
        "unreachable_boards": unreachable_boards,
        "board_receipts": board_receipts,
        "command_receipts": command_receipts,
    }


def result_hash_match_rate(board_receipts: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of board repeats whose hashes matched CPU references."""

    total = sum(int(receipt.get("repeat_count", 0)) for receipt in board_receipts)
    if total == 0:
        return 0.0
    matched = sum(int(receipt.get("matched_repeat_count", 0)) for receipt in board_receipts)
    return round(matched / total, 6)


def readiness_blockers(
    *,
    cpu_stable: bool,
    reachable_workload_count: int,
    match_rate: float,
    upstream_hash_match: bool = True,
) -> list[str]:
    """Explain why Exp5478 receipts are not ready."""

    blockers: list[str] = []
    if not upstream_hash_match:
        blockers.append("upstream_workload_hash_mismatch")
    if not cpu_stable:
        blockers.append("cpu_reference_unstable")
    if reachable_workload_count == 0:
        blockers.append("no_reachable_workload_board")
    elif match_rate < 1.0:
        blockers.append("board_hash_mismatch")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return the terminal receipt verdict while refusing speedup claims."""

    if ready:
        return (
            "complete: Exp5477 CPU and reachable-board workload hashes matched "
            "with repeated local receipts; hardware_speedup_claim=false"
        )
    joined = ",".join(blockers) if blockers else "hardware_receipts_not_ready"
    return f"blocked: {joined}; hardware_speedup_claim=false"


def default_tests_run() -> list[JsonDict]:
    """Keep CLI artifacts schema-valid before external test metadata is attached."""

    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    return [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp5478 artifact from CPU and reachable-board receipts."""

    started = clock()
    workload = load_exp5477_workloads(root)
    cpu_receipt = cpu_reference_receipt(workload, repeat_count=REPEAT_TARGET, clock=clock)
    boards = collect_board_receipts(
        workload=workload,
        repeat_count=REPEAT_TARGET,
        command_runner=command_runner,
    )
    match_rate = result_hash_match_rate(boards["board_receipts"])
    blockers = readiness_blockers(
        cpu_stable=bool(cpu_receipt["stable_result_hashes"]),
        reachable_workload_count=len(boards["board_receipts"]),
        match_rate=match_rate,
        upstream_hash_match=bool(workload["upstream_hash_match"]),
    )
    ready = not blockers
    timing_summary = {"cpu": cpu_receipt["timing_distribution"]}
    for board in WORKLOAD_BOARDS:
        receipt = next(
            (item for item in boards["board_receipts"] if item["board_identity"] == board),
            None,
        )
        timing_summary[board] = (
            receipt["timing_distribution"] if receipt is not None else timing_distribution([])
        )
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
        "source_artifact": str(UPSTREAM_EXP5477_RELATIVE_PATH),
        "source_experiment_id": workload["source_experiment_id"],
        "source_reproducibility_checksum": workload["source_reproducibility_checksum"],
        "upstream_workload_hashes": list(workload["upstream_workload_hashes"]),
        "cpu_reference_hashes": list(workload["cpu_reference_hashes"]),
        "reachable_boards": boards["reachable_boards"],
        "unreachable_boards": boards["unreachable_boards"],
        "board_receipts": boards["board_receipts"],
        "repeat_count": REPEAT_TARGET,
        "timing_summary": timing_summary,
        "result_hash_match_rate": match_rate,
        "hardware_receipts_ready": ready,
        "hardware_speedup_claim": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "readiness_blockers": blockers,
        "cpu_reference_receipt": cpu_receipt,
        "command_receipts": boards["command_receipts"],
        "selected_workload": {
            "workload_hashes": list(workload["workload_hashes"]),
            "upstream_hash_match": bool(workload["upstream_hash_match"]),
            "expected_workload_count": EXPECTED_WORKLOAD_COUNT,
        },
        "claim_limits": [
            "Exp5477 workload hashes only",
            "CPU and board result hashes must match before timing facts matter",
            "GateMate is diagnostic-only for this artifact",
            "no hardware speedup claim",
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalize_tests(tests_run),
        "research_conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for item in tests_run:
        _require(isinstance(item, Mapping), "tests_run")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run")


def _validate_board_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("board_receipts")
    _require(isinstance(receipts, list), "board_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "board_receipts")
        _require(receipt.get("board_identity") != "gatemate", "diagnostic-only board promoted")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "board command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_sha256")
        _require(
            receipt.get("workload_hashes") == artifact.get("upstream_workload_hashes"),
            "workload_hashes",
        )
        _require(
            isinstance(receipt.get("result_hashes"), list)
            and len(receipt["result_hashes"]) == EXPECTED_WORKLOAD_COUNT,
            "result_hashes",
        )
        _require(
            receipt.get("stdout_sha256") == sha256_text(str(receipt.get("stdout_combined", ""))),
            "stdout_sha256",
        )
        _require(
            receipt.get("stderr_sha256") == sha256_text(str(receipt.get("stderr_combined", ""))),
            "stderr_sha256",
        )
        _require(
            isinstance(receipt.get("timing_distribution"), Mapping),
            "timing_distribution",
        )
    _require("gatemate" not in artifact.get("reachable_boards", []), "diagnostic-only board promoted")


def _validate_command_receipts(receipts: Any) -> None:
    _require(isinstance(receipts, list) and receipts, "command_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipts")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command_receipts")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_receipts")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, mismatched hashes, or unsupported claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(
        isinstance(artifact.get("upstream_workload_hashes"), list)
        and len(artifact["upstream_workload_hashes"]) == EXPECTED_WORKLOAD_COUNT,
        "upstream_workload_hashes",
    )
    _require(
        isinstance(artifact.get("cpu_reference_hashes"), list)
        and len(artifact["cpu_reference_hashes"]) == EXPECTED_WORKLOAD_COUNT,
        "cpu_reference_hashes",
    )
    _require(isinstance(artifact.get("reachable_boards"), list), "reachable_boards")
    _require(isinstance(artifact.get("unreachable_boards"), list), "unreachable_boards")
    _require(artifact.get("repeat_count") == REPEAT_TARGET, "repeat_count")
    _require(isinstance(artifact.get("timing_summary"), Mapping), "timing_summary")
    _require(isinstance(artifact.get("result_hash_match_rate"), int | float), "result_hash_match_rate")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _validate_tests_run(artifact.get("tests_run"))
    _validate_board_receipts(artifact)
    _validate_command_receipts(artifact.get("command_receipts"))
    _require(artifact.get("result_hash_match_rate") == result_hash_match_rate(artifact["board_receipts"]), "result_hash_match_rate")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    if artifact.get("hardware_receipts_ready") is True:
        _require(artifact.get("result_hash_match_rate") == 1.0, "result_hash_match_rate")
        _require(artifact.get("board_receipts"), "board_receipts")
        _require(not artifact.get("readiness_blockers"), "readiness_blockers")
        _require(str(verdict).startswith("complete:"), "honest_verdict")
    else:
        _require(str(verdict).startswith("blocked:"), "honest_verdict")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact under ``root`` and return the path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    workload_root: str | Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Build, validate, and write Exp5478's JSON deliverable."""

    artifact = build_artifact(
        root=workload_root if workload_root is not None else repo_root,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit,
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--workload-root", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    tests = [{"command": command, "outcome": "passed"} for command in args.test_run] or None
    run_experiment(
        repo_root=args.output_root,
        workload_root=args.workload_root,
        tests_run=tests,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main(sys.argv[1:]))
