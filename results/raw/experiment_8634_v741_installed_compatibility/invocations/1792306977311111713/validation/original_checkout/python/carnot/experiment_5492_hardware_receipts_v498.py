#!/usr/bin/env python3
"""Exp5492: Exp5491 descriptor workload hardware receipts.

Spec refs: REQ-VERIFY-5492, SCENARIO-VERIFY-5492.

This module turns Exp5491's portable descriptors into a deterministic workload,
then records local CPU and safe reachable-board receipts for that exact
workload. The important boundary is that receipts are not speedup claims:
timing is only compared after input hashes and output hashes match, KV260 is
identity-only, GateMate is diagnostic-only unless a real execution path exists,
and the artifact keeps ``hardware_speedup_claim`` false.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420
from carnot import experiment_5491_active_constraint_subproblem_descriptor_v498 as exp5491


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5492_hardware_receipts_v498.json")
UPSTREAM_EXP5491_RELATIVE_PATH = exp5491.RESULT_RELATIVE_PATH

EXPERIMENT = 5492
EXPERIMENT_ID = "exp5492-hardware-receipts-v498"
MILESTONE = "2026.07.498"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5492
SCHEMA = "carnot.experiment_5492.hardware_receipts.v498"
SPEC_REFS = ("REQ-VERIFY-5492", "SCENARIO-VERIFY-5492")
INFERENCE_SUBSTRATE = "local_cpu_and_reachable_board_receipts"
REPEAT_TARGET = 3
EXPECTED_WORKLOAD_COUNT = exp5491.EXPECTED_DESCRIPTOR_COUNT
TERMINAL_PREFIXES = ("complete:", "blocked:")

SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 10.0
LOCAL_TIMEOUT_S = 10.0

KV260_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "printf 'board_identity=kv260\\nhostname=' && hostname && printf 'machine=' && uname -m",
)
POLARFIRE_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "printf 'board_identity=polarfire\\nhostname=' && hostname && printf 'machine=' && uname -m",
)
GATEMATE_DETECT_COMMAND = exp5420.GATEMATE_DETECT_COMMAND
WORKLOAD_BOARDS = ("polarfire",)
KV260_IDENTITY_ONLY_REASON = "identity_only_no_exp5491_workload_execution"
GATEMATE_DIAGNOSTIC_REASON = "diagnostic_only_no_exp5491_workload_receipt"
FORBIDDEN_COMMAND_TERMS = ("rm -rf", "mkfs", "dd ", "--write", "program", "flash")
HOST_STORAGE_MARKERS = ("/dev/mmcblk", "/dev/disk")
NON_LOCAL_CLAIM_MARKERS = ("TSU", "Kona", "Aleph")

FIELD_PRINCIPLES: dict[str, str] = {
    "workload_hashes": "Exp5491 descriptor workload identity",
    "cpu_baseline_receipts": "local CPU baseline timing and output hashes",
    "board_receipts": "authenticated local board receipts",
    "reachable_boards": "boards with authenticated safe local evidence",
    "blocked_boards": "per-board blockers without forced pass",
    "repeat_count": "repeatability support",
    "result_hash_match_rate": "matching output hashes before timing comparison",
    "timing_comparison_summary": "bounded timing comparison without speedup claim",
    "authenticated_board_identity_count": "board identity evidence count",
    "hardware_speedup_claim": "must remain false without matched local speedup evidence",
    "hardware_receipts_ready": "receipt readiness",
    "inference_substrate": "explicit local CPU and reachable board substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize a JSON value in a stable byte order for portable hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash command text, stdout, stderr, and canonical JSON receipts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def aggregate_output_hash(result_hashes: Sequence[str]) -> str:
    """Collapse per-workload output hashes into one repeat-level hash."""

    return sha256_json({"result_hashes": list(result_hashes)})


def command_to_string(command: Sequence[str]) -> str:
    """Render a command consistently with existing hardware receipt helpers."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded local command and convert expected hardware failures to receipts."""

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


def descriptor_workload_payload(descriptor: Mapping[str, Any]) -> JsonDict:
    """Return the portable descriptor subset that defines one Exp5492 workload."""

    return {
        "descriptor_id": str(descriptor["descriptor_id"]),
        "source_fixture_id": str(descriptor["source_fixture_id"]),
        "partition_id": str(descriptor["partition_id"]),
        "variables": [dict(item) for item in descriptor["variables"]],
        "domains": {str(name): list(values) for name, values in descriptor["domains"].items()},
        "hard_constraints": [dict(item) for item in descriptor["hard_constraints"]],
        "soft_preferences": [dict(item) for item in descriptor["soft_preferences"]],
        "coupling_type": str(descriptor["coupling_type"]),
        "update_schedule": dict(descriptor["update_schedule"]),
        "canonical_reference": dict(descriptor["canonical_reference"]),
        "exact_fallback": {
            "complete": bool(descriptor["exact_fallback"]["complete"]),
            "status": str(descriptor["exact_fallback"]["status"]),
            "solution_hash": str(descriptor["exact_fallback"]["solution_hash"]),
            "objective_score": descriptor["exact_fallback"]["objective_score"],
            "canonical_reference_agreement": bool(
                descriptor["exact_fallback"]["canonical_reference_agreement"]
            ),
        },
    }


def load_exp5491_workloads(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp5491 descriptors and derive canonical descriptor workload hashes."""

    upstream_path = Path(root) / UPSTREAM_EXP5491_RELATIVE_PATH
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    descriptors = [dict(item) for item in upstream.get("descriptors", [])]
    descriptor_workloads: list[JsonDict] = []
    for descriptor in descriptors:
        exp5491.validate_descriptor(descriptor)
        payload = descriptor_workload_payload(descriptor)
        payload["workload_hash"] = sha256_json(payload)
        descriptor_workloads.append(payload)
    workload_hashes = [str(payload["workload_hash"]) for payload in descriptor_workloads]
    cpu_reference_hashes = [
        reference_result_hash(payload) for payload in descriptor_workloads
    ]
    return {
        "source_artifact": str(UPSTREAM_EXP5491_RELATIVE_PATH),
        "source_experiment_id": str(upstream.get("experiment_id", "")),
        "source_reproducibility_checksum": str(upstream.get("reproducibility_checksum", "")),
        "source_descriptor_ready": bool(upstream.get("subproblem_descriptor_ready")),
        "descriptor_workloads": descriptor_workloads,
        "workload_hashes": workload_hashes,
        "cpu_reference_hashes": cpu_reference_hashes,
    }


def result_payload_from_workload(workload: Mapping[str, Any]) -> JsonDict:
    """Compute the exact descriptor result that CPU and board receipts must match."""

    exact = exp5491.solve_descriptor_exact(workload)
    solution_hash = exp5491.canonical_solution_hash(exact["solution"])
    reference = workload["canonical_reference"]
    return {
        "descriptor_id": str(workload["descriptor_id"]),
        "partition_id": str(workload["partition_id"]),
        "workload_hash": str(workload["workload_hash"]),
        "exact_solution": exact["solution"],
        "exact_objective_score": exact["objective_score"],
        "solution_hash": solution_hash,
        "canonical_reference_agreement": solution_hash == reference["solution_hash"],
        "hard_constraints_satisfied": exp5491.constraints_satisfied(
            exact["solution"],
            workload["hard_constraints"],
        ),
    }


def reference_result_hash(workload: Mapping[str, Any]) -> str:
    """Hash one exact descriptor result independently from timing receipts."""

    return sha256_json(result_payload_from_workload(workload))


def environment_metadata() -> JsonDict:
    """Capture enough CPU environment data to understand where timings came from."""

    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }


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


def cpu_baseline_receipts(
    workload: Mapping[str, Any],
    *,
    repeat_count: int = REPEAT_TARGET,
    clock: Clock = time.perf_counter,
) -> list[JsonDict]:
    """Repeat the local exact descriptor workload and record output hashes."""

    receipts: list[JsonDict] = []
    metadata = environment_metadata()
    for repeat_index in range(repeat_count):
        started = clock()
        result_hashes = [
            reference_result_hash(payload) for payload in workload["descriptor_workloads"]
        ]
        elapsed = round(max(clock() - started, 0.0), 9)
        output_hash = aggregate_output_hash(result_hashes)
        receipts.append(
            {
                "kind": "cpu_baseline",
                "substrate": "local_cpu_exp5491_descriptor_exact_reference",
                "repeat_index": repeat_index + 1,
                "repeat_count": repeat_count,
                "workload_hashes": list(workload["workload_hashes"]),
                "result_hashes": result_hashes,
                "output_hash": output_hash,
                "aggregate_output_hash": output_hash,
                "wall_time_s": elapsed,
                "environment_metadata": dict(metadata),
                "matches_expected": result_hashes == workload["cpu_reference_hashes"],
            }
        )
    return receipts


def cpu_reference_stable(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Return true when every CPU repeat produced the expected output hashes."""

    return bool(receipts) and all(receipt.get("matches_expected") is True for receipt in receipts)


def _remote_workload_source(board: str, workload: Mapping[str, Any]) -> str:
    payload = {
        "board_identity": board,
        "descriptor_workloads": workload["descriptor_workloads"],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return "\n".join(
        [
            "import hashlib,itertools,json,time",
            f"payload=json.loads({payload_json!r})",
            "started=time.perf_counter()",
            "def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True)",
            "def h(obj): return hashlib.sha256(canon(obj).encode()).hexdigest()",
            "def constraint_ok(assign,c):",
            "    kind=c.get('type')",
            "    if kind=='clause': return any(assign[str(l['variable'])]==l['equals'] for l in c['literals'])",
            "    if kind=='all_different':",
            "        vals=[assign[str(v)] for v in c['variables']]",
            "        return len(set(vals))==len(vals)",
            "    raise RuntimeError('constraint_type')",
            "def constraints_ok(assign,constraints): return all(constraint_ok(assign,c) for c in constraints)",
            "def pref_score(assign,p):",
            "    weight=float(p['weight']); kind=p.get('type')",
            "    if kind=='value_reward': return weight if assign[str(p['variable'])]==p['value'] else 0.0",
            "    if kind=='cut_edge': return weight if assign[str(p['left'])]!=assign[str(p['right'])] else 0.0",
            "    raise RuntimeError('preference_type')",
            "def score(w,assign):",
            "    soft=sum(pref_score(assign,p) for p in w['soft_preferences'])",
            "    return soft if constraints_ok(assign,w['hard_constraints']) else soft-1000000",
            "def solution_hash(sol): return h({'solution':dict(sol)})",
            "def solve(w):",
            "    names=[item['name'] for item in w['variables']]",
            "    best=None; best_score=None",
            "    for values in itertools.product(*(w['domains'][name] for name in names)):",
            "        assign=dict(zip(names,values))",
            "        if constraints_ok(assign,w['hard_constraints']):",
            "            sc=score(w,assign)",
            "            if best is None or sc>best_score or (sc==best_score and canon(assign)<canon(best)):",
            "                best=dict(assign); best_score=sc",
            "    if best is None: raise RuntimeError('exact_fallback_unsat')",
            "    return {'solution':best,'objective_score':round(float(best_score),6)}",
            "def result(w):",
            "    exact=solve(w); sol_hash=solution_hash(exact['solution'])",
            "    return {'descriptor_id':str(w['descriptor_id']),'partition_id':str(w['partition_id']),'workload_hash':str(w['workload_hash']),'exact_solution':exact['solution'],'exact_objective_score':exact['objective_score'],'solution_hash':sol_hash,'canonical_reference_agreement':sol_hash==w['canonical_reference']['solution_hash'],'hard_constraints_satisfied':constraints_ok(exact['solution'],w['hard_constraints'])}",
            "workload_hashes=[str(w['workload_hash']) for w in payload['descriptor_workloads']]",
            "result_hashes=[h(result(w)) for w in payload['descriptor_workloads']]",
            "receipt={'aggregate_output_hash':h({'result_hashes':result_hashes}),'board_identity':payload['board_identity'],'board_local':True,'descriptor_count':len(payload['descriptor_workloads']),'result_hashes':result_hashes,'wall_time_s':round(time.perf_counter()-started,9),'workload_hashes':workload_hashes}",
            "print(json.dumps(receipt,sort_keys=True))",
        ]
    )


def board_workload_command(board: str, workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the SSH command for one safe board-local Exp5491 descriptor replay."""

    target = {"polarfire": "polarfire"}[board]
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
    """Parse and validate one board-local Exp5491 descriptor workload receipt."""

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
    if receipt.get("aggregate_output_hash") != aggregate_output_hash(
        workload["cpu_reference_hashes"]
    ):
        errors.append("aggregate_output_hash mismatch")
    if receipt.get("descriptor_count") != len(workload["descriptor_workloads"]):
        errors.append("descriptor_count mismatch")
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
    """Build a compact bounded command transcript receipt."""

    return exp5420.command_receipt(probe, kind=kind, timeout_s=timeout_s, outcome=outcome)


def parse_identity_stdout(stdout: str, board: str) -> JsonDict:
    """Parse simple key-value SSH identity output into a board identity receipt."""

    values: JsonDict = {"board_identity": board}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _identity_status(
    board: str,
    probe: CommandProbe,
    *,
    identity_only_reason: str | None,
) -> tuple[bool, JsonDict | None, JsonDict | None]:
    authenticated = probe.exit_code == 0
    identity = parse_identity_stdout(probe.stdout, board) if authenticated else None
    if authenticated and identity_only_reason is None:
        return True, identity, None
    reason = (
        identity_only_reason
        if authenticated and identity_only_reason is not None
        else f"blocked_{board}_ssh_identity"
    )
    blocked = {
        "board_identity": board,
        "reachable": authenticated,
        "workload_execution_attempted": False,
        "blocked_reason": reason,
        "status_exit_code": int(probe.exit_code),
        "identity": identity,
    }
    return authenticated, identity, blocked


def _gatemate_status(probe: CommandProbe) -> tuple[bool, JsonDict]:
    detected = probe.exit_code == 0 and (
        "IDCODE" in probe.stdout or "GateMate" in probe.stdout or "GM1A" in probe.stdout
    )
    return detected, {
        "board_identity": "gatemate",
        "reachable": detected,
        "workload_execution_attempted": False,
        "diagnostic_only": True,
        "blocked_reason": GATEMATE_DIAGNOSTIC_REASON
        if detected
        else "blocked_gatemate_jtag_identity",
        "status_exit_code": int(probe.exit_code),
        "identity": {
            "board_identity": "gatemate",
            "transport": "dirtyJtag",
            "detect_stdout_sha256": sha256_text(probe.stdout),
        }
        if detected
        else None,
    }


def _collect_workload_board(
    board: str,
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    command_runner: CommandRunner,
) -> tuple[JsonDict, list[JsonDict], JsonDict | None]:
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
                kind=f"{board}_exp5491_descriptor_workload_repeat_{repeat_index + 1}",
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
        "receipt_kind": "workload",
        "command": command_text,
        "command_sha256": sha256_text(command_text),
        "workload_hashes": list(first_receipt.get("workload_hashes", [])),
        "result_hashes": list(first_receipt.get("result_hashes", [])),
        "aggregate_output_hash": str(first_receipt.get("aggregate_output_hash", "")),
        "repeat_count": len(attempts),
        "matched_repeat_count": matched_count,
        "invalid_repeat_count": len(attempts)
        - sum(int(attempt["valid"] is True) for attempt in attempts),
        "timing_distribution": timing_distribution(timings),
        "stdout_sha256": sha256_text(stdout_combined),
        "stderr_sha256": sha256_text(stderr_combined),
        "stdout_combined": stdout_combined,
        "stderr_combined": stderr_combined,
        "attempts": attempts,
    }
    blocked = None
    if matched_count != repeat_count:
        blocked = {
            "board_identity": board,
            "reachable": True,
            "workload_execution_attempted": True,
            "blocked_reason": "board_hash_mismatch",
            "matched_repeat_count": matched_count,
            "repeat_count": repeat_count,
        }
    return board_receipt, command_receipts, blocked


def collect_board_receipts(
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    command_runner: CommandRunner,
) -> JsonDict:
    """Collect safe identity receipts and PolarFire workload receipts when reachable."""

    command_receipts: list[JsonDict] = []
    board_receipts: list[JsonDict] = []
    reachable_boards: list[str] = []
    blocked_boards: dict[str, JsonDict] = {}
    authenticated_count = 0

    kv_probe = command_runner(KV260_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            kv_probe,
            kind="kv260_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome="identity_authenticated" if kv_probe.exit_code == 0 else "blocked",
        )
    )
    kv_authenticated, _, kv_blocked = _identity_status(
        "kv260",
        kv_probe,
        identity_only_reason=KV260_IDENTITY_ONLY_REASON,
    )
    if kv_authenticated:
        authenticated_count += 1
        reachable_boards.append("kv260")
    if kv_blocked is not None:
        blocked_boards["kv260"] = kv_blocked

    gate_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    gate_detected, gate_blocked = _gatemate_status(gate_probe)
    command_receipts.append(
        command_receipt(
            gate_probe,
            kind="gatemate_dirtyjtag_identity",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome="identity_authenticated" if gate_detected else "blocked",
        )
    )
    if gate_detected:
        authenticated_count += 1
        reachable_boards.append("gatemate")
    blocked_boards["gatemate"] = gate_blocked

    pf_probe = command_runner(POLARFIRE_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            pf_probe,
            kind="polarfire_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome="identity_authenticated" if pf_probe.exit_code == 0 else "blocked",
        )
    )
    pf_authenticated, _, pf_blocked = _identity_status(
        "polarfire",
        pf_probe,
        identity_only_reason=None,
    )
    if pf_authenticated:
        authenticated_count += 1
        reachable_boards.append("polarfire")
        board_receipt, repeat_receipts, workload_blocked = _collect_workload_board(
            "polarfire",
            workload=workload,
            repeat_count=repeat_count,
            command_runner=command_runner,
        )
        board_receipts.append(board_receipt)
        command_receipts.extend(repeat_receipts)
        if workload_blocked is not None:
            blocked_boards["polarfire"] = workload_blocked
    elif pf_blocked is not None:
        blocked_boards["polarfire"] = pf_blocked

    return {
        "reachable_boards": reachable_boards,
        "blocked_boards": blocked_boards,
        "board_receipts": board_receipts,
        "command_receipts": command_receipts,
        "authenticated_board_identity_count": authenticated_count,
    }


def result_hash_match_rate(board_receipts: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of workload board repeats whose hashes matched CPU."""

    total = sum(int(receipt.get("repeat_count", 0)) for receipt in board_receipts)
    if total == 0:
        return 0.0
    matched = sum(int(receipt.get("matched_repeat_count", 0)) for receipt in board_receipts)
    return round(matched / total, 6)


def readiness_blockers(
    *,
    source_descriptor_ready: bool,
    cpu_stable: bool,
    workload_board_receipt_count: int,
    match_rate: float,
) -> list[str]:
    """Explain why Exp5492 receipts are not ready."""

    blockers: list[str] = []
    if not source_descriptor_ready:
        blockers.append("source_descriptor_not_ready")
    if not cpu_stable:
        blockers.append("cpu_baseline_unstable")
    if workload_board_receipt_count == 0:
        blockers.append("no_reachable_workload_board")
    elif match_rate < 1.0:
        blockers.append("board_hash_mismatch")
    return blockers


def timing_comparison_summary(
    cpu_receipts: Sequence[Mapping[str, Any]],
    board_receipts: Sequence[Mapping[str, Any]],
    *,
    comparison_allowed: bool,
) -> JsonDict:
    """Summarize CPU and board timings without promoting them to speedup claims."""

    cpu_distribution = timing_distribution(
        [float(item["wall_time_s"]) for item in cpu_receipts]
    )
    boards: JsonDict = {}
    for receipt in board_receipts:
        board_name = str(receipt["board_identity"])
        board_distribution = receipt["timing_distribution"]
        board_mean = float(board_distribution["mean_s"])
        cpu_mean = float(cpu_distribution["mean_s"])
        boards[board_name] = {
            "matched": receipt.get("matched_repeat_count") == receipt.get("repeat_count"),
            "cpu_mean_s": cpu_mean,
            "board_mean_s": board_mean,
            "timing_ratio_cpu_over_board": round(cpu_mean / board_mean, 6)
            if board_mean > 0.0
            else 0.0,
            "speedup_claim_allowed": False,
        }
    return {
        "comparison_allowed": comparison_allowed,
        "hardware_speedup_claim": False,
        "cpu": cpu_distribution,
        "boards": boards,
        "summary": "timing is receipt-only; matching hashes are required before comparison",
    }


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return the terminal receipt verdict while refusing unsupported speedup."""

    if ready:
        return (
            "complete: Exp5491 descriptor workload hashes and reachable-board "
            "output hashes matched; timing is receipt-only; hardware_speedup_claim=false"
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
    """Build the Exp5492 artifact from Exp5491 CPU and reachable-board receipts."""

    started = clock()
    workload = load_exp5491_workloads(root)
    cpu_receipts = cpu_baseline_receipts(workload, repeat_count=REPEAT_TARGET, clock=clock)
    boards = collect_board_receipts(
        workload=workload,
        repeat_count=REPEAT_TARGET,
        command_runner=command_runner,
    )
    match_rate = result_hash_match_rate(boards["board_receipts"])
    blockers = readiness_blockers(
        source_descriptor_ready=bool(workload["source_descriptor_ready"]),
        cpu_stable=cpu_reference_stable(cpu_receipts),
        workload_board_receipt_count=len(boards["board_receipts"]),
        match_rate=match_rate,
    )
    ready = not blockers
    comparison = timing_comparison_summary(
        cpu_receipts,
        boards["board_receipts"],
        comparison_allowed=ready and match_rate == 1.0,
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
        "source_artifact": str(UPSTREAM_EXP5491_RELATIVE_PATH),
        "source_experiment_id": workload["source_experiment_id"],
        "source_reproducibility_checksum": workload["source_reproducibility_checksum"],
        "source_descriptor_ready": bool(workload["source_descriptor_ready"]),
        "workload_hashes": list(workload["workload_hashes"]),
        "cpu_baseline_receipts": cpu_receipts,
        "board_receipts": boards["board_receipts"],
        "reachable_boards": boards["reachable_boards"],
        "blocked_boards": boards["blocked_boards"],
        "repeat_count": REPEAT_TARGET,
        "result_hash_match_rate": match_rate,
        "timing_comparison_summary": comparison,
        "authenticated_board_identity_count": boards["authenticated_board_identity_count"],
        "hardware_speedup_claim": False,
        "hardware_receipts_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "readiness_blockers": blockers,
        "command_receipts": boards["command_receipts"],
        "selected_workload": {
            "source_descriptor_count": len(workload["descriptor_workloads"]),
            "expected_workload_count": EXPECTED_WORKLOAD_COUNT,
            "workload_hashes": list(workload["workload_hashes"]),
            "cpu_reference_hashes": list(workload["cpu_reference_hashes"]),
        },
        "claim_limits": [
            "Exp5491 descriptor workload hashes only",
            "CPU and board result hashes must match before timing facts matter",
            "KV260 is SSH identity-only in this task",
            "GateMate is diagnostic-only unless a physical/JTAG workload path is available",
            "no TSU, Kona, Aleph, remote, or non-local hardware execution claim",
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


def _command_uses_host_storage(command_text: str) -> bool:
    return any(marker in command_text for marker in HOST_STORAGE_MARKERS)


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for item in tests_run:
        _require(isinstance(item, Mapping), "tests_run")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run")


def _validate_cpu_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("cpu_baseline_receipts")
    _require(isinstance(receipts, list) and receipts, "cpu_baseline_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "cpu_baseline_receipts")
        _require(receipt.get("repeat_count") == REPEAT_TARGET, "cpu_baseline_receipts")
        _require(receipt.get("workload_hashes") == artifact.get("workload_hashes"), "workload_hashes")
        _require(isinstance(receipt.get("wall_time_s"), int | float), "wall_time_s")
        _require(isinstance(receipt.get("output_hash"), str) and len(receipt["output_hash"]) == 64, "output_hash")
        _require(isinstance(receipt.get("environment_metadata"), Mapping), "environment_metadata")


def _validate_board_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("board_receipts")
    _require(isinstance(receipts, list), "board_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "board_receipts")
        board = receipt.get("board_identity")
        _require(board != "gatemate", "diagnostic-only board promoted")
        _require(board != "kv260", "identity-only board promoted")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "board command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(not _command_uses_host_storage(command), "host storage command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_sha256")
        _require(receipt.get("workload_hashes") == artifact.get("workload_hashes"), "workload_hashes")
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
        _require(isinstance(receipt.get("timing_distribution"), Mapping), "timing_distribution")


def _validate_command_receipts(receipts: Any) -> None:
    _require(isinstance(receipts, list) and receipts, "command_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipts")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command_receipts")
        _require(not _command_is_destructive(command), "destructive command")
        _require(not _command_uses_host_storage(command), "host storage command")
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
        isinstance(artifact.get("workload_hashes"), list)
        and len(artifact["workload_hashes"]) == EXPECTED_WORKLOAD_COUNT,
        "workload_hashes",
    )
    _require(isinstance(artifact.get("reachable_boards"), list), "reachable_boards")
    _require(isinstance(artifact.get("blocked_boards"), Mapping), "blocked_boards")
    _require(artifact.get("repeat_count") == REPEAT_TARGET, "repeat_count")
    _require(isinstance(artifact.get("result_hash_match_rate"), int | float), "result_hash_match_rate")
    _require(isinstance(artifact.get("timing_comparison_summary"), Mapping), "timing_comparison_summary")
    _require(
        isinstance(artifact.get("authenticated_board_identity_count"), int),
        "authenticated_board_identity_count",
    )
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict")
    _require(not any(marker in verdict for marker in NON_LOCAL_CLAIM_MARKERS), "non-local claim")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _validate_tests_run(artifact.get("tests_run"))
    _validate_cpu_receipts(artifact)
    _validate_board_receipts(artifact)
    _validate_command_receipts(artifact.get("command_receipts"))
    _require(
        artifact.get("result_hash_match_rate") == result_hash_match_rate(artifact["board_receipts"]),
        "result_hash_match_rate",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    if artifact.get("hardware_receipts_ready") is True:
        _require(artifact.get("result_hash_match_rate") == 1.0, "result_hash_match_rate")
        _require(artifact.get("board_receipts"), "board_receipts")
        _require(not artifact.get("readiness_blockers"), "readiness_blockers")
        _require(str(verdict).startswith("complete:"), "honest_verdict")
        _require(
            artifact["timing_comparison_summary"].get("comparison_allowed") is True,
            "timing_comparison_summary",
        )
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
    """Build, validate, and write Exp5492's JSON deliverable."""

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
