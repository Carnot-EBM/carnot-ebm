#!/usr/bin/env python3
"""Exp5420: p-bit/QUBO hardware-transfer preflight with CPU reference timing.

Spec refs: REQ-HW-5420, SCENARIO-HW-5420.

This experiment is a readiness gate, not an acceleration benchmark. It selects
one tiny Exp5407 QUBO workload whose optimum was exact-enumerated, repeats the
same exact computation on the CPU, and then asks whether a reachable PolarFire
board can run that same workload hash with the same repeat count. The result can
only say that the workload is ready for a later Exp5424 timing run. It cannot
claim speedup because a preflight receipt is not a controlled hardware-vs-CPU
benchmark.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from carnot import experiment_5407_pbit_qubo_active_constraint_stress_v492 as exp5407


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5420_pbit_hardware_transfer_preflight_v493.json"
)
UPSTREAM_LNS_RELATIVE_PATH = Path("results/experiment_5419_active_constraint_lns_scale_v493.json")
UPSTREAM_PBIT_RELATIVE_PATH = exp5407.RESULT_RELATIVE_PATH

EXPERIMENT = 5420
EXPERIMENT_ID = "exp5420-pbit-hardware-transfer-preflight-v493"
MILESTONE = "2026.07.493"
RUN_DATE = "20260708"
RANDOM_SEED = 5420
SCHEMA = "carnot.experiment_5420.pbit_hardware_transfer_preflight.v493"
SPEC_REFS = ("REQ-HW-5420", "SCENARIO-HW-5420")
INFERENCE_SUBSTRATE = "hardware_preflight_with_cpu_reference"
SELECTED_FIXTURE_ID = "stress_synthetic_linear_review"
CPU_REPEAT_TARGET = 3
TERMINAL_PREFIXES = ("complete:", "blocked:")

SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 30.0
LOCAL_TIMEOUT_S = 10.0

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
POLARFIRE_STATUS_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")

HOST_STORAGE_MARKERS = ("/dev/mmcblk", "/dev/disk")
FORBIDDEN_COMMAND_TERMS = ("--write", "program", "flash")

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "hardware task must fail fast",
    "gated_upstream_ready": "structured gate provenance",
    "workload_hash": "same-workload comparison",
    "cpu_repeat_count": "timing reliability",
    "board_repeat_count": "board reliability",
    "exact_enumeration_match": "validity preservation",
    "same_workload_hash_match": "no apples-to-oranges timing",
    "polarfire_reachable": "hardware availability",
    "kv260_ssh_checked": "SSH-only discipline",
    "gatemate_diagnostic_checked": "physical/JTAG honesty",
    "timing_receipts": "reproducible evidence",
    "hardware_speedup_claim": "no unsupported speedup",
    "pbit_transfer_preflight_ready": "downstream evidence",
    "inference_substrate": "explicit substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class CommandProbe:
    """A bounded command transcript with enough data to audit what was probed."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def combined_output(self) -> str:
        """Return stdout and stderr together for simple identity checks."""

        return "\n".join(part for part in (self.stdout.strip(), self.stderr.strip()) if part)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON deterministically so hashes mean the same thing everywhere."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for command and receipt transcripts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while ignoring its own checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render a command exactly enough for receipts and storage-marker scans."""

    return shlex.join(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded local command without interactive prompts.

    File-not-found and timeout are converted into receipts instead of escaping,
    because missing board tools and unreachable aliases are expected hardware
    precondition outcomes.
    """

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
    """Read the Exp5419 gate that authorizes transfer preflight work."""

    path = Path(root) / UPSTREAM_LNS_RELATIVE_PATH
    record: JsonDict = {
        "artifact_path": str(UPSTREAM_LNS_RELATIVE_PATH),
        "gate_field": "active_constraint_lns_scale_ready",
        "gate_value": False,
        "source_status": "missing",
    }
    if not path.exists():
        return record
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        record["source_status"] = "unreadable"
        return record
    record["gate_value"] = bool(source.get("active_constraint_lns_scale_ready"))
    record["source_status"] = str(source.get("status", "unknown"))
    return record


def _load_upstream_pbit_artifact(root: str | Path = REPO_ROOT) -> JsonDict:
    path = Path(root) / UPSTREAM_PBIT_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def exact_result_hash(exact_result: Mapping[str, Any]) -> str:
    """Hash an exact-enumeration result independently from timing receipts."""

    return sha256_json(dict(exact_result))


def select_workload(root: str | Path = REPO_ROOT) -> JsonDict:
    """Select the exact-enumerated Exp5407 workload used for transfer preflight."""

    upstream = _load_upstream_pbit_artifact(root)
    baseline = next(
        row
        for row in upstream["qubo_baselines"]
        if row["fixture_id"] == SELECTED_FIXTURE_ID
    )
    fixture = next(
        item
        for item in exp5407.build_stress_fixtures()
        if item.fixture_id == SELECTED_FIXTURE_ID
    )
    workload: JsonDict = {
        "fixture_id": fixture.fixture_id,
        "source_artifact": str(UPSTREAM_PBIT_RELATIVE_PATH),
        "source_experiment_id": upstream["experiment_id"],
        "actions": list(fixture.actions),
        "precedence": [list(edge) for edge in fixture.precedence],
        "expected_sequence": list(fixture.expected_sequence),
        "upstream_exact_result": {
            "exact_min_energy": baseline["exact_min_energy"],
            "exact_best_sequence": list(baseline["exact_best_sequence"]),
            "enumerated_permutation_count": baseline["enumerated_permutation_count"],
            "exact_valid_permutation_count": baseline["exact_valid_permutation_count"],
        },
        "seed": RANDOM_SEED,
    }
    exact = exact_enumerate_workload(workload)
    workload["exact_result"] = exact
    workload["exact_result_hash"] = exact_result_hash(exact)
    workload["workload_hash"] = sha256_json(
        {
            "fixture_id": workload["fixture_id"],
            "source_artifact": workload["source_artifact"],
            "actions": workload["actions"],
            "precedence": workload["precedence"],
            "exact_result": workload["exact_result"],
            "seed": RANDOM_SEED,
        }
    )
    return workload


def precedence_energy(workload: Mapping[str, Any], sequence: Sequence[str]) -> int:
    """Score one action order with the same precedence energy as Exp5407."""

    actions = list(workload["actions"])
    precedence = [tuple(edge) for edge in workload["precedence"]]
    if len(sequence) != len(actions) or set(sequence) != set(actions):
        return 10 * len(precedence) + len(actions)
    positions = {action: index for index, action in enumerate(sequence)}
    return 10 * sum(int(positions[before] > positions[after]) for before, after in precedence)


def exact_enumerate_workload(workload: Mapping[str, Any]) -> JsonDict:
    """Enumerate every permutation so the preflight preserves bounded validity."""

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
    if best_sequence is None or best_energy is None:  # pragma: no cover - itertools emits an empty tuple for empty inputs.
        raise ValueError("workload has no permutations")
    return {
        "exact_min_energy": int(best_energy),
        "exact_best_sequence": list(best_sequence),
        "enumerated_permutation_count": enumerated,
        "exact_valid_permutation_count": valid_count,
    }


def cpu_reference_receipt(
    workload: Mapping[str, Any],
    *,
    repeat_count: int = CPU_REPEAT_TARGET,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Repeat exact enumeration on CPU and record timings for the same hash."""

    repeats: list[JsonDict] = []
    timings: list[float] = []
    upstream_exact = workload["upstream_exact_result"]
    for repeat_index in range(repeat_count):
        started = clock()
        exact = exact_enumerate_workload(workload)
        elapsed = round(max(clock() - started, 0.0), 6)
        result_hash = exact_result_hash(exact)
        timings.append(elapsed)
        repeats.append(
            {
                "repeat_index": repeat_index + 1,
                "workload_hash": workload["workload_hash"],
                "seed": RANDOM_SEED,
                "exact_result_hash": result_hash,
                "exact_result": exact,
                "wall_time_s": elapsed,
                "matches_upstream_exact": exact == upstream_exact,
            }
        )
    return {
        "kind": "cpu_reference",
        "substrate": "cpu_exact_enumeration",
        "workload_hash": workload["workload_hash"],
        "seed": RANDOM_SEED,
        "repeat_count": repeat_count,
        "repeat_timings_s": timings,
        "exact_result": dict(workload["exact_result"]),
        "exact_result_hash": workload["exact_result_hash"],
        "exact_enumeration_match": all(row["matches_upstream_exact"] for row in repeats),
        "repeats": repeats,
    }


def _remote_workload_source(workload: Mapping[str, Any]) -> str:
    payload = {
        "actions": workload["actions"],
        "precedence": workload["precedence"],
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
            "receipt={'board_local':True,'workload_hash':workload['workload_hash'],'seed':workload['seed'],'exact_min_energy':exact['exact_min_energy'],'exact_best_sequence':exact['exact_best_sequence'],'exact_result_hash':hashlib.sha256(encoded.encode()).hexdigest(),'wall_time_s':round(time.perf_counter()-started,6)}",
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
    """Parse and validate one PolarFire same-workload receipt."""

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
    if receipt.get("exact_min_energy") != workload["exact_result"]["exact_min_energy"]:
        errors.append("exact_min_energy mismatch")
    if receipt.get("exact_best_sequence") != workload["exact_result"]["exact_best_sequence"]:
        errors.append("exact_best_sequence mismatch")
    if receipt.get("exact_result_hash") != workload["exact_result_hash"]:
        errors.append("exact_result_hash mismatch")
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
    """Build a compact command receipt without storing full hardware output."""

    command = command_to_string(probe.command)
    return {
        "kind": kind,
        "command": command,
        "command_sha256": sha256_text(command),
        "exit_code": int(probe.exit_code),
        "duration_s": round(float(probe.duration_s), 6),
        "timeout_s": float(timeout_s),
        "outcome": outcome,
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
        "stdout_excerpt": probe.stdout.strip()[:240],
        "stderr_excerpt": probe.stderr.strip()[:240],
    }


def _board_exact_matches(attempts: Sequence[Mapping[str, Any]]) -> bool:
    for attempt in attempts:
        receipt = attempt.get("receipt")
        if isinstance(receipt, Mapping) and attempt.get("exact_match") is not True:
            return False
    return True


def collect_hardware_preflight(
    *,
    workload: Mapping[str, Any],
    cpu_repeat_count: int,
    command_runner: CommandRunner,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    """Run only the allowed hardware preflight paths and summarize readiness."""

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
        for repeat_index in range(cpu_repeat_count):
            probe = command_runner(board_command, SSH_TIMEOUT_S)
            receipt, parse_error = parse_board_workload_stdout(probe.stdout, workload)
            exact_match = (
                isinstance(receipt, Mapping)
                and receipt.get("exact_min_energy") == workload["exact_result"]["exact_min_energy"]
                and receipt.get("exact_best_sequence")
                == workload["exact_result"]["exact_best_sequence"]
                and receipt.get("exact_result_hash") == workload["exact_result_hash"]
            )
            valid = probe.exit_code == 0 and receipt is not None and parse_error is None
            board_attempts.append(
                {
                    "repeat_index": repeat_index + 1,
                    "valid": valid,
                    "parse_error": parse_error,
                    "exact_match": exact_match,
                    "receipt": receipt,
                    "workload_hash": receipt.get("workload_hash")
                    if isinstance(receipt, Mapping)
                    else None,
                    "wall_time_s": receipt.get("wall_time_s")
                    if isinstance(receipt, Mapping)
                    else None,
                }
            )
            command_receipts.append(
                command_receipt(
                    probe,
                    kind=f"polarfire_same_workload_repeat_{repeat_index + 1}",
                    timeout_s=SSH_TIMEOUT_S,
                    outcome="valid_repeat" if valid else "invalid_repeat",
                )
            )

    gate_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            gate_probe,
            kind="gatemate_non_destructive_dirtyjtag_detect",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome="detected" if "gatemate" in gate_probe.combined_output.lower() else "blocked",
        )
    )

    valid_attempts = [attempt for attempt in board_attempts if attempt["valid"] is True]
    board_receipt: JsonDict = {
        "kind": "polarfire_board_preflight",
        "substrate": "polarfire_board_local_python",
        "reachable": polarfire_reachable,
        "workload_hash": workload["workload_hash"] if polarfire_reachable else None,
        "repeat_target": cpu_repeat_count,
        "repeat_count": len(board_attempts),
        "valid_repeat_count": len(valid_attempts),
        "invalid_repeat_count": len(board_attempts) - len(valid_attempts),
        "repeat_timings_s": [
            float(attempt["wall_time_s"])
            for attempt in valid_attempts
            if isinstance(attempt.get("wall_time_s"), int | float)
        ],
        "attempts": board_attempts,
    }
    summary = {
        "kv260_ssh_checked": True,
        "kv260_ssh_reachable": kv_probe.exit_code == 0,
        "polarfire_reachable": polarfire_reachable,
        "gatemate_diagnostic_checked": True,
        "gatemate_reachable": "gatemate" in gate_probe.combined_output.lower(),
        "board_repeat_count": len(valid_attempts),
        "same_workload_hash_match": (
            polarfire_reachable
            and len(valid_attempts) == cpu_repeat_count
            and all(attempt.get("workload_hash") == workload["workload_hash"] for attempt in valid_attempts)
            and len(board_attempts) == cpu_repeat_count
        ),
        "board_exact_match": _board_exact_matches(board_attempts),
    }
    return summary, command_receipts, board_receipt


def default_tests_run() -> list[JsonDict]:
    """Keep CLI artifacts valid before the external verifier records pass/fail."""

    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    tests = [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]
    return tests


def _readiness_blockers(
    *,
    gated_upstream_ready: bool,
    exact_enumeration_match: bool,
    same_workload_hash_match: bool,
    polarfire_reachable: bool,
    cpu_repeat_count: int,
    board_repeat_count: int,
) -> list[str]:
    blockers: list[str] = []
    if not gated_upstream_ready:
        blockers.append("active_constraint_lns_scale_not_ready")
    if not exact_enumeration_match:
        blockers.append("exact_enumeration_mismatch")
    if not polarfire_reachable:
        blockers.append("polarfire_unreachable")
    if polarfire_reachable and not same_workload_hash_match:
        blockers.append("same_workload_hash_mismatch")
    if polarfire_reachable and board_repeat_count != cpu_repeat_count:
        blockers.append("board_repeat_count_mismatch")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """State readiness without turning preflight evidence into speedup evidence."""

    if ready:
        return (
            "complete: p-bit/QUBO workload has hash-matched repeated CPU and "
            "PolarFire preflight receipts for Exp5424 timing; hardware_speedup_claim=false"
        )
    joined = ",".join(blockers) if blockers else "preflight_not_ready"
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
    """Build the Exp5420 preflight artifact from CPU and optional board receipts."""

    started = clock()
    gate_source = load_upstream_gate(root)
    gated = gate_source["gate_value"] is True
    tests = _normalize_tests(tests_run)

    workload_hash = ""
    cpu_repeat_count = 0
    board_repeat_count = 0
    exact_match = False
    same_hash = False
    polarfire_reachable = False
    kv260_checked = False
    gatemate_checked = False
    timing_receipts: list[JsonDict] = []
    command_receipts: list[JsonDict] = []
    hardware_summary: JsonDict = {}
    workload: JsonDict | None = None

    if gated:
        workload = select_workload(root)
        workload_hash = str(workload["workload_hash"])
        cpu_receipt = cpu_reference_receipt(workload, repeat_count=CPU_REPEAT_TARGET, clock=clock)
        timing_receipts.append(cpu_receipt)
        cpu_repeat_count = int(cpu_receipt["repeat_count"])
        hardware_summary, command_receipts, board_receipt = collect_hardware_preflight(
            workload=workload,
            cpu_repeat_count=cpu_repeat_count,
            command_runner=command_runner,
        )
        timing_receipts.append(board_receipt)
        board_repeat_count = int(hardware_summary["board_repeat_count"])
        exact_match = bool(cpu_receipt["exact_enumeration_match"] and hardware_summary["board_exact_match"])
        same_hash = bool(hardware_summary["same_workload_hash_match"])
        polarfire_reachable = bool(hardware_summary["polarfire_reachable"])
        kv260_checked = bool(hardware_summary["kv260_ssh_checked"])
        gatemate_checked = bool(hardware_summary["gatemate_diagnostic_checked"])

    blockers = _readiness_blockers(
        gated_upstream_ready=gated,
        exact_enumeration_match=exact_match,
        same_workload_hash_match=same_hash,
        polarfire_reachable=polarfire_reachable,
        cpu_repeat_count=cpu_repeat_count,
        board_repeat_count=board_repeat_count,
    )
    ready = bool(gated and exact_match and same_hash and not blockers)
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
        "gated_upstream_ready": gated,
        "gate_source": gate_source,
        "selected_workload": workload,
        "workload_hash": workload_hash,
        "cpu_repeat_count": cpu_repeat_count,
        "board_repeat_count": board_repeat_count,
        "exact_enumeration_match": exact_match,
        "same_workload_hash_match": same_hash,
        "polarfire_reachable": polarfire_reachable,
        "kv260_ssh_checked": kv260_checked,
        "gatemate_diagnostic_checked": gatemate_checked,
        "timing_receipts": timing_receipts,
        "command_receipts": command_receipts,
        "hardware_summary": hardware_summary,
        "hardware_speedup_claim": False,
        "pbit_transfer_preflight_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "readiness_blockers": blockers,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": tests,
        "claim_limits": [
            "preflight only for Exp5424 timing readiness",
            "CPU exact-enumeration receipt is a reference, not a board baseline speedup claim",
            "PolarFire receipt is same-workload availability and timing evidence only",
            "KV260 check is SSH-only and never host storage evidence",
            "GateMate check is non-destructive DirtyJTAG diagnostics only",
        ],
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


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for index, item in enumerate(tests_run):
        _require(isinstance(item, Mapping), f"tests_run[{index}]")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run command")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run outcome")


def _validate_command_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("command_receipts")
    _require(isinstance(receipts, list), "command_receipts")
    if artifact.get("gated_upstream_ready") is not True:
        _require(receipts == [], "blocked gate should not run hardware commands")
        return
    _require(receipts, "command_receipts")
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
    """Fail closed on apples-to-oranges timing, unsafe commands, or speedup claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("preconditions_checked") is True, "preconditions_checked")
    _require(isinstance(artifact.get("gated_upstream_ready"), bool), "gated_upstream_ready")
    _require(isinstance(artifact.get("workload_hash"), str), "workload_hash")
    _require(isinstance(artifact.get("cpu_repeat_count"), int), "cpu_repeat_count")
    _require(isinstance(artifact.get("board_repeat_count"), int), "board_repeat_count")
    _require(isinstance(artifact.get("exact_enumeration_match"), bool), "exact_enumeration_match")
    _require(isinstance(artifact.get("same_workload_hash_match"), bool), "same_workload_hash_match")
    _require(isinstance(artifact.get("polarfire_reachable"), bool), "polarfire_reachable")
    _require(isinstance(artifact.get("kv260_ssh_checked"), bool), "kv260_ssh_checked")
    _require(
        isinstance(artifact.get("gatemate_diagnostic_checked"), bool),
        "gatemate_diagnostic_checked",
    )
    _require(isinstance(artifact.get("timing_receipts"), list), "timing_receipts")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(isinstance(artifact.get("pbit_transfer_preflight_ready"), bool), "ready")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict speedup boundary")
    _require(not _artifact_mentions_host_storage(artifact), "host block-device evidence present")
    _validate_tests_run(artifact.get("tests_run"))
    _validate_command_receipts(artifact)

    if artifact.get("gated_upstream_ready") is True:
        _require(artifact.get("workload_hash") and len(artifact["workload_hash"]) == 64, "workload_hash")
        _require(artifact.get("cpu_repeat_count") >= CPU_REPEAT_TARGET, "cpu_repeat_count")
        _require(artifact.get("kv260_ssh_checked") is True, "kv260_ssh_checked")
        _require(artifact.get("gatemate_diagnostic_checked") is True, "gatemate_checked")
        _require(artifact.get("exact_enumeration_match") is True, "exact_enumeration_match")
    if artifact.get("pbit_transfer_preflight_ready") is True:
        _require(artifact.get("gated_upstream_ready") is True, "gate")
        _require(artifact.get("polarfire_reachable") is True, "polarfire")
        _require(artifact.get("same_workload_hash_match") is True, "same_workload_hash_match")
        _require(
            artifact.get("board_repeat_count") == artifact.get("cpu_repeat_count"),
            "repeat count match",
        )
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum",
    )


def write_output(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write stable JSON after validating the Exp5420 artifact contract."""

    validate_artifact(artifact)
    path = Path(repo_root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Run Exp5420 and write the requested deliverable JSON."""

    artifact = build_artifact(
        root=repo_root,
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
        run_date=args.date,
        commit=args.commit,
        tests_run=tests_run,
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(f"artifact: {path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
