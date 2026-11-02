"""Audit RTX 3090 capacity and one no-model owned lease lifecycle.

The audit repairs a narrow precondition bug. Capacity says that at least one
device can be leased. Ownership says that this process actually acquired one.
The two facts stay separate, and this module never loads model weights.

Spec refs: REQ-VERIFY-7422 and SCENARIO-VERIFY-7422-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7400_v649_assignment_canary as canary
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
PHASE = 1
EXPERIMENT_ID = "exp7422-v651-runtime-ownership"
TASK_ID = "experiment_7422_v651_runtime_ownership"
SCHEMA = "carnot.exp7422.v651.runtime_ownership.v1"
MODEL_SPECS: list[str] = []
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
LEASE_WAIT_TIMEOUT_S = 120.0
RELEASE_TIMEOUT_S = 5.0
RANDOM_SEED = None

MODULE_PATH = Path("python/carnot/experiment_7422_v651_runtime_ownership.py")
SHARED_MODULE_PATH = Path("python/carnot/experiment_7400_v649_assignment_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7422_v651_runtime_ownership.py")
TEST_PATH = Path("tests/python/test_experiment_7422_v651_runtime_ownership.py")
SHARED_TEST_PATH = Path("tests/python/test_experiment_7400_v649_assignment_canary.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/constraint-verification/spec.md"
RESULT_PATH = Path("results/experiment_7422_v651_runtime_ownership.json")
RAW_DIR = Path("results/raw/experiment_7422_v651_runtime_ownership")
HISTORICAL_PATH = Path("results/experiment_7416_v650_anchored_extraction.json")

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), SHARED_TEST_PATH.as_posix()),
    changed_modules=(MODULE_PATH.as_posix(), SHARED_MODULE_PATH.as_posix()),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Use a versioned plain top-level schema with experiment, milestone, and terminal state.",
    "run_date": "Use 20260919 and retain actual UTC and monotonic boundaries.",
    "preconditions_checked": "Record exact source, path, identity, inventory, and owner observations before dependent work.",
    "MODEL_SPECS": "Use an empty list because this audit invokes no current LLM.",
    "model_invoked": "Set true only after an actual current model load or generation attempt.",
    "invocation_counts": "Keep each current owned load and generation disposition explicit and zero here.",
    "inference_substrate": "Name the no-model host resource audit; keep device facts in substrate details.",
    "inference_substrate_class": "Use no_model_load because this task never opens model weights.",
    "execution_venue": "Use the closed host value; CUDA and device identities are details.",
    "duration_s": "Measure current work monotonically and separate resource, validation, and cold-reader time.",
    "phase_spans": "Keep real phase boundaries, completed units, and checkpoint references.",
    "random_seed": "Use null because this deterministic ownership audit has no sampling.",
    "reproducibility_checksum": "Bind code, protocol, inputs, rows, lease identity, and validation scope.",
    "source_artifact_hashes": "Hash exact input bytes and label historical model-shaped evidence as non-current.",
    "rows": "Keep every capacity fixture, host observation, and lease lifecycle disposition.",
    "sample_size_budget": "Predeclare planned, attempted, completed, failed, censored, and unstarted independent units.",
    "acceptance_gate_results": "Keep validity, capacity, ownership, validation, safety, and promotion checks separate.",
    "gate_check_summary": "Name the exact upstream, path, field, operator, expected value, and observed value.",
    "verifier_is_oracle": "Use true because the lease protocol and journal validator define ownership correctness.",
    "honest_verdict": "Start completed findings with complete_ and unchanged unavailable inputs with blocked_.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings; flagged evidence cannot supply readiness.",
    "validation_receipts": "Retain exact scoped commands, environments, exits, durations, and hashed logs.",
    "field_principles": "Explain field intent separately and keep gate scalars as plain values.",
    "promotion_score": "Keep zero; this audit changes no rollout, publication, or generator weight.",
    "runtime_ownership_ready_score": "Set one only after repaired capacity checks and one current acquire-read-release lifecycle pass.",
    "capacity_rows": "Separate query success, integer capacity, predicate outcome, and owner state for each fixture and host.",
    "lease_rows": "Keep actual acquisition, child readback, terminal transition, release identity, and timestamps.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


def utc_now() -> str:  # pragma: no cover - actual wall-clock boundary.
    """Return one current UTC boundary while durations use a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every boundary so resource waits and subprocesses stay observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7422] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a referenced source or row cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for artifact and independent reduction evidence."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object only after its bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for absent or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact without recursively hashing its checksum."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return canonical_hash(value)


def compare(operator: str, observed: Any, expected: Any) -> bool:
    """Apply only the scalar operators declared by this experiment."""

    if operator == "==":
        return observed == expected
    if operator == ">=":
        return observed >= expected
    if operator == "in":
        return observed in expected
    raise ValueError(f"unsupported_operator:{operator}")


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    category: str = "precondition",
    principle: str | None = None,
) -> JsonDict:
    """Keep both gate operands so any failure names the exact observed value."""

    try:
        passed = compare(operator, observed, expected)
    except (TypeError, ValueError):
        passed = False
    return {
        "category": category,
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "operator": operator,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": passed,
        "principle": principle or "The declared operands determine this check.",
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failure while retaining every failed check."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("path") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "artifact_field": first.get("artifact_field") if first else "gate_check_summary",
        "operator": first.get("operator") if first else "==",
        "expected_value": deepcopy(first.get("expected_value")) if first else True,
        "observed_value": deepcopy(first.get("observed_value")) if first else True,
        "passed": not failures,
    }


def zero_counts() -> JsonDict:
    """Return explicit zero current model load and generation dispositions."""

    return deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)


def _lease_state(lease: Mapping[str, Any]) -> JsonDict:
    """Keep owner identity and whether the shipped protocol may recover safely."""

    owner_live = lease.get("owner_live") is True
    canonical = lease.get("canonical") is True and lease.get("readable") is True
    unreleased = lease.get("released") is not True
    return {
        "lease_id": lease.get("lease_id"),
        "device_uuid": lease.get("device_uuid"),
        "owner_pid": lease.get("owner_pid"),
        "owner_start_ticks": lease.get("owner_start_ticks"),
        "owner_live": owner_live,
        "canonical": canonical,
        "released": lease.get("released"),
        "fresh": lease.get("fresh"),
        "recovery_eligible": bool(canonical and unreleased and not owner_live),
        "error": lease.get("error"),
    }


def reduce_available_rtx3090s(
    process_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce read-only inventory without converting capacity into ownership.

    Any process makes its device unavailable to this new task. A live lease
    also blocks acquisition. A canonical lease whose PID/start identity is no
    longer live is recoverable by the shipped kernel-lock protocol. Malformed
    unreleased lease evidence blocks all devices because its scope is unknown.
    """

    owner_state = [_lease_state(row) for row in lease_rows]
    owner_state.extend(
        {
            "lease_id": row.get("matching_lease_id"),
            "device_uuid": row.get("gpu_uuid"),
            "owner_pid": row.get("pid"),
            "owner_start_ticks": row.get("start_time_ticks"),
            "owner_live": row.get("proc_exists") is True,
            "ownership_classification": row.get("ownership_classification"),
            "recovery_eligible": False,
            "ownership_evidence_errors": deepcopy(list(row.get("ownership_evidence_errors") or [])),
        }
        for row in process_rows
        if row.get("pid") is not None
    )
    unknown_unreleased = any(
        row.get("released") is not True
        and (row.get("readable") is not True or row.get("canonical") is not True)
        for row in lease_rows
    )
    available: list[str] = []
    devices = sorted({str(row.get("gpu_uuid")) for row in process_rows if row.get("gpu_uuid")})
    for device_uuid in devices:
        rows = [row for row in process_rows if str(row.get("gpu_uuid")) == device_uuid]
        sample = rows[0]
        processes = [row for row in rows if row.get("pid") is not None]
        live_or_unknown_lease = any(
            row.get("device_uuid") == device_uuid
            and row.get("released") is not True
            and (
                row.get("owner_live") is True
                or row.get("readable") is not True
                or row.get("canonical") is not True
            )
            for row in lease_rows
        )
        identity_known = all(
            row.get("proc_exists") is True
            and row.get("ownership_classification") in {"owned", "adoptable", "conflicting"}
            for row in processes
        )
        idle = (
            not processes
            and "RTX 3090" in str(sample.get("gpu_name") or "")
            and int(sample.get("gpu_utilization_pct", 100) or 0)
            <= canary.runtime.lease_preflight.MAX_IDLE_UTILIZATION_PCT
            and int(sample.get("gpu_memory_free_mb", 0) or 0)
            >= canary.runtime.lease_preflight.MIN_IDLE_FREE_MB
            and int(sample.get("gpu_memory_used_mb", 0) or 0)
            <= canary.runtime.lease_preflight.MAX_IDLE_USED_MB
        )
        if idle and identity_known and not live_or_unknown_lease and not unknown_unreleased:
            available.append(device_uuid)
    return {
        "available_gpu_uuids": available,
        "observed_owner_state": owner_state,
        "unknown_unreleased_lease": unknown_unreleased,
    }


def reduce_capacity_row(
    fixture_id: str,
    process_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    query_receipts: Sequence[Mapping[str, Any]],
    *,
    expected_query_passed: bool | None = None,
    expected_capacity_passed: bool | None = None,
) -> JsonDict:
    """Independently reduce one fixture or host observation to scalar gates."""

    availability = reduce_available_rtx3090s(process_rows, lease_rows)
    available = list(availability["available_gpu_uuids"])
    gates = canary.rtx3090_capacity_gate_rows(query_receipts, available)
    query_ok = bool(gates[0]["observed_value"])
    capacity_passed = bool(gates[1]["passed"])
    expected_query = query_ok if expected_query_passed is None else expected_query_passed
    expected_capacity = (
        capacity_passed if expected_capacity_passed is None else expected_capacity_passed
    )
    legacy_expected = {"query_ok": True, "minimum_rtx3090_slots": 1}
    legacy_observed = {
        "query_ok": query_ok,
        "minimum_rtx3090_slots": len(available),
    }
    return {
        "row_kind": "capacity",
        "fixture_id": fixture_id,
        "query_receipts": [deepcopy(dict(row)) for row in query_receipts],
        "process_rows": [deepcopy(dict(row)) for row in process_rows],
        "lease_observations": [deepcopy(dict(row)) for row in lease_rows],
        "query_ok": query_ok,
        "available_gpu_uuids": available,
        "available_capacity": len(available),
        "capacity_operator": ">=",
        "minimum_required_capacity": 1,
        "capacity_predicate_passed": capacity_passed,
        "legacy_expected": legacy_expected,
        "legacy_observed": legacy_observed,
        "legacy_dictionary_equality_passed": legacy_observed == legacy_expected,
        "observed_owner_state": availability["observed_owner_state"],
        "unknown_unreleased_lease": availability["unknown_unreleased_lease"],
        "expected_query_passed": expected_query,
        "expected_capacity_passed": expected_capacity,
        "fixture_expectation_passed": (
            query_ok is expected_query and capacity_passed is expected_capacity
        ),
        "passed": all(row["passed"] for row in gates),
        "gate_rows": gates,
    }


def _fixture_gpu(uuid: str, *, pid: int | None = None, known: bool = True) -> JsonDict:
    """Create private inventory bytes for deterministic capacity controls."""

    return {
        "gpu_index": int(uuid[-1]),
        "gpu_uuid": uuid,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_utilization_pct": 0,
        "gpu_memory_total_mb": 24576,
        "gpu_memory_used_mb": 4 if pid is None else 12000,
        "gpu_memory_free_mb": 24572 if pid is None else 12576,
        "pid": pid,
        "gpu_process_memory_mb": 0 if pid is None else 11996,
        "proc_exists": known if pid is not None else None,
        "start_time_ticks": 1234 if pid is not None and known else None,
        "ownership_classification": "idle" if pid is None else "conflicting",
        "matching_lease_id": None,
        "ownership_evidence_errors": ([] if pid is None else ["current_canonical_lease_missing"]),
    }


def build_private_capacity_rows() -> list[JsonDict]:
    """Build frozen 0/1/2-device and ownership controls without host access."""

    success = [{"returncode": 0}, {"returncode": 0}]
    failure = [{"returncode": 0}, {"returncode": 9}]
    stale = {
        "device_uuid": "GPU-0",
        "readable": True,
        "canonical": True,
        "released": False,
        "fresh": True,
        "owner_live": False,
        "owner_pid": 50,
        "owner_start_ticks": 1,
        "lease_id": "lease:stale-owner",
        "error": None,
    }
    return [
        reduce_capacity_row(
            "failed_inventory",
            [_fixture_gpu("GPU-0"), _fixture_gpu("GPU-1")],
            [],
            failure,
            expected_query_passed=False,
            expected_capacity_passed=True,
        ),
        reduce_capacity_row(
            "zero_free_devices",
            [],
            [],
            success,
            expected_query_passed=True,
            expected_capacity_passed=False,
        ),
        reduce_capacity_row(
            "one_free_device",
            [_fixture_gpu("GPU-0")],
            [],
            success,
            expected_query_passed=True,
            expected_capacity_passed=True,
        ),
        reduce_capacity_row(
            "two_free_devices",
            [_fixture_gpu("GPU-0"), _fixture_gpu("GPU-1")],
            [],
            success,
            expected_query_passed=True,
            expected_capacity_passed=True,
        ),
        reduce_capacity_row(
            "one_busy_device",
            [_fixture_gpu("GPU-0", pid=50), _fixture_gpu("GPU-1")],
            [],
            success,
            expected_query_passed=True,
            expected_capacity_passed=True,
        ),
        reduce_capacity_row(
            "unknown_process_state",
            [_fixture_gpu("GPU-0", pid=50, known=False)],
            [],
            success,
            expected_query_passed=True,
            expected_capacity_passed=False,
        ),
        reduce_capacity_row(
            "stale_owner_identity",
            [_fixture_gpu("GPU-0")],
            [stale],
            success,
            expected_query_passed=True,
            expected_capacity_passed=True,
        ),
    ]


def complete_no_model_lease(lease: lease_api.GpuLease) -> JsonDict:
    """Make one no-model lease terminal and release only its kernel lock."""

    lease.transition("admitted")
    lease.transition("terminal_blocked")
    release_started = time.monotonic()
    release = lease.release()
    duration = time.monotonic() - release_started
    return {
        **deepcopy(release),
        "release_passed": release.get("released") is True and duration <= RELEASE_TIMEOUT_S,
        "release_duration_s": duration,
        "terminal_phase": release.get("phase"),
    }


def build_no_model_current_receipt(
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    sidecars: Sequence[Mapping[str, Any]],
    *,
    owner_pid: int | None = None,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build current provenance with no model events and explicit zero counts."""

    return current_work_receipt.build_current_work_receipt(
        run_id=f"{TASK_ID}:{started_monotonic_ns}",
        owner_pid=os.getpid() if owner_pid is None else owner_pid,
        events=[],
        inference_substrate="deterministic_runtime_receipt_validation_no_llm",
        inference_substrate_details={"lease_protocol": "gpu_lease_phase_journal.v1"},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=sidecars,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False},
    )


def _model_path_prerequisite() -> JsonDict:
    """Resolve model metadata without opening tensor or tokenizer payloads."""

    resolved = cached_current_model()
    path = Path(str(resolved.get("model_path"))) if resolved else None
    exists = bool(path and path.is_file())
    content_hash = canary.runtime._content_addressed_hash(path) if exists and path else None
    return {
        "hf_id": resolved.get("hf_id") if resolved else None,
        "path": str(path.resolve()) if exists and path else None,
        "bytes": path.stat().st_size if exists and path else None,
        "sha256": content_hash,
        "weights_opened": False,
        "exists": exists,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate branch-local inputs and one path-only model prerequisite."""

    required = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7400_v649_assignment_canary.py"),
        Path("python/carnot/experiment_7416_v650_anchored_extraction.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("openspec/capabilities/constraint-verification/spec.md"),
        HISTORICAL_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        SHARED_TEST_PATH,
    )
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in required:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
    spec = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.relative_to(root).as_posix(),
            "REQ-*",
            "==",
            "REQ-VERIFY-7422",
            "REQ-VERIFY-7422" if "REQ-VERIFY-7422" in spec else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7422" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "experiment_id",
            "==",
            False,
            excluded,
        )
    )
    model = _model_path_prerequisite()
    checks.append(
        gate_row(
            "current_model_path_metadata",
            "cached_current_model",
            "model_file_metadata",
            "==",
            {"exists": True, "hash_present": True, "weights_opened": False},
            {
                "exists": model["exists"],
                "hash_present": bool(model["sha256"]),
                "weights_opened": model["weights_opened"],
            },
        )
    )
    return checks, {
        "source_hashes": hashes,
        "model_path_prerequisite": model,
        "historical": load_object(root / HISTORICAL_PATH),
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze the Exp7358 command plan for exactly the affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad targets, missing parents, and any command-plan drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one passing terminal receipt for each declared name."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is not True
        for name in names
    )


def _fixture_audit_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require every private control to match its declared expected outcome."""

    expected_ids = [row["fixture_id"] for row in build_private_capacity_rows()]
    fixture = [row for row in rows if row.get("fixture_id") != "host"]
    return [row.get("fixture_id") for row in fixture] == expected_ids and all(
        row.get("fixture_expectation_passed") is True for row in fixture
    )


def _lease_lifecycle_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require exactly one matching acquire, fresh readback, and bounded release."""

    if len(rows) != 1:
        return False
    row = rows[0]
    start_tick = row.get("owner_pid_start_ticks")
    return bool(
        row.get("task_id") == TASK_ID
        and isinstance(row.get("owner_pid"), int)
        and isinstance(start_tick, int)
        and row.get("fresh_child_readback_passed") is True
        and row.get("release_passed") is True
        and isinstance(row.get("release_duration_s"), (int, float))
        and not isinstance(row.get("release_duration_s"), bool)
        and float(row["release_duration_s"]) <= RELEASE_TIMEOUT_S
        and row.get("terminal_phase") == "terminal_blocked"
        and row.get("signals_sent") == []
        and bool(row.get("lease_id"))
        and bool(row.get("device_uuid"))
    )


def _acceptance_gates(
    capacity_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions_ok: bool,
    flagged: bool,
) -> list[JsonDict]:
    """Keep capacity, ownership, validation, and safety decisions separate."""

    host_rows = [row for row in capacity_rows if row.get("fixture_id") == "host"]
    host_ready = len(host_rows) == 1 and host_rows[0].get("passed") is True
    return [
        gate_row(
            "branch_local_preconditions",
            EXPERIMENT_ID,
            "preconditions_checked",
            "==",
            True,
            preconditions_ok,
            category="validity",
        ),
        gate_row(
            "private_capacity_controls",
            EXPERIMENT_ID,
            "capacity_rows",
            "==",
            True,
            _fixture_audit_passed(capacity_rows),
            category="validity",
        ),
        gate_row(
            "host_available_capacity",
            "nvidia-smi_and_gpu_lease_journal",
            "available_rtx3090_slots",
            "==",
            True,
            host_ready,
            category="capacity",
        ),
        gate_row(
            "owned_lease_lifecycle",
            "gpu_lease_phase_journal.GpuLease",
            "lease_rows",
            "==",
            True,
            _lease_lifecycle_passed(lease_rows),
            category="ownership",
        ),
        gate_row(
            "affected_validation",
            "validation_receipts",
            "required_checks_passed",
            "==",
            True,
            receipts_pass(receipts, REQUIRED_CHECK_NAMES),
            category="validation",
        ),
        gate_row(
            "terminal_readers",
            "validation_receipts",
            "terminal_checks_passed",
            "==",
            True,
            receipts_pass(receipts, TERMINAL_CHECK_NAMES),
            category="validation",
        ),
        gate_row(
            "no_current_model_invocation",
            "current_work_receipt",
            "model_invoked",
            "==",
            False,
            False,
            category="safety",
        ),
        gate_row(
            "adversarial_clean",
            "adversarial_verify",
            "flagged_adversarial",
            "==",
            False,
            flagged,
            category="safety",
        ),
        gate_row(
            "promotion_disabled",
            EXPERIMENT_ID,
            "promotion_score",
            "==",
            0,
            0,
            category="promotion",
        ),
    ]


def _base_artifact() -> JsonDict:
    """Return one complete ordinary-field shape for every terminal disposition."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_not_started",
        "run_date": RUN_DATE,
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": zero_counts(),
        "current_invocation_events": [],
        "current_run_id": None,
        "current_owner_pid": None,
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_substrate": "deterministic_runtime_receipt_validation_no_llm",
        "inference_substrate_details": {},
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "started_monotonic_ns": 0,
        "ended_monotonic_ns": 0,
        "duration_s": 0.0,
        "resource_audit_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "receipt_sidecars": [],
        "small_ebm_training": {"performed": False},
        "model_file_metadata": {},
        "rows": [],
        "raw_row_manifest": [],
        "sample_size_budget": {
            "planned": 8,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 8,
            "independent_groups": 8,
            "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
            "release_timeout_s": RELEASE_TIMEOUT_S,
            "stop_rule": "run seven frozen capacity controls and one current owned lease lifecycle once",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_not_started",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "unrelated_known_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "runtime_ownership_ready_score": 0,
        "capacity_rows": [],
        "lease_rows": [],
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "fresh_process_cold_replay"],
        "learning_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_changed": False,
        "runtime_reservation_retained": False,
    }


def build_artifact_for_test(
    *,
    capacity_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    capacity_values = [deepcopy(dict(row)) for row in capacity_rows]
    if not any(row.get("fixture_id") == "host" for row in capacity_values):
        source = next(row for row in capacity_values if row.get("fixture_id") == "one_free_device")
        capacity_values.append(
            reduce_capacity_row(
                "host",
                list(source["process_rows"]),
                list(source["lease_observations"]),
                list(source["query_receipts"]),
            )
        )
    current = build_no_model_current_receipt(0, 1_000_000_000, [], owner_pid=os.getpid())
    gates = _acceptance_gates(
        capacity_values,
        lease_rows,
        receipts,
        preconditions_ok=True,
        flagged=False,
    )
    ready = int(all(row["passed"] for row in gates))
    artifact = _base_artifact()
    rows = [deepcopy(dict(row)) for row in (*capacity_values, *lease_rows)]
    artifact.update(
        {
            "status": "complete_runtime_ownership_ready",
            "started_at_utc": "2026-09-19T00:00:00Z",
            "completed_at_utc": "2026-09-19T00:00:01Z",
            "preconditions_checked": [
                gate_row("fixture", "unit_test", "fixture", "==", True, True)
            ],
            **current,
            "duration_s": 1.0,
            "resource_audit_duration_s": 0.4,
            "validation_duration_s": 0.5,
            "cold_start_duration_s": 0.1,
            "phase_spans": [],
            "source_artifact_hashes": {"fixture": canonical_hash(rows)},
            "model_file_metadata": {"weights_opened": False},
            "rows": rows,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "planned": len(rows),
                "attempted": len(rows),
                "completed": len(rows),
                "unstarted": 0,
                "independent_groups": len(rows),
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "honest_verdict": "complete_null_runtime_ownership_ready_no_reservation",
            "verdict_class": "null",
            "validation_receipts": [deepcopy(dict(row)) for row in receipts],
            "repository_health": {
                "status": "healthy",
                "unrelated_known_failures": [],
                "affects_required_checks": False,
            },
            "runtime_ownership_ready_score": ready,
            "capacity_rows": capacity_values,
            "lease_rows": [deepcopy(dict(row)) for row in lease_rows],
        }
    )
    return artifact


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    capacity_rows: Sequence[Mapping[str, Any]] = (),
    owner_state: Sequence[Mapping[str, Any]] = (),
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Publish unavailable input or capacity with exact state and zero model work."""

    artifact = _base_artifact()
    summary = gate_check_summary(checks)
    rows = [deepcopy(dict(row)) for row in capacity_rows]
    artifact.update(
        {
            "status": "blocked_precondition",
            "started_at_utc": started_at,
            "completed_at_utc": started_at,
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "duration_s": duration_s,
            "resource_audit_duration_s": duration_s,
            "rows": rows,
            "capacity_rows": rows,
            "inference_substrate_details": {"observed_owner_state": deepcopy(list(owner_state))},
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary['check']}",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _recompute_capacity_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[str]]:
    """Rebuild each raw capacity decision without trusting stored scalar fields."""

    rebuilt: list[JsonDict] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            reduced = reduce_capacity_row(
                str(row["fixture_id"]),
                list(row["process_rows"]),
                list(row["lease_observations"]),
                list(row["query_receipts"]),
                expected_query_passed=bool(row["expected_query_passed"]),
                expected_capacity_passed=bool(row["expected_capacity_passed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"capacity_row_invalid:{index}:{type(exc).__name__}:{exc}")
            continue
        rebuilt.append(reduced)
        if reduced != dict(row):
            errors.append(f"capacity_row_mismatch:{index}")
    return rebuilt, errors


def _raw_row_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Reload exact raw row files so embedded evidence cannot replace changed bytes."""

    manifest = artifact.get("raw_row_manifest")
    if not isinstance(manifest, list):
        return ["raw_row_manifest_invalid"]
    errors: list[str] = []
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping):
            errors.append(f"raw_row_reference_invalid:{index}")
            continue
        path = Path(str(row.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            errors.append(f"raw_row_path_missing:{index}")
            continue
        if sha256_file(resolved) != row.get("sha256"):
            errors.append(f"raw_row_hash_mismatch:{index}")
            continue
        payload = load_object(resolved)
        field = str(row.get("field") or "")
        if field not in {"capacity_rows", "lease_rows"}:
            errors.append(f"raw_row_field_invalid:{index}")
        elif payload.get(field) != artifact.get(field):
            errors.append(f"raw_row_payload_mismatch:{index}")
    return errors


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Recompute capacity, lease lifecycle, current counts, and readiness."""

    raw_errors = _raw_row_errors(artifact, root)
    capacity = artifact.get("capacity_rows")
    if not isinstance(capacity, list):
        return [*raw_errors, "capacity_rows_invalid"]
    leases = artifact.get("lease_rows")
    if not isinstance(leases, list):
        return [*raw_errors, "lease_rows_invalid"]
    rebuilt, errors = _recompute_capacity_rows(capacity)
    errors = [*raw_errors, *errors]
    if errors:
        errors.append("capacity_rows_mismatch")
    private_expected = build_private_capacity_rows()
    if rebuilt[: len(private_expected)] != private_expected:
        errors.append("private_capacity_fixtures_mismatch")
    if not _lease_lifecycle_passed(leases):
        errors.append("lease_lifecycle_not_ready")
    events = artifact.get("current_invocation_events")
    if not isinstance(events, list):
        errors.append("current_invocation_events_invalid")
    else:
        counts, event_errors = current_work_receipt._reduce_events(
            events,
            run_id=str(artifact.get("current_run_id")),
            owner_pid=int(artifact.get("current_owner_pid", -1)),
        )
        errors.extend(event_errors)
        if counts != zero_counts() or artifact.get("invocation_counts") != counts:
            errors.append("invocation_counts_mismatch")
    receipts = artifact.get("validation_receipts")
    receipt_rows = list(receipts) if isinstance(receipts, list) else []
    gates = _acceptance_gates(
        rebuilt,
        leases,
        receipt_rows,
        preconditions_ok=all(
            row.get("passed") is True for row in artifact.get("preconditions_checked") or []
        ),
        flagged=artifact.get("flagged_adversarial") is True,
    )
    expected_ready = int(all(row["passed"] for row in gates))
    if artifact.get("runtime_ownership_ready_score") != expected_ready:
        errors.append("runtime_ownership_ready_score_mismatch")
    return list(dict.fromkeys(errors))


def validate_artifact(
    value: Any,
    *,
    require_terminal: bool = False,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Cold-check identity, no-model provenance, reduction, receipts, and hash."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"missing_required_field:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_invalid")
    if value.get("MODEL_SPECS") != []:
        errors.append("model_specs_invalid")
    if value.get("model_invoked") is not False:
        errors.append("model_invoked_invalid")
    if value.get("invocation_counts") != zero_counts():
        errors.append("invocation_counts_invalid")
    if value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if value.get("random_seed") is not None:
        errors.append("random_seed_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    if value.get("small_ebm_training") != {"performed": False}:
        errors.append("small_ebm_training_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    honest = str(value.get("honest_verdict") or "")
    if value.get("verdict_class") in {"null", "positive", "circular_positive", "disqualified"}:
        if not (honest.startswith("complete_") or honest.startswith("complete:")):
            errors.append("honest_verdict_prefix_invalid")
    if value.get("verdict_class") == "blocked" and not honest.startswith("blocked_"):
        errors.append("blocked_verdict_prefix_invalid")
    if value.get("verdict_class") != "blocked":
        errors.extend(independent_reduce_artifact(value, root=root))
    current_errors = current_work_receipt.validate_current_work_receipt(value, root=root)
    errors.extend(current_errors)
    if require_terminal and not receipts_pass(
        list(value.get("validation_receipts") or []),
        (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES),
    ):
        errors.append("required_validation_receipts_invalid")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:  # pragma: no cover - measured orchestration only.
    """Close one real phase with its monotonic boundaries and checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint_reference": checkpoint,
        "progress_events_flushed": True,
    }


def _fresh_child_readback(
    root: Path,
    lease: lease_api.GpuLease,
    raw_dir: Path,
    started: float,
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - real fresh process boundary.
    """Have a fresh child authenticate the owner's journal without mutating it."""

    code = (
        "import json,sys,time;"
        "from pathlib import Path;"
        "from carnot import gpu_lease_phase_journal as g;"
        "p=Path(sys.argv[1]);e=json.loads(sys.argv[2]);d=g.read_journal(p);"
        "x=g.validate_journal_document(d,expected_pid=e['pid'],"
        "expected_pid_start_ticks=e['pid_start_ticks'],"
        "expected_device_uuid=e['device_uuid'],expected_model=e['expected_model']);"
        "o={'lease_id':d.get('lease_id'),'task_id':d.get('task_id'),"
        "'pid':d.get('owner',{}).get('pid'),"
        "'pid_start_ticks':d.get('owner',{}).get('pid_start_ticks'),"
        "'device_uuid':d.get('device_uuid'),'errors':x,"
        "'owner_process_matches':g.process_start_matches(e['pid'],e['pid_start_ticks']),"
        "'read_monotonic_ns':time.monotonic_ns()};"
        "print(json.dumps(o,sort_keys=True),flush=True);raise SystemExit(bool(x))"
    )
    expected = lease.owner_receipt()
    command = CommandSpec(
        "fresh_child_lease_readback",
        (
            str(root / ".venv/bin/python"),
            "-u",
            "-c",
            code,
            str(lease.journal_path),
            json.dumps(
                {
                    "pid": expected["pid"],
                    "pid_start_ticks": expected["pid_start_ticks"],
                    "device_uuid": expected["device_uuid"],
                    "expected_model": expected["expected_model"],
                },
                sort_keys=True,
            ),
        ),
        "owned_lease_identity",
        30.0,
    )
    progress(started, "ownership", "before_subprocess", operation=command.name)
    receipt = run_commands(
        root,
        [command],
        log_dir=raw_dir / "child_readback",
        heartbeat_s=60.0,
    )[0]
    progress(
        started,
        "ownership",
        "after_subprocess",
        operation=command.name,
        exit_code=receipt["exit_code"],
    )
    observed: JsonDict = {}
    for line in reversed(str(receipt.get("output_tail") or "").splitlines()):
        if line.lstrip().startswith("{"):
            observed = json.loads(line)
            break
    passed = bool(
        receipt.get("passed") is True
        and observed.get("errors") == []
        and observed.get("owner_process_matches") is True
        and observed.get("lease_id") == lease.lease_id
        and observed.get("task_id") == TASK_ID
        and observed.get("pid") == lease.pid
        and observed.get("pid_start_ticks") == lease.pid_start_ticks
        and observed.get("device_uuid") == lease.device_uuid
    )
    return observed, {**receipt, "identity_passed": passed}


def _observe_host_capacity(  # pragma: no cover - current NVIDIA and procfs boundary.
    started: float,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Use the shipped readers to capture one read-only host capacity row."""

    progress(started, "capacity", "before_subprocess", operation="gpu_inventory")
    process_rows, query_receipts = canary.runtime.lease_preflight.collect_gpu_process_rows()
    lease_rows = canary.runtime.lease_preflight.scan_lease_rows(
        canary.runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = canary.runtime.lease_preflight.classify_process_rows(
        process_rows,
        lease_rows,
        current_task_id=TASK_ID,
    )
    row = reduce_capacity_row("host", classified, lease_rows, query_receipts)
    progress(
        started,
        "capacity",
        "after_subprocess",
        operation="gpu_inventory",
        available=row["available_capacity"],
        query_ok=row["query_ok"],
    )
    return row, classified, lease_rows


def _acquire_current_lease(
    root: Path,
    model_path: str,
    raw_dir: Path,
    started: float,
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - current host ownership boundary.
    """Wait at most 120 seconds, acquire one available device, and release it."""

    deadline = time.monotonic() + LEASE_WAIT_TIMEOUT_S
    race_rows: list[JsonDict] = []
    latest_host: JsonDict = {}
    while True:
        latest_host, classified, _lease_rows = _observe_host_capacity(started)
        for device_uuid in latest_host["available_gpu_uuids"]:
            sample = next(row for row in classified if row.get("gpu_uuid") == device_uuid)
            progress(
                started,
                "ownership",
                "before_acquire",
                device_uuid=device_uuid,
                completed_attempts=len(race_rows),
            )
            try:
                lease = lease_api.GpuLease.acquire(
                    runtime_dir=canary.runtime.lease_preflight.LEASE_RUNTIME_DIR,
                    task_id=TASK_ID,
                    device_uuid=device_uuid,
                    expected_model=model_path,
                    vram_before_mb=int(sample.get("gpu_memory_used_mb", 0) or 0),
                    ttl_s=LEASE_WAIT_TIMEOUT_S,
                )
            except (lease_api.LeaseBusy, lease_api.RecoveryError, lease_api.JournalError) as exc:
                race_rows.append(
                    {
                        "device_uuid": device_uuid,
                        "error": f"{type(exc).__name__}:{exc}",
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                progress(
                    started,
                    "ownership",
                    "acquisition_race",
                    device_uuid=device_uuid,
                    error=type(exc).__name__,
                )
                continue
            progress(started, "ownership", "after_acquire", lease_id=lease.lease_id)
            owner = lease.owner_receipt()
            child, child_receipt = _fresh_child_readback(root, lease, raw_dir, started)
            progress(started, "ownership", "before_release", lease_id=lease.lease_id)
            release = complete_no_model_lease(lease)
            progress(
                started,
                "ownership",
                "after_release",
                lease_id=lease.lease_id,
                released=release["released"],
            )
            row = {
                "row_kind": "lease_lifecycle",
                "device_uuid": device_uuid,
                "lease_id": owner["lease_id"],
                "task_id": owner["task_id"],
                "owner_pid": owner["pid"],
                "owner_pid_start_ticks": owner["pid_start_ticks"],
                "acquired_monotonic_ns": owner["acquired_monotonic_ns"],
                "child_readback_monotonic_ns": child.get("read_monotonic_ns"),
                "released_monotonic_ns": release.get("released_monotonic_ns"),
                "fresh_child_readback_passed": child_receipt["identity_passed"],
                "fresh_child_readback": child,
                "fresh_child_command_receipt": child_receipt,
                "release_passed": release["release_passed"],
                "release_duration_s": release["release_duration_s"],
                "terminal_phase": release["terminal_phase"],
                "signals_sent": release["signals_sent"],
                "recovery": owner["recovery"],
                "acquisition_races": race_rows,
            }
            return row, latest_host
        now = time.monotonic()
        if now >= deadline:
            return {
                "row_kind": "lease_lifecycle",
                "device_uuid": None,
                "lease_id": None,
                "task_id": TASK_ID,
                "owner_pid": os.getpid(),
                "owner_pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "fresh_child_readback_passed": False,
                "release_passed": False,
                "terminal_phase": None,
                "signals_sent": [],
                "error": "lease_wait_timeout",
                "acquisition_races": race_rows,
            }, latest_host
        progress(
            started,
            "ownership",
            "lease_wait_outstanding",
            remaining_s=round(deadline - now, 3),
            completed_attempts=len(race_rows),
        )
        time.sleep(min(1.0, max(0.0, deadline - now)))


def _source_hashes(
    root: Path,
    context: Mapping[str, Any],
    raw_manifest: Sequence[Mapping[str, Any]],
    sidecar: Mapping[str, Any],
) -> JsonDict:
    """Bind current code, historical bytes, model path metadata, and raw rows."""

    hashes = deepcopy(dict(context.get("source_hashes") or {}))
    hashes["model_file_metadata"] = deepcopy(dict(context["model_path_prerequisite"]))
    hashes["historical_receipt_sidecar"] = deepcopy(dict(sidecar))
    hashes["raw_rows"] = [deepcopy(dict(row)) for row in raw_manifest]
    for path in (MODULE_PATH, SHARED_MODULE_PATH, WRAPPER_PATH, TEST_PATH, SHARED_TEST_PATH):
        hashes[path.as_posix()] = sha256_file(root / path)
    return hashes


def _raw_manifest(root: Path, raw_dir: Path, rows: Mapping[str, Any]) -> list[JsonDict]:
    """Persist raw capacity and lease rows outside the terminal publication path."""

    manifest: list[JsonDict] = []
    for name, value in rows.items():
        path = raw_dir / f"{name}.json"
        atomic_json(path, {name: deepcopy(value)})
        manifest.append(
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "field": name}
        )
    return manifest


def _run_affected_validation(
    root: Path,
    raw_dir: Path,
    started: float,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - bounded subprocess plan.
    """Run the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7422-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    errors = validate_validation_plan(root, commands)
    progress(started, "validation", "before_affected_subprocesses", plan_errors=len(errors))
    receipts: list[JsonDict] = []
    if not errors:
        receipts = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
            heartbeat_s=60.0,
        )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    if errors:
        reduced["plan_errors"] = errors
        reduced["passed"] = False
    progress(started, "validation", "after_affected_subprocesses", passed=reduced["passed"])
    return receipts, reduced


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7422_v651_runtime_ownership import independent_reduce_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=independent_reduce_artifact(v);print(json.dumps({'errors':e}),flush=True);"
        "raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


def run_experiment(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | None = None,
) -> JsonDict:  # pragma: no cover - live no-model orchestration.
    """Authenticate, acquire one real lease, validate, and publish atomically."""

    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR / f"attempt-{started_ns}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start", completed_units=0)
    checks, context = collect_preconditions(root)
    checks.insert(
        0, gate_row("run_date", "execution_contract", "run_date", "==", RUN_DATE, run_date)
    )
    spans.append(
        _span("preconditions", phase_started, started, len(checks), "preconditions_checked")
    )
    progress(
        started,
        "preconditions",
        "end",
        completed_units=len(checks),
        passed=all(row["passed"] for row in checks),
    )
    if not all(row["passed"] for row in checks):
        blocked = build_blocked_artifact(
            checks,
            capacity_rows=build_private_capacity_rows(),
            started_at=started_at,
            duration_s=time.monotonic() - started,
        )
        blocked["phase_spans"] = spans
        blocked["completed_at_utc"] = utc_now()
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        progress(started, "write", "before_atomic_publish", status=blocked["status"])
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_publish", status=blocked["status"])
        return blocked

    sidecar = current_work_receipt.write_immutable_sidecar(
        raw_dir / "historical_exp7416_receipt.json",
        scope="historical_model_receipts",
        payload={
            "artifact_path": HISTORICAL_PATH.as_posix(),
            "artifact_sha256": sha256_file(root / HISTORICAL_PATH),
            "original_status": context["historical"].get("status"),
            "original_honest_verdict": context["historical"].get("honest_verdict"),
            "original_model_invoked": context["historical"].get("model_invoked"),
            "authorizes_current_invocation_counts": False,
        },
        root=root,
    )

    phase_started = time.monotonic()
    progress(started, "capacity", "start", completed_units=0)
    capacity_rows = build_private_capacity_rows()
    model_path = str(context["model_path_prerequisite"]["path"])
    lease_row, host_row = _acquire_current_lease(root, model_path, raw_dir, started)
    capacity_rows.append(host_row)
    lease_rows = [lease_row]
    spans.append(
        _span("resource_audit", phase_started, started, len(capacity_rows) + 1, "lease_rows")
    )
    resource_duration = time.monotonic() - phase_started
    progress(
        started,
        "capacity",
        "end",
        completed_units=len(capacity_rows) + 1,
        lease_ready=_lease_lifecycle_passed(lease_rows),
    )
    raw_manifest = _raw_manifest(
        root,
        raw_dir,
        {"capacity_rows": capacity_rows, "lease_rows": lease_rows},
    )

    validation_started = time.monotonic()
    phase_started = validation_started
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    spans.append(
        _span(
            "affected_validation", phase_started, started, len(affected_receipts), "affected_logs"
        )
    )

    ended_ns = time.monotonic_ns() - started_ns
    current = build_no_model_current_receipt(
        0,
        ended_ns,
        [sidecar],
        phase_spans=spans,
    )
    preterminal_gates = _acceptance_gates(
        capacity_rows,
        lease_rows,
        affected_receipts,
        preconditions_ok=True,
        flagged=False,
    )
    artifact = _base_artifact()
    all_rows = [deepcopy(dict(row)) for row in (*capacity_rows, *lease_rows)]
    artifact.update(
        {
            "status": "complete_candidate_awaiting_terminal_readers",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            **current,
            "duration_s": ended_ns / 1_000_000_000,
            "resource_audit_duration_s": resource_duration,
            "validation_duration_s": time.monotonic() - validation_started,
            "cold_start_duration_s": 0.0,
            "phase_spans": spans,
            "model_file_metadata": deepcopy(context["model_path_prerequisite"]),
            "rows": all_rows,
            "raw_row_manifest": raw_manifest,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "planned": len(all_rows),
                "attempted": len(all_rows),
                "completed": len(all_rows),
                "failed": int(not _lease_lifecycle_passed(lease_rows)),
                "unstarted": 0,
                "independent_groups": len(all_rows),
            },
            "acceptance_gate_results": preterminal_gates,
            "gate_check_summary": gate_check_summary([*checks, *preterminal_gates]),
            "honest_verdict": "complete_null_runtime_ownership_awaiting_terminal_readers",
            "verdict_class": "null",
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected["passed"] else "required_checks_failed",
                "unrelated_known_failures": [],
                "affects_required_checks": not affected["passed"],
                "affected_reduction": affected,
            },
            "runtime_ownership_ready_score": 0,
            "capacity_rows": [deepcopy(dict(row)) for row in capacity_rows],
            "lease_rows": [deepcopy(dict(row)) for row in lease_rows],
        }
    )
    artifact["source_artifact_hashes"] = _source_hashes(root, context, raw_manifest, sidecar)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate, artifact)

    phase_started = time.monotonic()
    progress(started, "validation", "before_terminal_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                command,
                "safety" if command.name == "adversarial_verify" else "completion",
                True,
            )
            for command in _terminal_commands(root, candidate)
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    cold_duration = time.monotonic() - phase_started
    spans.append(
        _span(
            "terminal_validation", phase_started, started, len(terminal_receipts), "terminal_logs"
        )
    )
    progress(
        started,
        "validation",
        "after_terminal_subprocesses",
        completed_units=len(terminal_receipts),
    )

    receipts = [*affected_receipts, *terminal_receipts]
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    gates = _acceptance_gates(
        capacity_rows,
        lease_rows,
        receipts,
        preconditions_ok=True,
        flagged=flagged,
    )
    ready = int(all(row["passed"] for row in gates))
    affected_ok = receipts_pass(affected_receipts, REQUIRED_CHECK_NAMES)
    terminal_ok = receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    final_elapsed_ns = time.monotonic_ns() - started_ns
    final_current = build_no_model_current_receipt(
        0,
        final_elapsed_ns,
        [sidecar],
        phase_spans=spans,
    )
    artifact.update(
        {
            **final_current,
            "status": (
                "complete_runtime_ownership_ready"
                if ready
                else "complete_required_validation_failed"
            ),
            "completed_at_utc": utc_now(),
            "duration_s": final_elapsed_ns / 1_000_000_000,
            "validation_duration_s": time.monotonic() - validation_started,
            "cold_start_duration_s": cold_duration,
            "phase_spans": spans,
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary([*checks, *gates]),
            "honest_verdict": (
                "complete_null_runtime_ownership_ready_no_reservation"
                if ready
                else "complete_disqualified_runtime_ownership_required_check_failed"
            ),
            "verdict_class": "null" if ready else "disqualified",
            "flagged_adversarial": flagged,
            "validation_receipts": receipts,
            "repository_health": {
                "status": "healthy" if affected_ok and terminal_ok else "required_checks_failed",
                "unrelated_known_failures": [],
                "affects_required_checks": not (affected_ok and terminal_ok),
                "affected_reduction": affected,
            },
            "runtime_ownership_ready_score": ready,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, require_terminal=True, root=root)
    if errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "verdict_class": "disqualified",
                "flagged_adversarial": True,
                "runtime_ownership_ready_score": 0,
                "internal_validation_errors": errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(started, "write", "before_atomic_publish", status=artifact["status"])
    atomic_json(output, artifact)
    progress(
        started,
        "write",
        "after_atomic_publish",
        status=artifact["status"],
        completed_units=len(all_rows),
    )
    return artifact


def _date_argument(value: str) -> str:
    """Reject execution outside the fixed V651 date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the live audit or cold-validate one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = load_object(args.validate)
        errors = [
            *validate_artifact(value, require_terminal=False, root=REPO_ROOT),
            *independent_reduce_artifact(value, root=REPO_ROOT),
        ]
        errors = list(dict.fromkeys(errors))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "runtime_ownership_ready_score": result["runtime_ownership_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
